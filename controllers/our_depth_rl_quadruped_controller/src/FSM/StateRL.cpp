//
// Created by jialong on 25-9-2.
//

// 本文件实现 RL 控制状态（StateRL）
//
// 功能概述：
// - 每个控制周期完成：读取状态 -> 构建观测 -> 策略推理 -> 映射动作 -> 写入指令
// - 观测包含：IMU、遥控、关节位置/速度、上一时刻动作、足接触、可选深度特征与历史
// - 策略为 TorchScript 导出模型；可选加载深度编码器以融合环境几何信息
//
// 数据流：
// - state_interface 读入 -> robot_state_/control_ 缓存
// - 观测拼接（proprio + depth_latent + history） -> policy.forward()
// - 策略输出 actions 经裁剪/缩放/EMA 低通 -> 目标关节角 output_dof_pos_
// - robot_command_ 写入 command_interface（位置/速度/KP/KD/力矩）
//
// 线程模型：
// - 主线程：getState() +（可选）runModel() + setCommand()
// - 独立 RL 线程（use_rl_thread_）：按 decimation 降频执行 runModel()，降低主循环算力占用
//
// 重要参数：
// - action_scale/action_filter_alpha：动作缩放与指数滑动平均系数（抑制抖动）
// - rl_kp/rl_kd/torque_limits：关节增益与力矩上限
// - clip_actions_[upper/lower]：动作裁剪边界，防止异常输出
// - use_camera/use_depth_cnn/max_depth 等：深度特征与预处理参数
//
// 与训练对齐（legged_robot.py）：
// - proprio 53 维顺序、dof/feet reindex、obs_scales、clip_actions/action_scale 与 compute_observations/_compute_torques 一致
// - 接触：contact_filt = contact|last_contacts，阈值与训练 norm(contact_forces)>2 对应（config: contact_use_last=true, foot_force_threshold=2）
// - 历史：本步 policy 使用更新前的 history [t-9..t-1]，再 insert 当前步；首步全填当前 proprio

#include "our_depth_rl_quadruped_controller/FSM/StateRL.h"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/logging.hpp>
#include <fstream>
#include <sensor_msgs/image_encodings.hpp>
#include "std_msgs/msg/float32_multi_array.hpp"
#include <yaml-cpp/yaml.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>  
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <torch/nn/functional.h>
#include <chrono>

template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node& node)
{
    // 从 YAML 列表节点读取一维向量（不做转置/重排）
    std::vector<T> values;
    for (const auto& val : node)
    {
        values.push_back(val.as<T>());
    }
    return values;
}

template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node& node, const std::string& framework, const int& rows,
                                  const int& cols)
{
    // 从 YAML 读取一维向量，并在需要时按训练框架约定进行矩阵“转置展开”
    // - isaacgym：按行主序直接读取
    // - isaacsim：训练时可能使用列主序/不同排布，这里通过 rows/cols 做一次转置展开
    std::vector<T> values;
    for (const auto& val : node)
    {
        values.push_back(val.as<T>());
    }

    if (framework == "isaacsim")
    {
        std::vector<T> transposed_values(cols * rows);
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                transposed_values[c * rows + r] = values[r * cols + c];
            }
        }
        return transposed_values;
    }
    if (framework == "isaacgym")
    {
        return values;
    }
    RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "ReadVectorFromYaml: Unsupported framework: %s", framework.c_str());
    throw std::invalid_argument("Unsupported framework: " + framework);
}

StateRL::StateRL(CtrlInterfaces& ctrl_interfaces,
                 CtrlComponent& ctrl_component,
                 const std::vector<double>& target_pos) :
    FSMState(FSMStateName::RL, "rl", ctrl_interfaces),
    node_(ctrl_component.node_),
    enable_estimator_(ctrl_component.enable_estimator_),
    estimator_(ctrl_component.estimator_),
    use_rl_thread_(false),
    running_(false),
    model_loaded_(false)
{
    // 构造阶段做“资源初始化”，包括：
    // - 读取 ROS 参数（robot_pkg/model_folder/use_rl_thread/use_camera）
    // - 读取 config.yaml（策略/观测/缩放/相机配置等）
    // - 注册相机订阅（可选）
    // - 初始化历史观测缓存
    // - 加载 TorchScript 模型（策略网络 + 可选深度编码器）
    // - 若启用独立线程，则启动 RL 推理线程
    RCLCPP_INFO(node_->get_logger(), "StateRL constructor started");

    if (!node_->has_parameter("robot_pkg")) {
        node_->declare_parameter("robot_pkg", robot_pkg_);
    }
    if (!node_->has_parameter("model_folder")) {
        node_->declare_parameter("model_folder", model_folder_);
    }
    if (!node_->has_parameter("use_rl_thread")) {
        node_->declare_parameter("use_rl_thread", use_rl_thread_);
    }
    if (!node_->has_parameter("config_folder")) {
        node_->declare_parameter("config_folder", std::string());
    }

    robot_pkg_ = node_->get_parameter("robot_pkg").as_string();
    model_folder_ = node_->get_parameter("model_folder").as_string();
    use_rl_thread_ = node_->get_parameter("use_rl_thread").as_bool();
    config_folder_ = node_->get_parameter("config_folder").as_string();
    params_.use_camera = node_->get_parameter("use_camera").as_bool();
    RCLCPP_INFO(node_->get_logger(), "Loaded parameters - robot_pkg: %s, model_folder: %s, use_rl_thread: %s, use_camera: %s",
               robot_pkg_.c_str(), model_folder_.c_str(), use_rl_thread_ ? "true" : "false", params_.use_camera ? "true" : "false");

    RCLCPP_INFO(node_->get_logger(), "Using robot model from %s", robot_pkg_.c_str());
    const std::string package_share_directory = ament_index_cpp::get_package_share_directory(robot_pkg_);
    std::string model_path = package_share_directory + "/config/" + model_folder_;
    if (!config_folder_.empty()) {
        model_path = config_folder_;
    }
    
    // target_pos 是初始/默认姿态（12 个关节），用于对齐训练时的 default_dof_pos
    if (target_pos.size() != 12) {
        RCLCPP_FATAL(node_->get_logger(), "StateRL requires 12 target positions, got %zu", target_pos.size());
        throw std::invalid_argument("Invalid target_pos size");
    }
    
    for (int i = 0; i < 12; i++)
    {
        init_pos_[i] = target_pos[i];
    }

    // 从 YAML 加载策略侧配置（包括模型文件名、观测布局、缩放系数、KP/KD、动作范围、相机参数等）
    loadYaml(model_path);

    // 如果启用相机：注册深度与 RGB 订阅，并把深度预处理后的张量缓存起来供推理使用
    if(params_.use_camera){
        // 深度图像发布器
        depth_image_pub_ = node_->create_publisher<sensor_msgs::msg::Image>("/depth_image/processed", 10);
        
        // RGB图像发布器
        rgb_image_pub_ = node_->create_publisher<sensor_msgs::msg::Image>("/rgb_image/processed", 10);
        
        // 深度图像订阅器
        depth_image_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
            "/rgbd_d435/depth_image", 10,
            [this](const sensor_msgs::msg::Image::SharedPtr msg)
            {
                try {
                    if (!msg) {
                        return;
                    }
                    if (msg->data.empty() || msg->height == 0 || msg->width == 0) {
                        return;
                    }

                    cv::Mat depth_m;
                    if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1 || msg->encoding == "32FC1") {
                        depth_m = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1)->image;
                    } else if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1 || msg->encoding == "16UC1") {
                        cv::Mat depth_mm = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
                        depth_mm.convertTo(depth_m, CV_32F, 0.001);
                    } else {
                        RCLCPP_WARN(rclcpp::get_logger("StateRL"), "Unsupported depth image encoding: %s", msg->encoding.c_str());
                        return;
                    }
                    if (depth_m.empty()) {
                        return;
                    }

                    (void)preprocessDepthImage(msg);

                    const float far_clip = std::max(1e-6f, static_cast<float>(params_.max_depth));
                    cv::Mat invalid_mask_raw = depth_m <= 1e-6f;
                    const int invalid_count_raw = cv::countNonZero(invalid_mask_raw);
                    const int total_count_raw = depth_m.rows * depth_m.cols;
                    const float invalid_ratio_raw = total_count_raw > 0
                                                   ? static_cast<float>(invalid_count_raw) / static_cast<float>(total_count_raw)
                                                   : 1.0f;
                    // 深度帧几乎全无效时不更新可视化，避免“黑屏”覆盖上一帧
                    if (invalid_ratio_raw > 0.98f)
                    {
                        RCLCPP_WARN_THROTTLE(
                            rclcpp::get_logger("StateRL"), *node_->get_clock(), 1000,
                            "Depth frame mostly invalid (%.1f%%), skip visualization publish.",
                            invalid_ratio_raw * 100.0f);
                        return;
                    }

                    cv::Mat depth_vis = depth_m.clone();
                    cv::medianBlur(depth_vis, depth_vis, 5);
                    cv::patchNaNs(depth_vis, far_clip);
                    cv::max(depth_vis, 0.0, depth_vis);
                    cv::min(depth_vis, far_clip, depth_vis);
                    cv::Mat invalid_mask = depth_vis <= 1e-6f;
                    // 可视化里把无效值设为 far_clip（白色），便于观察有效区域
                    depth_vis.setTo(far_clip, invalid_mask);

                    double min_val = 0.0;
                    double max_val = 0.0;
                    cv::minMaxLoc(depth_vis, &min_val, &max_val);
                    const int invalid_count = cv::countNonZero(invalid_mask);
                    const int total_count = depth_vis.rows * depth_vis.cols;
                    RCLCPP_DEBUG_THROTTLE(
                        rclcpp::get_logger("StateRL"), *node_->get_clock(), 2000,
                        "Depth vis stats: min=%.3f max=%.3f far=%.3f invalid=%d/%d enc=%s",
                        min_val, max_val, far_clip, invalid_count, total_count, msg->encoding.c_str());

                    cv_bridge::CvImage img_msg;
                    img_msg.header = msg->header;
                    img_msg.encoding = sensor_msgs::image_encodings::MONO8;
                    cv::Mat depth_u8;
                    depth_vis.convertTo(depth_u8, CV_8U, 255.0f / far_clip);
                    img_msg.image = depth_u8;
                    depth_image_pub_->publish(*img_msg.toImageMsg());

                } catch (const cv_bridge::Exception& e) {
                    RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "cv_bridge异常: %s", e.what());
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "处理深度图像时出错: %s", e.what());
                }
            }
        );
        
        // RGB图像订阅器
        rgb_image_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
            "/rgbd_d435/image", 10,
            [this](const sensor_msgs::msg::Image::SharedPtr msg)
            {
                try {
                    std::lock_guard<std::mutex> lock(rgb_mutex_);
                    // RGB 图像仅用于可视化：直接转 OpenCV 后转回 ROS 消息发布
                    rgb_image_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
                    
                    // 发布RGB图像用于直接显示
                    cv_bridge::CvImage img_msg;
                    img_msg.header = msg->header;
                    img_msg.encoding = sensor_msgs::image_encodings::BGR8;
                    img_msg.image = rgb_image_;
                    rgb_image_pub_->publish(*img_msg.toImageMsg());
                    
                } catch (const cv_bridge::Exception& e) {
                    RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "cv_bridge异常: %s", e.what());
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "处理RGB图像时出错: %s", e.what());
                }
            }
        );
        
        RCLCPP_INFO(node_->get_logger(), "Depth and RGB camera subscriptions initialized");
    }

    if (params_.publish_contact_states)
    {
        contact_states_pub_ = node_->create_publisher<std_msgs::msg::Float32MultiArray>(
            params_.contact_states_topic, 10);
        RCLCPP_DEBUG(node_->get_logger(), "Contact states publisher: %s", params_.contact_states_topic.c_str());
    }
    if (params_.publish_foot_force_debug)
    {
        foot_force_debug_pub_ = node_->create_publisher<std_msgs::msg::Float32MultiArray>(
            params_.foot_force_debug_topic, 10);
        RCLCPP_DEBUG(node_->get_logger(), "Foot force debug publisher: %s (data: [FL,FR,RL,RR,threshold])",
            params_.foot_force_debug_topic.c_str());
    }
    if (params_.publish_proprio)
    {
        proprio_pub_ = node_->create_publisher<std_msgs::msg::Float32MultiArray>(
            params_.proprio_topic, 10);
        RCLCPP_DEBUG(node_->get_logger(), "Proprio publisher: %s", params_.proprio_topic.c_str());
    }
    if (params_.publish_yaw_diff)
    {
        yaw_diff_pub_ = node_->create_publisher<std_msgs::msg::Float32MultiArray>(
            params_.yaw_diff_topic, 10);
        RCLCPP_DEBUG(node_->get_logger(), "Yaw diff publisher: %s (data: [manual_yaw, policy_yaw, diff])",
            params_.yaw_diff_topic.c_str());
    }
    if (params_.publish_history_order_debug && params_.num_hist_len > 0)
    {
        history_order_debug_pub_ = node_->create_publisher<std_msgs::msg::Float32MultiArray>(
            params_.history_order_debug_topic, 10);
        RCLCPP_INFO(node_->get_logger(),
            "观测历史顺序调试: 发布 %d 段均值到 %s (index 0=最旧, %d=最新)",
            params_.num_hist_len, params_.history_order_debug_topic.c_str(), params_.num_hist_len - 1);
    }
    if (params_.publish_obs_freq_debug)
    {
        obs_freq_debug_pub_ = node_->create_publisher<std_msgs::msg::Float32MultiArray>(
            params_.obs_freq_debug_topic, 10);
        RCLCPP_INFO(node_->get_logger(),
            "观测频率调试: 发布 [实际步长ms, 期望步长ms] 到 %s（与训练 sim.dt*decimation 一致应为 %.1f ms）",
            params_.obs_freq_debug_topic.c_str(), params_.control_step_ms);
    }
    // 读取观测维度，默认53
    const int history_obs_dim = params_.num_proprio > 0 ? params_.num_proprio : 53;
    // 读取历史观测长度
    const int history_steps = params_.num_hist_len > 0 ? params_.num_hist_len : static_cast<int>(params_.observations_history.size());
    // 历史缓存用于“history encoding”：把过去若干步 proprio 观测存起来，供策略使用
    history_obs_buf_ = std::make_shared<ObservationBuffer>(1, history_obs_dim,
                                                           std::max(history_steps, 1));
    RCLCPP_INFO(node_->get_logger(), "历史观测缓冲区初始化：维度=%d, 历史窗口大小=%d",
                history_obs_dim, std::max(history_steps, 1));

    RCLCPP_INFO(node_->get_logger(), "Model loading: %s", params_.model_name.c_str());
    
    // 策略模型加载：优先从 robot_pkg 的 share/config/model_folder 找，
    // 再尝试开发目录/安装目录等备选路径，方便调试不同导出位置的 jit 文件。
    std::vector<std::string> possible_model_paths;
    possible_model_paths.push_back(model_path + "/" + params_.model_name); // 默认路径
    
    bool model_loaded = false;
    std::string loaded_path;
    
    for (const auto& path : possible_model_paths) {
        RCLCPP_INFO(node_->get_logger(), "尝试加载模型: %s", path.c_str());
        try {
            policy_module_ = torch::jit::load(path);
            policy_module_.to(device_);
            policy_module_.eval();
            model_loaded = true;
            loaded_path = path;
            model_loaded_ = true;
            break;
        } catch (const std::exception& e) {
            RCLCPP_WARN(node_->get_logger(), "模型加载失败: %s (%s)", path.c_str(), e.what());
        }
    }
    
    if (!model_loaded) {
        RCLCPP_ERROR(node_->get_logger(), "所有可能的模型路径都加载失败！请检查模型文件是否存在");
        throw std::runtime_error("Failed to load model from any possible path");
    }
    
    RCLCPP_INFO(node_->get_logger(), "Successfully loaded model from %s", loaded_path.c_str());

    // 可选：加载 onboard 模型（封装 estimator + history_encoder + actor_backbone）
    if (params_.use_onboard_actor_backbone)
    {
        const std::string onboard_path = model_path + "/" + params_.onboard_model_name;
        try
        {
            onboard_policy_module_ = torch::jit::load(onboard_path);
            onboard_policy_module_.to(device_);
            onboard_policy_module_.eval();
            onboard_model_loaded_ = true;
            use_onboard_actor_backbone_ = true;
            RCLCPP_DEBUG(node_->get_logger(), "Loaded onboard policy from %s", onboard_path.c_str());
        }
        catch (const std::exception& e)
        {
            onboard_model_loaded_ = false;
            use_onboard_actor_backbone_ = false;
            RCLCPP_WARN(node_->get_logger(),
                        "Failed to load onboard policy %s, fallback to policy forward: %s",
                        onboard_path.c_str(), e.what());
        }
    }

    if (use_rl_thread_)
    {
        // 独立 RL 线程：按 (frequency / decimation) 的频率执行推理
        // 主 update() 线程仍然读取状态并写命令，从而减少抖动。
        rl_thread_ = std::thread([&]{
            while (true)
            {
                try
                {
                    // lambda + 控制频率
                    executeAndSleep(
                        [&]
                        {
                            if (running_)
                            {
                                // 运行模型并计算控制指令
                                runModel();
                            }
                        },
                        ctrl_interfaces_.frequency_ / params_.decimation);
                }
                catch (const std::exception& e)
                {
                    running_ = false;
                    RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "Error in RL thread: %s", e.what());
                }
            }
        });
        // 性能实时性增强
        setThreadPriority(60, rl_thread_);
    }
}

void StateRL::enter()
{
    // 进入 RL 状态时初始化各类张量/缓存，确保后续拼接与推理不会因为维度错误而崩溃
    // 注意：这里也对一些维度做兜底，避免 YAML 缺字段导致负维度/零维度错误
    if (params_.num_feet <= 0 || params_.mass_params_dim <= 0 || params_.friction_dim <= 0 || params_.num_of_dofs <= 0) {
        RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "Invalid dimensions detected: num_feet=%d, mass_params_dim=%d, friction_dim=%d, num_of_dofs=%d",
                    params_.num_feet, params_.mass_params_dim, params_.friction_dim, params_.num_of_dofs);
        // 使用安全的默认值
        params_.num_feet = std::max(4, params_.num_feet);
        params_.mass_params_dim = std::max(4, params_.mass_params_dim);
        params_.friction_dim = std::max(1, params_.friction_dim);
        params_.num_of_dofs = std::max(12, params_.num_of_dofs);
        RCLCPP_WARN(rclcpp::get_logger("StateRL"), "Using safe default dimensions: num_feet=%d, mass_params_dim=%d, friction_dim=%d, num_of_dofs=%d", 
                    params_.num_feet, params_.mass_params_dim, params_.friction_dim, params_.num_of_dofs);
    }
    
    obs_.ang_vel = torch::tensor({{0.0, 0.0, 0.0}});
    obs_.commands = torch::tensor({{0.0, 0.0, 0.0}});
    obs_.dof_pos = params_.default_dof_pos;  // 关节位置 [1, K]（K为关节数，复用默认姿态）
    obs_.dof_vel = torch::zeros({1, params_.num_of_dofs});  // 关节速度 [1, K]
    obs_.actions = torch::zeros({1, params_.num_of_dofs});  // 上一步动作 [1, K]
    obs_.roll = torch::tensor(0.0).unsqueeze(0);  // 横滚角 [1]（标量，批量维度为1）
    obs_.pitch = torch::tensor(0.0).unsqueeze(0);  // 俯仰角 [1]
    obs_.delta_yaw = torch::tensor(0.0).unsqueeze(0);  // 偏航角偏差 [1]（标量）
    obs_.contact_states = torch::zeros({1, params_.num_feet});  // 足部接触状态 [1, F]（F为足数，如4）
    obs_.depth_latent = torch::zeros({1, params_.depth_feature_dim});
    last_contact_bool_ = torch::zeros(
        {1, params_.num_feet},
        torch::TensorOptions().dtype(torch::kBool).device(device_));
    last_depth_latent_ = torch::zeros({1, params_.depth_feature_dim});
    depth_update_counter_ = 0;
    latest_depth_tensor_ = torch::Tensor();

    // 输出缓存：关节目标/力矩等
    output_torques = torch::zeros({1, params_.num_of_dofs});
    output_dof_pos_ = params_.default_dof_pos;

    // 遥控命令清零
    control_.x = 0.0;
    control_.y = 0.0;
    control_.yaw = 0.0;

    for (int i = 0; i < 12; i++)
    {
        start_pos_[i] = ctrl_interfaces_.joint_position_state_interface_[i].get().get_value();
    }
    transition_percent_ = 0.0;
    transition_duration_ = std::max(1.0, ctrl_interfaces_.frequency_ * 1.0);

    // 历史观测缓存清空（避免切换状态时把旧历史带入策略）；下一帧将做「首步全填」
    if (!params_.observations_history.empty()) {
        history_obs_buf_->clear();
        history_steps_since_clear_ = 0;
    }

    running_ = true;
}

void StateRL::run(const rclcpp::Time&/*time*/, const rclcpp::Duration&/*period*/)
{
    // 1) 无论是否使用独立线程，都在主循环内读取最新 state（IMU/关节/遥控）
    getState();
    
    // 3) 若未启用独立 RL 线程，则在此处同步推理；否则推理在 rl_thread_ 里进行
    if (!use_rl_thread_)
    {
        runModel();
    }
    
    // 4) 将 robot_command_ 写入 ros2_control 的 command_interface
    setCommand();
}

void StateRL::exit()
{
    running_ = false;
}

FSMStateName StateRL::checkChange()
{
    // FSM 状态切换逻辑：
    // - 若启用 estimator 且安全检查失败，则立即切换到 FIXEDSTAND（比 PASSIVE 更不易瞬间瘫倒）
    // - 遥控 command=1/2 可切到 PASSIVE/FIXEDDOWN
    if (enable_estimator_ and !estimator_->safety())
    {
        // FIXEDSTAND will hold posture with PD gains, preventing sudden collapse.
        RCLCPP_WARN_THROTTLE(
            rclcpp::get_logger("StateRL"), *node_->get_clock(), 500,
            "Estimator safety check failed -> switching RL -> FIXEDSTAND for safety.");
        return FSMStateName::FIXEDSTAND;
    }
    switch (ctrl_interfaces_.control_inputs_.command)
    {
    case 1:
        return FSMStateName::PASSIVE;
    case 2:
        return FSMStateName::FIXEDDOWN;
    default:
        return FSMStateName::RL;
    }
}

torch::Tensor StateRL::computeObservation() {
    // 兼容父类接口：本实现把观测构建/推理/写命令集中在 runModel()，
    // 因此 computeObservation() 返回占位张量即可，避免上层误用。
    (void)params_;
    return torch::zeros({1, 1}, torch::TensorOptions().dtype(torch::kFloat32));
}

void StateRL::loadYaml(const std::string& config_path)
{
    // 从 `config_path/config.yaml` 读取策略参数。
    // 该 YAML 用于对齐训练时的约定：观测布局、缩放系数、动作范围、KP/KD、相机参数、模型文件名等。
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(config_path + "/config.yaml");
    }
    catch ([[maybe_unused]] YAML::BadFile& e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "配置文件不存在: %s/config.yaml", config_path.c_str());
        return;
    }
    catch (YAML::ParserException& e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "配置文件解析错误: %s", e.what());
        return;
    }

    // 模型文件名（通常是 TorchScript 导出的 *.pt）
    params_.model_name = config["model_name"].as<std::string>();

    // framework 用于处理部分张量在 YAML 里的排列方式，以及 IMU 四元数顺序（isaacgym/isaacsim）
    params_.framework = config["framework"].as<std::string>();
    const int rows = config["rows"].as<int>();
    const int cols = config["cols"].as<int>();

    if (config["observations_history"].IsNull())
    {
        params_.observations_history = {};
    }
    else
    {
        params_.observations_history = ReadVectorFromYaml<int>(config["observations_history"]);
    }
    params_.decimation = config["decimation"].as<int>();
    params_.num_observations = config["num_observations"].as<int>(); // 53或45
    params_.observations = ReadVectorFromYaml<std::string>(config["observations"]);
    params_.clip_obs = config["clip_obs"].as<double>();
    params_.clip_actions = config["clip_actions"].as<double>(1.2);
    // 动作裁剪上下限：用于防止策略输出异常大导致关节指令过激
    if (config["clip_actions_lower"].IsNull() && config["clip_actions_upper"].IsNull())
    {
        params_.clip_actions_upper = torch::tensor({}).view({1, -1});
        params_.clip_actions_lower = torch::tensor({}).view({1, -1});
    }
    else
    {
        auto upper_vec = ReadVectorFromYaml<double>(config["clip_actions_upper"], params_.framework, rows, cols);
        auto lower_vec = ReadVectorFromYaml<double>(config["clip_actions_lower"], params_.framework, rows, cols);

        params_.clip_actions_upper = torch::tensor(upper_vec, torch::kFloat32).view({1, -1});
        params_.clip_actions_lower = torch::tensor(lower_vec, torch::kFloat32).view({1, -1});
    }

    params_.action_scale = config["action_scale"].as<double>();
    params_.action_filter_alpha = config["action_filter_alpha"].as<double>(0.8);
    params_.hip_scale_reduction = config["hip_scale_reduction"].as<double>();
    params_.hip_scale_reduction_indices = ReadVectorFromYaml<int>(config["hip_scale_reduction_indices"]);
    params_.num_of_dofs = config["num_of_dofs"].as<int>();
    params_.lin_vel_scale = config["lin_vel_scale"].as<double>();
    params_.ang_vel_scale = config["ang_vel_scale"].as<double>();
    params_.dof_pos_scale = config["dof_pos_scale"].as<double>();
    params_.dof_vel_scale = config["dof_vel_scale"].as<double>();
    params_.dof_vel_filter_alpha = config["dof_vel_filter_alpha"].as<double>(0.0);

    // 维度参数：用于拼接观测、分配张量形状（需在 reindex 解析前设置）
    params_.num_feet = config["num_feet"].as<int>(4);  // 默认4足机器人
    params_.mass_params_dim = config["mass_params_dim"].as<int>(4);  // 默认质量参数维度
    params_.friction_dim = config["friction_dim"].as<int>(1);  // 默认摩擦系数维度

    // 足端力顺序重排：ROS 顺序 -> 策略/仿真顺序
    {
        std::vector<int> identity(params_.num_feet);
        std::iota(identity.begin(), identity.end(), 0);
        params_.feet_reindex = identity;
        if (!config["feet_reindex"].IsNull())
        {
            try
            {
                auto v = ReadVectorFromYaml<int>(config["feet_reindex"]);
                if (static_cast<int>(v.size()) == params_.num_feet)
                {
                    std::vector<int> used(params_.num_feet, 0);
                    bool valid = true;
                    for (const auto idx : v)
                    {
                        if (idx < 0 || idx >= params_.num_feet)
                        {
                            valid = false;
                            break;
                        }
                        used[idx] += 1;
                    }
                    for (const auto cnt : used)
                    {
                        if (cnt != 1)
                        {
                            valid = false;
                            break;
                        }
                    }
                    if (valid)
                    {
                        params_.feet_reindex = v;
                    }
                    else
                    {
                        RCLCPP_WARN(rclcpp::get_logger("StateRL"),
                                    "Invalid feet_reindex (out of range or duplicated). Using identity.");
                    }
                }
                else
                {
                    RCLCPP_WARN(rclcpp::get_logger("StateRL"),
                                "feet_reindex size=%zu != num_feet=%d. Using identity.",
                                v.size(), params_.num_feet);
                }
            }
            catch (const std::exception& e)
            {
                RCLCPP_WARN(rclcpp::get_logger("StateRL"),
                            "Failed to parse feet_reindex: %s. Using identity.", e.what());
            }
        }
    }

    // 关节顺序重排：ROS 顺序 -> 策略/仿真顺序
    {
        std::vector<int> identity(params_.num_of_dofs);
        std::iota(identity.begin(), identity.end(), 0);

        params_.dof_reindex = identity;
        if (!config["dof_reindex"].IsNull())
        {
            try
            {
                auto v = ReadVectorFromYaml<int>(config["dof_reindex"]);
                if (static_cast<int>(v.size()) == params_.num_of_dofs)
                {
                    std::vector<int> used(params_.num_of_dofs, 0);
                    bool valid = true;
                    for (const auto idx : v)
                    {
                        if (idx < 0 || idx >= params_.num_of_dofs)
                        {
                            valid = false;
                            break;
                        }
                        used[idx] += 1;
                    }
                    for (const auto cnt : used)
                    {
                        if (cnt != 1)
                        {
                            valid = false;
                            break;
                        }
                    }
                    if (valid)
                    {
                        params_.dof_reindex = v;
                    }
                    else
                    {
                        RCLCPP_WARN(rclcpp::get_logger("StateRL"),
                                    "Invalid dof_reindex (out of range or duplicated). Using identity.");
                    }
                }
                else
                {
                    RCLCPP_WARN(rclcpp::get_logger("StateRL"),
                                "dof_reindex size=%zu != num_of_dofs=%d. Using identity.",
                                v.size(), params_.num_of_dofs);
                }
            }
            catch (const std::exception& e)
            {
                RCLCPP_WARN(rclcpp::get_logger("StateRL"),
                            "Failed to parse dof_reindex: %s. Using identity.", e.what());
            }
        }

        // 逆映射：策略/仿真顺序 -> ROS 顺序
        params_.dof_reindex_inv = identity;
        for (int sim_idx = 0; sim_idx < params_.num_of_dofs; ++sim_idx)
        {
            const int ros_idx = params_.dof_reindex[sim_idx];
            params_.dof_reindex_inv[ros_idx] = sim_idx;
        }
    }
    params_.delta_yaw_scale = config["delta_yaw_scale"].as<double>();
    params_.forward_command_speed = config["forward_command_speed"].as<double>(0.0);
    if (!config["lin_vel_x_range"].IsNull())
    {
        auto range = ReadVectorFromYaml<double>(config["lin_vel_x_range"]);
        if (range.size() >= 2)
        {
            params_.lin_vel_x_min = range[0];
            params_.lin_vel_x_max = range[1];
        }
        else
        {
            params_.lin_vel_x_min = params_.forward_command_speed;
            params_.lin_vel_x_max = params_.forward_command_speed;
        }
    }
    else
    {
        params_.lin_vel_x_min = params_.forward_command_speed;
        params_.lin_vel_x_max = params_.forward_command_speed;
    }
    params_.foot_force_threshold = config["foot_force_threshold"].as<double>(5.0); // 默认阈值设为5.0N
    params_.contact_use_last = config["contact_use_last"].as<bool>(false);
    params_.use_onboard_actor_backbone = config["use_onboard_actor_backbone"].as<bool>(false);
    params_.onboard_model_name = config["onboard_model_name"].as<std::string>("onboard_jit.pt");
    params_.dry_run = config["dry_run"].as<bool>(false);
    params_.warm_up_steps = config["warm_up_steps"].as<int>(2);
    params_.log_inference_latency = config["log_inference_latency"].as<bool>(false);
    params_.latency_log_interval_ms = config["latency_log_interval_ms"].as<int>(2000);
    params_.control_step_ms = config["control_step_ms"].as<double>(20.0);
    params_.latency_log_file = config["latency_log_file"].as<std::string>("");

    // 维度参数已提前设置

    // 观测布局：这些字段必须与训练导出的模型签名一致，否则会维度对不上或语义错位
    params_.num_proprio = config["num_proprio"].as<int>(53);
    params_.num_scan = config["num_scan"].as<int>(132);  // save_jit 导出 base_jit 时为 132，与 traced obs 维度一致
    params_.num_hist_len = config["num_hist_len"].as<int>(10);
    params_.num_priv_explicit = config["num_priv_explicit"].as<int>(0);
    params_.num_priv_latent = config["num_priv_latent"].as<int>(0);
    params_.direction_mode = config["direction_mode"].as<int>(0); // 默认 0 (自动)
    RCLCPP_INFO(
        rclcpp::get_logger("StateRL"),
        "观测布局配置：proprio=%d, scan=%d, priv_explicit=%d, priv_latent=%d, hist_len=%d",
        params_.num_proprio, params_.num_scan, params_.num_priv_explicit,
        params_.num_priv_latent, params_.num_hist_len);

    // 相机相关参数：当 use_camera/use_depth_cnn 打开时，会尝试加载深度特征模型做编码
    // 注意：这里的 depth_feature_dim 与训练侧保持一致（Extreme Parkour 里通常是 32 维 latent + 2 维 yaw）
    params_.use_camera = config["use_camera"].as<bool>(false);
    params_.use_depth_cnn = config["use_depth_cnn"].as<bool>(false);
    params_.publish_contact_states = config["publish_contact_states"].as<bool>(false);
    params_.contact_states_topic = config["contact_states_topic"].as<std::string>("/our_depth_rl/contact_states");
    params_.publish_foot_force_debug = config["publish_foot_force_debug"].as<bool>(false);
    params_.foot_force_debug_topic = config["foot_force_debug_topic"].as<std::string>("/our_depth_rl/foot_force_debug");
    params_.publish_proprio = config["publish_proprio"].as<bool>(false);
    params_.proprio_topic = config["proprio_topic"].as<std::string>("/our_depth_rl/proprio");
    params_.log_proprio_stats = config["log_proprio_stats"].as<bool>(true);
    params_.proprio_log_interval_ms = config["proprio_log_interval_ms"].as<int>(2000);
    params_.publish_yaw_diff = config["publish_yaw_diff"].as<bool>(false);
    params_.yaw_diff_topic = config["yaw_diff_topic"].as<std::string>("/our_depth_rl/yaw_diff");
    params_.publish_history_order_debug = config["publish_history_order_debug"].as<bool>(false);
    params_.history_order_debug_topic = config["history_order_debug_topic"].as<std::string>("/our_depth_rl/history_order_debug");
    params_.publish_obs_freq_debug = config["publish_obs_freq_debug"].as<bool>(false);
    params_.obs_freq_debug_topic = config["obs_freq_debug_topic"].as<std::string>("/our_depth_rl/obs_freq_debug");
    if (params_.use_camera)
    {
        params_.depth_width = config["depth_width"].as<int>(58);   // 与 Python 中的 58x87 保持一致
        params_.depth_height = config["depth_height"].as<int>(87);
        params_.max_depth = config["max_depth"].as<double>(2.0);   // 有效深度范围上限
        params_.depth_crop_top = config["depth_crop_top"].as<int>(0);
        params_.depth_crop_bottom = config["depth_crop_bottom"].as<int>(0);
        params_.depth_crop_left = config["depth_crop_left"].as<int>(0);
        params_.depth_crop_right = config["depth_crop_right"].as<int>(0);
        params_.depth_update_interval = config["depth_update_interval"].as<int>(1);
        if (params_.depth_update_interval < 1) params_.depth_update_interval = 1;
        // 深度历史堆叠长度（硬件侧/encoder 可能使用 buffer_len 维度）
        params_.depth_buffer_len = config["depth_buffer_len"].as<int>(1);
        if (params_.depth_buffer_len < 1) params_.depth_buffer_len = 1;
        params_.depth_cnn_model = config["depth_cnn_model"].as<std::string>("vision_jit.pt");
        params_.depth_yaw_clip = config["depth_yaw_clip"].as<double>(0.8);
        params_.depth_yaw_alpha = config["depth_yaw_alpha"].as<double>(0.9);
        
        // Extreme Parkour 的 TorchScript 视觉骨干输出 32 维 latent + 2 维偏航
        // 允许从 config.yaml 覆盖，便于不同模型复用
        params_.depth_feature_dim = config["depth_feature_dim"].as<int>(32);
        
        if (params_.use_depth_cnn)
        {
            const std::vector<std::string> depth_model_candidates = {
                config_path + "/" + params_.depth_cnn_model,
            };

            std::string last_depth_model_error;
            for (const auto& candidate : depth_model_candidates)
            {
                try
                {
                    depth_encoder_module_ = torch::jit::load(candidate);
                    depth_encoder_module_.to(device_);
                    depth_encoder_module_.eval();
                    depth_model_loaded_ = true;
                    RCLCPP_INFO(rclcpp::get_logger("StateRL"), "成功加载深度特征模型: %s", candidate.c_str());
                    break;
                }
                catch (const std::exception& e)
                {
                    last_depth_model_error = e.what();
                    RCLCPP_WARN(rclcpp::get_logger("StateRL"), "尝试加载深度模型失败: %s (%s)", candidate.c_str(), e.what());
                }
            }
            
            if (!depth_model_loaded_)
            {
                if (!last_depth_model_error.empty())
                {
                    RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "深度特征模型加载失败 %s: %s，禁用相机特征。", params_.depth_cnn_model.c_str(), last_depth_model_error.c_str());
                }
                else
                {
                    RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "未找到可用的深度特征模型 %s，禁用相机特征。", params_.depth_cnn_model.c_str());
                }
                params_.use_camera = false;
                params_.use_depth_cnn = false;
            }
        }
    }
    // 指令缩放：把遥控输入（m/s, rad/s 等）映射到训练时的数值范围
    params_.commands_scale = torch::tensor({params_.lin_vel_scale, params_.lin_vel_scale, params_.ang_vel_scale});
    params_.rl_kp = torch::tensor(ReadVectorFromYaml<double>(config["rl_kp"], params_.framework, rows, cols)).view({
        1, -1
    });
    params_.rl_kd = torch::tensor(ReadVectorFromYaml<double>(config["rl_kd"], params_.framework, rows, cols)).view({
        1, -1
    });
    params_.torque_limits = torch::tensor(
        ReadVectorFromYaml<double>(config["torque_limits"], params_.framework, rows, cols)).view({1, -1});
    
    // 默认关节角：
    // - 默认使用 enter() 传入的 stand_pos（init_pos_）初始化
    // - 若 config.yaml 显式提供 default_dof_pos，则优先使用它（便于与训练端 default_joint_angles 完全对齐/快速扫参）
    params_.default_dof_pos = torch::from_blob(init_pos_, {12}, torch::kDouble).clone().to(torch::kFloat).unsqueeze(0);
    if (!config["default_dof_pos"].IsNull())
    {
        try
        {
            auto v = ReadVectorFromYaml<double>(config["default_dof_pos"], params_.framework, rows, cols);
            if (static_cast<int>(v.size()) == params_.num_of_dofs)
            {
                params_.default_dof_pos = torch::tensor(v, torch::kFloat32).view({1, -1});
                RCLCPP_DEBUG(rclcpp::get_logger("StateRL"), "Using default_dof_pos from config.yaml");
            }
            else
            {
                RCLCPP_WARN(
                    rclcpp::get_logger("StateRL"),
                    "config.yaml default_dof_pos size=%zu != num_of_dofs=%d, ignore and use stand_pos init.",
                    v.size(), params_.num_of_dofs);
            }
        }
        catch (const std::exception& e)
        {
            RCLCPP_WARN(rclcpp::get_logger("StateRL"), "Failed to parse default_dof_pos from config.yaml: %s", e.what());
        }
    }
    
    // params_.default_dof_pos = torch::tensor(
    //     ReadVectorFromYaml<double>(config["default_dof_pos"], params_.framework, rows, cols)).view({1, -1});
}

torch::Tensor StateRL::quatRotateInverse(const torch::Tensor& q, const torch::Tensor& v, const std::string& framework)
{
    // ======== quat_rotate_inverse 对齐分析 ========
    //
    // 训练端 (legged_robot.py):
    //   base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])
    //   Isaac Gym 的 root_states 角速度是 **世界坐标系 (world frame)**，
    //   quat_rotate_inverse(q, v) = q⁻¹·v·q，将世界系向量旋转到机体系 (body frame)。
    //   → 最终 base_ang_vel 是 body frame 角速度。
    //
    // 部署端 (Gazebo / 实机):
    //   Gazebo IMU sensor 的 angular_velocity 默认输出坐标系为
    //   **传感器局部坐标系 = body frame**（gz-sim 文档: IMU angular velocity
    //   is reported in sensor frame when <localization> is not set to CUSTOM）。
    //   → 读取的 gyroscope 已经是 body frame，无需再做 quat_rotate_inverse。
    //
    // 因此：直接返回原向量是正确的。
    //
    // 注意: obs_.lin_vel（来自 odometer）实际是世界系线速度，与训练端的 body frame
    //       base_lin_vel 不同。但 base_jit.pt 内部 estimator 会覆写 priv_explicit 段，
    //       所以 lin_vel 填什么值都不影响策略输出。
    //
    // 若未来使用 world frame 角速度源，需在此实现真正的逆旋转:
    //   v_body = q_conj * v * q  (Hamilton 四元数乘法)
    (void)q;
    (void)framework;
    return v.clone();
}


void StateRL::getState()
{
    // 从 ros2_control 的 state_interface 读取最新传感器与关节状态，并读取遥控输入。
    // 这里把数据写入 robot_state_ 与 control_，供 runModel() 拼接观测使用。
    try {
        if (params_.framework == "isaacgym")
        {
            robot_state_.imu.quaternion[3] = ctrl_interfaces_.imu_state_interface_[0].get().get_value();
            robot_state_.imu.quaternion[0] = ctrl_interfaces_.imu_state_interface_[1].get().get_value();
            robot_state_.imu.quaternion[1] = ctrl_interfaces_.imu_state_interface_[2].get().get_value();
            robot_state_.imu.quaternion[2] = ctrl_interfaces_.imu_state_interface_[3].get().get_value();
        }
        else if (params_.framework == "isaacsim")
        {
            robot_state_.imu.quaternion[0] = ctrl_interfaces_.imu_state_interface_[0].get().get_value();
            robot_state_.imu.quaternion[1] = ctrl_interfaces_.imu_state_interface_[1].get().get_value();
            robot_state_.imu.quaternion[2] = ctrl_interfaces_.imu_state_interface_[2].get().get_value();
            robot_state_.imu.quaternion[3] = ctrl_interfaces_.imu_state_interface_[3].get().get_value();
        }
        else {
            RCLCPP_WARN(rclcpp::get_logger("StateRL"), "getState: unknown framework: %s", params_.framework.c_str());
        }

        robot_state_.imu.gyroscope[0] = ctrl_interfaces_.imu_state_interface_[4].get().get_value();
        robot_state_.imu.gyroscope[1] = ctrl_interfaces_.imu_state_interface_[5].get().get_value();
        robot_state_.imu.gyroscope[2] = ctrl_interfaces_.imu_state_interface_[6].get().get_value();

        robot_state_.imu.accelerometer[0] = ctrl_interfaces_.imu_state_interface_[7].get().get_value();
        robot_state_.imu.accelerometer[1] = ctrl_interfaces_.imu_state_interface_[8].get().get_value();
        robot_state_.imu.accelerometer[2] = ctrl_interfaces_.imu_state_interface_[9].get().get_value();

        for (int i = 0; i < 12; i++)
        {
            robot_state_.motor_state.q[i] = ctrl_interfaces_.joint_position_state_interface_[i].get().get_value();
            robot_state_.motor_state.dq[i] = ctrl_interfaces_.joint_velocity_state_interface_[i].get().get_value();
            robot_state_.motor_state.tauEst[i] = ctrl_interfaces_.joint_effort_state_interface_[i].get().get_value();
        }

        // 遥控输入映射
        // 从接口读取摇杆值：ly控制前后，lx控制左右平移，rx控制转向
        control_.x = ctrl_interfaces_.control_inputs_.ly;
        control_.y = ctrl_interfaces_.control_inputs_.lx;
        control_.yaw = ctrl_interfaces_.control_inputs_.rx;

        // 读取里程计线速度
        double odom_vx = 0.0, odom_vy = 0.0, odom_vz = 0.0;
        if (ctrl_interfaces_.odom_state_interface_.size() >= 6)
        {
            odom_vx = ctrl_interfaces_.odom_state_interface_[3].get().get_value();
            odom_vy = ctrl_interfaces_.odom_state_interface_[4].get().get_value();
            odom_vz = ctrl_interfaces_.odom_state_interface_[5].get().get_value();
        }
        const auto cpu_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        obs_.lin_vel = torch::tensor({{static_cast<float>(odom_vx),
                                       static_cast<float>(odom_vy),
                                       static_cast<float>(odom_vz)}}, cpu_f32);

        // 足力统计
        if (!ctrl_interfaces_.foot_force_state_interface_.empty())
        {
            const int n = std::min<int>(4, ctrl_interfaces_.foot_force_state_interface_.size());
            for (int i = 0; i < n; ++i)
            {
                const double v = ctrl_interfaces_.foot_force_state_interface_[i].get().get_value();
                if (!foot_force_stats_initialized_)
                {
                    foot_force_min_[i] = v;
                    foot_force_max_[i] = v;
                }
                else
                {
                    foot_force_min_[i] = std::min(foot_force_min_[i], v);
                    foot_force_max_[i] = std::max(foot_force_max_[i], v);
                }
            }
            foot_force_stats_initialized_ = true;
            RCLCPP_DEBUG_THROTTLE(
                rclcpp::get_logger("StateRL"), *node_->get_clock(), 2000,
                "Foot force range [N] FR(%.2f-%.2f) FL(%.2f-%.2f) RR(%.2f-%.2f) RL(%.2f-%.2f), thr=%.2f",
                foot_force_min_[0], foot_force_max_[0],
                foot_force_min_[1], foot_force_max_[1],
                foot_force_min_[2], foot_force_max_[2],
                foot_force_min_[3], foot_force_max_[3],
                params_.foot_force_threshold);
        }

        updated_ = true;
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "Exception in StateRL::getState(): %s", e.what());
        throw;
    }
}

// 发布观测/策略步长调试： [实际步长ms, 期望步长ms]，用于与训练 sim.dt*decimation=20ms 对齐
void StateRL::publishObsFreqDebug()
{
    const rclcpp::Time now = node_->get_clock()->now();
    if (params_.publish_obs_freq_debug && obs_freq_debug_pub_ && last_run_model_time_.nanoseconds() > 0)
    {
        const double actual_ms = (now - last_run_model_time_).seconds() * 1000.0;
        std_msgs::msg::Float32MultiArray msg;
        msg.data.resize(2);
        msg.data[0] = static_cast<float>(actual_ms);
        msg.data[1] = static_cast<float>(params_.control_step_ms);
        obs_freq_debug_pub_->publish(msg);
    }
    last_run_model_time_ = now;
}

// 构建关节/足端重排张量：ROS 顺序 -> 策略顺序（与 legged_robot.py reindex / reindex_feet 一致）
RunModelReindex StateRL::buildReindexTensors()
{
    RunModelReindex r;
    r.dof_reindex = torch::tensor(params_.dof_reindex, torch::TensorOptions().dtype(torch::kLong).device(device_));
    r.dof_reindex_inv = torch::tensor(params_.dof_reindex_inv, torch::TensorOptions().dtype(torch::kLong).device(device_));
    if (static_cast<int>(params_.feet_reindex.size()) == params_.num_feet)
        r.feet_reindex = torch::tensor(params_.feet_reindex, torch::TensorOptions().dtype(torch::kLong).device(device_));
    else
    {
        r.feet_reindex = torch::arange(params_.num_feet, torch::TensorOptions().dtype(torch::kLong).device(device_));
        RCLCPP_WARN_THROTTLE(rclcpp::get_logger("StateRL"), *node_->get_clock(), 2000,
                             "feet_reindex size mismatch (%zu vs %d), fallback to identity.",
                             params_.feet_reindex.size(), params_.num_feet);
    }
    return r;
}

// 构建 53 维本体观测 proprio，与 legged_robot.compute_observations() 中 obs_buf 顺序一致：
// [角速度*scale, imu(roll,pitch), yaw_info(3), commands(3), env_class(2), (pos-default)*scale, vel*scale, last_actions, contact]
RunModelProprioResult StateRL::buildProprio(const RunModelReindex& reindex,
                                           const torch::TensorOptions& options,
                                           const std::function<torch::Tensor(const torch::Tensor&)>& sanitize_tensor)
{
    const torch::Tensor& dof_reindex = reindex.dof_reindex;
    const torch::Tensor& feet_reindex = reindex.feet_reindex;

    // ---------- 角速度（3 维，与训练 base_ang_vel * obs_scales.ang_vel 一致）----------
        std::array<float, 3> ang_vel_arr{
            static_cast<float>(robot_state_.imu.gyroscope[0]),
            static_cast<float>(robot_state_.imu.gyroscope[1]),
            static_cast<float>(robot_state_.imu.gyroscope[2])
        };
        torch::Tensor ang_vel = torch::from_blob(ang_vel_arr.data(), {1, 3}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
        ang_vel = ang_vel * static_cast<float>(params_.ang_vel_scale);

        // ---------- IMU 姿态：roll, pitch（与训练 imu_obs = stack(roll, pitch) 一致）----------
        tf2::Quaternion base_quat_tf(robot_state_.imu.quaternion[0],
                                     robot_state_.imu.quaternion[1],
                                     robot_state_.imu.quaternion[2],
                                     robot_state_.imu.quaternion[3]);
        tf2::Matrix3x3 rot_mat(base_quat_tf);
        double roll, pitch, yaw;
        rot_mat.getRPY(roll, pitch, yaw);
        
        // IMU 观测：这里只取 roll/pitch（是否需要 yaw 取决于训练侧设计）
        // training obs_buf uses imu_obs = [roll, pitch]
        torch::Tensor imu = sanitize_tensor(torch::tensor({{static_cast<float>(roll), static_cast<float>(pitch)}}, options));

        // ---------- 偏航信息 3 维：训练为 0*delta_yaw, delta_yaw, delta_next_yaw；无目标时填 0 保持维度一致 ----------
        torch::Tensor yaw_info = torch::zeros({1, 3}, options);

        // ---------- 速度指令：前两维置 0，第三维为前向速度（与训练 commands[:,0:1] 一致）----------
        // 键盘 ly (control_.x) 映射到 [lin_vel_x_min, lin_vel_x_max]；无键盘或 ly=0 用范围中点或 forward_command_speed。
        double fwd_raw = params_.forward_command_speed;
        if (params_.lin_vel_x_min < params_.lin_vel_x_max)
        {
            // ly in [-1, 1] -> [min, max]; ly=0 -> mid
            const double ly = static_cast<double>(control_.x);
            fwd_raw = params_.lin_vel_x_min + (ly + 1.0) * 0.5 * (params_.lin_vel_x_max - params_.lin_vel_x_min);
            fwd_raw = std::max(params_.lin_vel_x_min, std::min(params_.lin_vel_x_max, fwd_raw));
        }
        const float fwd_cmd = static_cast<float>(fwd_raw);
        torch::Tensor commands = torch::tensor({{0.0f, 0.0f, fwd_cmd}}, options);

        // ---------- 环境类别 one-hot：训练 (env_class!=17), (env_class==17)；部署默认 (1,0) ----------
        torch::Tensor env_class = torch::tensor({{1.0f, 0.0f}}, options);

        // ---------- 关节位置/速度：ROS 顺序读入，按 dof_reindex 转为策略顺序，再减默认、乘 scale ----------
        std::vector<float> joint_pos_vec(params_.num_of_dofs);
        std::vector<float> joint_vel_vec(params_.num_of_dofs);
        for (int i = 0; i < params_.num_of_dofs; ++i)
        {
            joint_pos_vec[i] = static_cast<float>(robot_state_.motor_state.q[i]);
            joint_vel_vec[i] = static_cast<float>(robot_state_.motor_state.dq[i]);
        }

        torch::Tensor joint_pos_raw = sanitize_tensor(torch::tensor(joint_pos_vec, options).view({1, params_.num_of_dofs}));
        torch::Tensor joint_vel_raw = sanitize_tensor(torch::tensor(joint_vel_vec, options).view({1, params_.num_of_dofs}));

        torch::Tensor default_dof_pos_raw = params_.default_dof_pos.to(options);

        torch::Tensor joint_pos = joint_pos_raw.index_select(1, dof_reindex);
        torch::Tensor joint_vel = joint_vel_raw.index_select(1, dof_reindex);
        torch::Tensor default_dof_pos = default_dof_pos_raw.index_select(1, dof_reindex);

        joint_pos = (joint_pos - default_dof_pos) * static_cast<float>(params_.dof_pos_scale);
        joint_vel = joint_vel * static_cast<float>(params_.dof_vel_scale);

        // 关节速度低通滤波（实机速度反馈噪声大时可开启，alpha 建议 0.2~0.5）
        const float vel_alpha = static_cast<float>(params_.dof_vel_filter_alpha);
        if (vel_alpha > 1e-6f && vel_alpha <= 1.0f)
        {
            if (dof_vel_filtered_prev_.defined() && dof_vel_filtered_prev_.numel() == joint_vel.numel())
                joint_vel = vel_alpha * joint_vel + (1.0f - vel_alpha) * dof_vel_filtered_prev_.to(options);
            dof_vel_filtered_prev_ = joint_vel.clone().to(torch::kCPU);
        }

        // 关节分布调试：用于检查关节顺序/尺度是否异常
        {
            auto joint_pos_cpu = joint_pos.to(torch::kCPU);
            auto joint_vel_cpu = joint_vel.to(torch::kCPU);
            const double pos_mean = joint_pos_cpu.mean().item<double>();
            const double pos_std = joint_pos_cpu.std(false).item<double>();
            const double vel_mean = joint_vel_cpu.mean().item<double>();
            const double vel_std = joint_vel_cpu.std(false).item<double>();
            RCLCPP_DEBUG_THROTTLE(
                rclcpp::get_logger("StateRL"), *node_->get_clock(), 2000,
                "Joint stats: pos_mean=%.4f pos_std=%.4f vel_mean=%.4f vel_std=%.4f",
                pos_mean, pos_std, vel_mean, vel_std);
        }

        // ---------- 上一拍动作（与训练 action_history_buf[:,-1] 语义一致：上一步策略输出）----------
        torch::Tensor last_actions_raw = obs_.actions.numel() == 0
            ? torch::zeros({1, params_.num_of_dofs}, options)
            : sanitize_tensor(obs_.actions.to(options));
        torch::Tensor last_actions = last_actions_raw.index_select(1, dof_reindex);

        // ---------- 足端接触：训练 contact = norm(contact_forces)>2, contact_filt = contact|last_contacts；此处用标量力与阈值，contact_use_last 控制是否 OR ----------
        torch::Tensor contact_bool = torch::zeros(
            {1, params_.num_feet},
            torch::TensorOptions().dtype(torch::kBool).device(device_));
        if (ctrl_interfaces_.foot_force_state_interface_.empty())
        {
            RCLCPP_WARN_THROTTLE(
                rclcpp::get_logger("StateRL"),
                *node_->get_clock(),
                2000,
                "foot_force_state_interface_ is empty, contact states fallback to -0.5");
            contact_bool.fill_(false);
        }
        else
        {
            const int n = std::min<int>(params_.num_feet, ctrl_interfaces_.foot_force_state_interface_.size());
            if (n != params_.num_feet)
            {
                RCLCPP_WARN_THROTTLE(
                    rclcpp::get_logger("StateRL"),
                    *node_->get_clock(),
                    2000,
                    "foot_force_state_interface_ size mismatch: got %zu, expected %d",
                    ctrl_interfaces_.foot_force_state_interface_.size(),
                    params_.num_feet);
            }
            // Align with training logic (legged_robot.py):
            //   contact = norm(contact_forces(feet)) > thr
            //   contact_filt = contact OR last_contacts
            // Here we only have a scalar foot force per foot; use a single threshold and
            // apply the same one-step OR filter to reduce flicker.
            const float thr = static_cast<float>(params_.foot_force_threshold);
            for (int i = 0; i < n; ++i)
            {
                const float v = static_cast<float>(ctrl_interfaces_.foot_force_state_interface_[i].get().get_value());
                const bool contact_now = (v > thr);
                contact_bool[0][i] = contact_now;
            }
            if (n < params_.num_feet)
            {
                contact_bool.index_put_({0, torch::indexing::Slice(n, params_.num_feet)}, false);
            }

            // EMA 统计（用于评估 foot_force_threshold 是否合理）
            const double alpha = 0.1;  // EMA 衰减系数
            if (!foot_contact_ema_initialized_)
            {
                for (int i = 0; i < n; ++i)
                {
                    const double v = static_cast<double>(ctrl_interfaces_.foot_force_state_interface_[i].get().get_value());
                    foot_force_mean_ema_[i] = v;
                    contact_ratio_ema_[i] = contact_bool[0][i].item<bool>() ? 1.0 : 0.0;
                }
                foot_contact_ema_initialized_ = true;
            }
            else
            {
                for (int i = 0; i < n; ++i)
                {
                    const double v = static_cast<double>(ctrl_interfaces_.foot_force_state_interface_[i].get().get_value());
                    const double contact_val = contact_bool[0][i].item<bool>() ? 1.0 : 0.0;
                    foot_force_mean_ema_[i] = foot_force_mean_ema_[i] * (1.0 - alpha) + v * alpha;
                    contact_ratio_ema_[i] = contact_ratio_ema_[i] * (1.0 - alpha) + contact_val * alpha;
                }
            }
            RCLCPP_DEBUG_THROTTLE(
                rclcpp::get_logger("StateRL"), *node_->get_clock(), 2000,
                "Foot force EMA [N] idx0=%.2f idx1=%.2f idx2=%.2f idx3=%.2f | contact ratio idx0=%.2f idx1=%.2f idx2=%.2f idx3=%.2f | thr=%.2f",
                foot_force_mean_ema_[0], foot_force_mean_ema_[1], foot_force_mean_ema_[2], foot_force_mean_ema_[3],
                contact_ratio_ema_[0], contact_ratio_ema_[1], contact_ratio_ema_[2], contact_ratio_ema_[3],
                params_.foot_force_threshold);
        }

        if (!last_contact_bool_.defined() || last_contact_bool_.sizes() != contact_bool.sizes())
        {
            last_contact_bool_ = torch::zeros_like(contact_bool);
        }
        torch::Tensor contact_filt_bool = contact_bool;
        if (params_.contact_use_last)
        {
            contact_filt_bool = contact_bool | last_contact_bool_;
        }
        last_contact_bool_ = contact_bool.clone();

        torch::Tensor contact_raw = torch::where(
            contact_filt_bool,
            torch::full({1, params_.num_feet}, 0.5f, options),
            torch::full({1, params_.num_feet}, -0.5f, options));

        // 训练 reindex_feet(contact_filt.float()-0.5)，即接触 {0.5, -0.5}；此处 contact_raw 已是 ±0.5，再按 feet_reindex 重排
        torch::Tensor contact = contact_raw.index_select(1, feet_reindex);
        obs_.ang_vel = ang_vel.to(torch::kCPU);
        obs_.commands = commands.to(torch::kCPU);
        obs_.dof_pos = joint_pos.to(torch::kCPU);
        obs_.dof_vel = joint_vel.to(torch::kCPU);
        obs_.roll = torch::tensor(static_cast<float>(roll)).unsqueeze(0);
        obs_.pitch = torch::tensor(static_cast<float>(pitch)).unsqueeze(0);
        obs_.delta_yaw = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32));
        obs_.contact_states = contact.to(torch::kCPU);
        if (params_.publish_contact_states && contact_states_pub_)
        {
            std_msgs::msg::Float32MultiArray msg;
            msg.data.resize(params_.num_feet);
            for (int i = 0; i < params_.num_feet; ++i)
            {
                msg.data[i] = contact[0][i].item<float>();
            }
            contact_states_pub_->publish(msg);
        }
        if (params_.publish_foot_force_debug && foot_force_debug_pub_ && !ctrl_interfaces_.foot_force_state_interface_.empty())
        {
            std_msgs::msg::Float32MultiArray msg;
            msg.data.resize(5);
            const int n = std::min<int>(4, static_cast<int>(ctrl_interfaces_.foot_force_state_interface_.size()));
            for (int i = 0; i < n; ++i)
                msg.data[i] = static_cast<float>(ctrl_interfaces_.foot_force_state_interface_[i].get().get_value());
            for (int i = n; i < 4; ++i)
                msg.data[i] = 0.0f;
            msg.data[4] = static_cast<float>(params_.foot_force_threshold);
            foot_force_debug_pub_->publish(msg);
        }

        // ---------- 按训练顺序拼接 53 维 proprio（与 legged_robot obs_buf 一致）----------
        torch::Tensor proprio = torch::cat(
            {ang_vel, imu, yaw_info, commands, env_class, joint_pos, joint_vel, last_actions, contact}, 1);
        proprio = sanitize_tensor(proprio.to(device_));
        const torch::Tensor proprio_cpu_log = proprio.to(torch::kCPU);
        if (params_.log_proprio_stats)
        {
            const int interval = std::max(500, params_.proprio_log_interval_ms);
            const int dim = static_cast<int>(proprio_cpu_log.numel());
            const bool all_finite = torch::isfinite(proprio_cpu_log).all().item<bool>();
            const double mean = proprio_cpu_log.mean().item<double>();
            const double std = proprio_cpu_log.std(false).item<double>();
            const double vmin = proprio_cpu_log.min().item<double>();
            const double vmax = proprio_cpu_log.max().item<double>();
            RCLCPP_DEBUG_THROTTLE(
                rclcpp::get_logger("StateRL"), *node_->get_clock(), interval,
                "Proprio stats: dim=%d mean=%.4f std=%.4f min=%.4f max=%.4f finite=%s",
                dim, mean, std, vmin, vmax, all_finite ? "true" : "false");
            if (dim != params_.num_proprio)
            {
                RCLCPP_WARN_THROTTLE(
                    rclcpp::get_logger("StateRL"), *node_->get_clock(), interval,
                    "Proprio dim mismatch: got %d, expected %d", dim, params_.num_proprio);
            }
            if (!all_finite)
            {
                RCLCPP_WARN_THROTTLE(
                    rclcpp::get_logger("StateRL"), *node_->get_clock(), interval,
                    "Proprio contains NaN/Inf values.");
            }
        }
        if (params_.publish_proprio && proprio_pub_)
        {
            std_msgs::msg::Float32MultiArray msg;
            const int dim = static_cast<int>(proprio_cpu_log.numel());
            msg.data.resize(dim);
            for (int i = 0; i < dim; ++i)
            {
                msg.data[i] = proprio_cpu_log.view({-1})[i].item<float>();
            }
            proprio_pub_->publish(msg);
        }
    return RunModelProprioResult{ proprio, default_dof_pos };
}

// 深度编码：用最新深度图+proprio 跑 vision_jit，得到 depth_latent；并将 yaw 两维写回 proprio_inout[6:8]（与训练观测一致）
torch::Tensor StateRL::updateDepthLatentAndYaw(torch::Tensor& proprio_inout,
                                               const torch::TensorOptions& options,
                                               const std::function<torch::Tensor(const torch::Tensor&)>& sanitize_tensor)
{
    torch::Tensor depth_latent = torch::zeros({1, params_.depth_feature_dim}, options);
    if (last_depth_latent_.defined() && last_depth_latent_.numel() > 0)
        depth_latent = last_depth_latent_.to(options);
    const bool depth_ok = last_depth_frame_valid_.load();
    depth_update_counter_++;
    const bool should_update_depth = (depth_update_counter_ % params_.depth_update_interval) == 0;
    if (params_.use_camera && params_.use_depth_cnn && depth_model_loaded_ && depth_ok && should_update_depth)
    {
        try
        {
            const auto t_depth_start = std::chrono::steady_clock::now();
            torch::Tensor latest_depth_copy;
            {
                std::lock_guard<std::mutex> lock(depth_mutex_);
                if (latest_depth_tensor_.defined() && latest_depth_tensor_.numel() > 0)
                    latest_depth_copy = latest_depth_tensor_.clone();
            }
            if (latest_depth_copy.defined() && latest_depth_copy.numel() > 0)
            {
                if (latest_depth_copy.dim() == 5 && latest_depth_copy.size(1) == 1)
                    latest_depth_copy = latest_depth_copy.squeeze(1);
                if (latest_depth_copy.dim() == 4)
                    latest_depth_copy = latest_depth_copy.index({torch::indexing::Slice(), -1, torch::indexing::Slice(), torch::indexing::Slice()});
                else if (latest_depth_copy.dim() == 2)
                    latest_depth_copy = latest_depth_copy.unsqueeze(0);
                if (latest_depth_copy.dim() != 3)
                {
                    RCLCPP_WARN_THROTTLE(rclcpp::get_logger("StateRL"), *node_->get_clock(), 2000,
                                         "Depth tensor dim unexpected: %ld (expect 3). Skip depth encoder.", static_cast<long>(latest_depth_copy.dim()));
                    latest_depth_copy = torch::Tensor();
                }
                if (latest_depth_copy.defined() && latest_depth_copy.numel() > 0)
                {
                    if (latest_depth_copy.dim() == 4 && latest_depth_copy.size(0) == 1)
                        latest_depth_copy = latest_depth_copy.index({0, -1}).unsqueeze(0);

                    // 初始化 GRU hidden state（首次调用时）
                    // GRU 参数：num_layers=1, batch=1, hidden_size=512
                    if (!depth_gru_hidden_initialized_ || !depth_gru_hidden_.defined())
                    {
                        depth_gru_hidden_ = torch::zeros({1, 1, 512}, options);
                        depth_gru_hidden_initialized_ = true;
                    }

                    std::vector<torch::jit::IValue> depth_inputs;
                    depth_inputs.emplace_back(latest_depth_copy.to(device_));
                    depth_inputs.emplace_back(proprio_inout);
                    depth_inputs.emplace_back(depth_gru_hidden_.to(device_));

                    // vision_stateful_jit.pt 返回 tuple(depth_output, new_hidden)
                    auto forward_result = depth_encoder_module_.forward(depth_inputs);
                    torch::Tensor depth_output;
                    if (forward_result.isTuple())
                    {
                        // stateful 模型：(output [1, 34], new_hidden [1, 1, 512])
                        auto tuple_elements = forward_result.toTuple()->elements();
                        depth_output = sanitize_tensor(tuple_elements[0].toTensor());
                        depth_gru_hidden_ = tuple_elements[1].toTensor().to(torch::kCPU);
                    }
                    else
                    {
                        // 兼容旧版 vision_jit.pt（无 state，返回单个 tensor）
                        depth_output = sanitize_tensor(forward_result.toTensor());
                    }
                    if (depth_output.sizes().size() == 2 && depth_output.size(0) == 1 && depth_output.size(1) >= params_.depth_feature_dim + 2)
                    {
                        depth_latent = depth_output.index({0, torch::indexing::Slice(0, params_.depth_feature_dim)}).unsqueeze(0);
                        torch::Tensor yaw_input;
                        if (params_.direction_mode == 1)
                        {
                            float manual_yaw = static_cast<float>(control_.yaw * params_.delta_yaw_scale);
                            yaw_input = torch::tensor({{manual_yaw, manual_yaw}}, options);
                        }
                        else
                            yaw_input = sanitize_tensor(depth_output.index({0, torch::indexing::Slice(params_.depth_feature_dim, params_.depth_feature_dim + 2)}).unsqueeze(0) * 1.5f);
                        const float yaw_clip = static_cast<float>(params_.depth_yaw_clip);
                        yaw_input = torch::clamp(yaw_input, -yaw_clip, yaw_clip);
                        const float alpha = static_cast<float>(params_.depth_yaw_alpha);
                        if (!has_depth_yaw_filtered_ || !depth_yaw_filtered_.defined() || depth_yaw_filtered_.sizes() != yaw_input.sizes())
                        {
                            depth_yaw_filtered_ = yaw_input.clone();
                            has_depth_yaw_filtered_ = true;
                        }
                        else
                            depth_yaw_filtered_ = depth_yaw_filtered_ * alpha + yaw_input * (1.0f - alpha);
                        proprio_inout.index_put_({torch::indexing::Slice(), torch::indexing::Slice(6, 8)}, depth_yaw_filtered_);
                        if (params_.publish_yaw_diff && yaw_diff_pub_)
                        {
                            const float manual_yaw_scaled = static_cast<float>(std::clamp(control_.yaw * params_.delta_yaw_scale, -params_.depth_yaw_clip, params_.depth_yaw_clip));
                            torch::Tensor policy_yaw_raw = sanitize_tensor(depth_output.index({0, torch::indexing::Slice(params_.depth_feature_dim, params_.depth_feature_dim + 2)}).unsqueeze(0) * 1.5f);
                            const float yaw_clip_f = static_cast<float>(params_.depth_yaw_clip);
                            float policy_yaw = std::clamp(policy_yaw_raw[0][0].item<float>(), -yaw_clip_f, yaw_clip_f);
                            std_msgs::msg::Float32MultiArray msg;
                            msg.data = {manual_yaw_scaled, policy_yaw, manual_yaw_scaled - policy_yaw};
                            yaw_diff_pub_->publish(msg);
                        }
                    }
                    else
                        RCLCPP_WARN_THROTTLE(rclcpp::get_logger("StateRL"), *node_->get_clock(), 2000, "Depth encoder output shape unexpected.");
                }
                const auto t_depth_end = std::chrono::steady_clock::now();
                if (params_.log_inference_latency)
                {
                    const double depth_ms = std::chrono::duration<double, std::milli>(t_depth_end - t_depth_start).count();
                    latency_depth_ms_ema_ = latency_ema_initialized_ ? (0.9 * latency_depth_ms_ema_ + 0.1 * depth_ms) : depth_ms;
                }
            }
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "Depth encoder forward failed: %s", e.what());
        }
    }
    else if (params_.use_camera && params_.use_depth_cnn && depth_model_loaded_ && !depth_ok)
    {
        RCLCPP_WARN_THROTTLE(rclcpp::get_logger("StateRL"), *node_->get_clock(), 1000,
                             "Depth invalid -> skip depth encoder (depth_latent=zeros) for stability.");
    }
    obs_.depth_latent = depth_latent.to(torch::kCPU);
    last_depth_latent_ = obs_.depth_latent.clone();
    return depth_latent;
}

// 观测历史：与训练一致——本步 policy 使用「更新前」的 history（[t-9..t-1]），再写入当前步（yaw 6:8 置 0）
torch::Tensor StateRL::getHistoryFlatAndUpdate(const torch::Tensor& proprio,
                                               const torch::TensorOptions& options,
                                               const std::function<torch::Tensor(const torch::Tensor&)>& sanitize_tensor)
{
    (void)options;
    torch::Tensor proprio_cpu = sanitize_tensor(proprio.to(torch::kCPU));
    torch::Tensor proprio_history = proprio_cpu.clone();
    proprio_history.index_put_({torch::indexing::Slice(), torch::indexing::Slice(6, 8)}, 0.0f);  // 写入历史的 proprio 屏蔽 yaw
    std::vector<int> history_indices = params_.observations_history;
    if (history_indices.empty())
    {
        history_indices.resize(std::max(params_.num_hist_len, 1));
        std::iota(history_indices.begin(), history_indices.end(), 0);
    }
    torch::Tensor history_flat;
    if (history_steps_since_clear_ == 0)
    {
        history_obs_buf_->reset({0}, proprio_history);
        history_steps_since_clear_ = 1;
        history_flat = sanitize_tensor(history_obs_buf_->getObsVec(history_indices).to(device_));
    }
    else
    {
        history_flat = sanitize_tensor(history_obs_buf_->getObsVec(history_indices).to(device_));
        history_obs_buf_->insert(proprio_history);
    }
    if (params_.publish_history_order_debug && history_order_debug_pub_ && params_.num_hist_len > 0 && params_.num_proprio > 0)
    {
        std_msgs::msg::Float32MultiArray msg;
        msg.data.resize(static_cast<size_t>(params_.num_hist_len));
        for (int k = 0; k < params_.num_hist_len; ++k)
        {
            const int start = k * params_.num_proprio;
            const int end = start + params_.num_proprio;
            torch::Tensor seg = history_flat.index({0, torch::indexing::Slice(start, end)});
            msg.data[static_cast<size_t>(k)] = static_cast<float>(seg.mean().item<double>());
        }
        history_order_debug_pub_->publish(msg);
    }
    return history_flat;
}

// 策略前向：onboard 路径输入 (proprio, proprio_history_seq, depth_latent)；否则拼 policy_obs = [proprio|scan|priv_explicit|priv_latent|history_flat] 再与 depth_latent 输入
torch::Tensor StateRL::runPolicyForward(const torch::Tensor& proprio,
                                        const torch::Tensor& history_flat,
                                        const torch::Tensor& depth_latent,
                                        const torch::TensorOptions& options,
                                        const std::function<torch::Tensor(const torch::Tensor&)>& sanitize_tensor)
{
    if (use_onboard_actor_backbone_ && onboard_model_loaded_)
    {
        torch::Tensor proprio_history_seq = history_flat.view({1, params_.num_hist_len, params_.num_proprio});
        std::vector<torch::jit::IValue> onboard_inputs;
        onboard_inputs.emplace_back(proprio);
        onboard_inputs.emplace_back(proprio_history_seq);
        onboard_inputs.emplace_back(depth_latent);
        return sanitize_tensor(onboard_policy_module_.forward(onboard_inputs).toTensor().to(device_));
    }
    const int policy_obs_dim = params_.num_proprio + params_.num_scan + params_.num_priv_explicit + params_.num_priv_latent + params_.num_hist_len * params_.num_proprio;
    torch::Tensor policy_obs = torch::zeros({1, policy_obs_dim}, options);
    policy_obs.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, params_.num_proprio)}, proprio);
    if (params_.num_priv_explicit > 0)
    {
        const int priv_offset = params_.num_proprio + params_.num_scan;
        torch::Tensor priv_explicit = torch::zeros({1, params_.num_priv_explicit}, options);
        torch::Tensor base_lin_vel = obs_.lin_vel.defined() ? obs_.lin_vel.to(options) : torch::zeros({1, 3}, options);
        base_lin_vel = base_lin_vel * static_cast<float>(params_.lin_vel_scale);
        const int copy_dims = std::min(3, params_.num_priv_explicit);
        priv_explicit.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, copy_dims)}, base_lin_vel.index({0, torch::indexing::Slice(0, copy_dims)}).unsqueeze(0));
        policy_obs.index_put_({torch::indexing::Slice(), torch::indexing::Slice(priv_offset, priv_offset + params_.num_priv_explicit)}, priv_explicit);
    }
    if (params_.num_hist_len > 0)
    {
        const int history_offset = policy_obs_dim - params_.num_hist_len * params_.num_proprio;
        policy_obs.index_put_({torch::indexing::Slice(), torch::indexing::Slice(history_offset, policy_obs_dim)}, history_flat);
    }
    std::vector<torch::jit::IValue> policy_inputs;
    policy_inputs.emplace_back(policy_obs);
    policy_inputs.emplace_back(depth_latent);
    return sanitize_tensor(policy_module_.forward(policy_inputs).toTensor().to(device_));
}

// 动作后处理：与训练一致——先按 clip_actions/action_scale 裁剪，再髋关节缩放、EMA 滤波、乘 action_scale 加 default 得目标关节角，再过渡混合、写 obs_.actions
void StateRL::processActions(const torch::Tensor& actions,
                             const torch::Tensor& default_dof_pos_reindexed,
                             const torch::Tensor& dof_reindex_inv,
                             const torch::TensorOptions& options,
                             const std::function<torch::Tensor(const torch::Tensor&)>& sanitize_tensor)
{
    const float clip_actions = static_cast<float>(params_.clip_actions);
    const float action_scale_cfg = static_cast<float>(params_.action_scale);
    torch::Tensor actions_clamped = actions;  // 训练：clip_actions/action_scale 后 clip
    if (action_scale_cfg > 1e-6f)
    {
        const float clip_abs = clip_actions / action_scale_cfg;
        actions_clamped = torch::clamp(actions, -clip_abs, clip_abs);
    }
    for (const int index : params_.hip_scale_reduction_indices)
    {
        if (index >= 0 && index < params_.num_of_dofs)
            actions_clamped[0][index] *= static_cast<float>(params_.hip_scale_reduction);
    }
    const float action_alpha = static_cast<float>(params_.action_filter_alpha);
    if (!has_actions_filtered_ || !actions_filtered_.defined() || actions_filtered_.sizes() != actions_clamped.sizes())
    {
        actions_filtered_ = actions_clamped.clone();
        has_actions_filtered_ = true;
    }
    else
        actions_filtered_ = actions_filtered_ * action_alpha + actions_clamped * (1.0f - action_alpha);
    torch::Tensor actions_scaled = sanitize_tensor(actions_filtered_ * static_cast<float>(params_.action_scale));  // 与训练 actions*action_scale 一致
    torch::Tensor output_dof_pos_policy = sanitize_tensor(actions_scaled + default_dof_pos_reindexed);
    output_dof_pos_ = sanitize_tensor(output_dof_pos_policy.index_select(1, dof_reindex_inv));  // 策略顺序 -> ROS 顺序
    if (transition_percent_ < 1.5)  // 上电/切状态时的平滑过渡
    {
        transition_percent_ += 1.0 / transition_duration_;
        const double phase = std::tanh(transition_percent_);
        torch::Tensor start_pos_tensor = torch::from_blob(const_cast<double*>(start_pos_), {1, params_.num_of_dofs}, torch::TensorOptions().dtype(torch::kFloat64)).to(options);
        output_dof_pos_ = output_dof_pos_ * phase + start_pos_tensor * (1.0 - phase);
    }
    obs_.actions = actions_filtered_.index_select(1, dof_reindex_inv).to(torch::kCPU);
}

// 记录本步推理总耗时 EMA，并可选写入 latency_log_file（时间戳, 耗时ms, 是否小于 control_step_ms）
void StateRL::logInferenceLatency(const std::chrono::steady_clock::time_point& t_infer_start)
{
    const auto t_infer_end = std::chrono::steady_clock::now();
    if (!params_.log_inference_latency)
        return;
    const double total_ms = std::chrono::duration<double, std::milli>(t_infer_end - t_infer_start).count();
    latency_total_ms_ema_ = latency_ema_initialized_ ? (0.9 * latency_total_ms_ema_ + 0.1 * total_ms) : total_ms;
    latency_ema_initialized_ = true;
    const int interval = std::max(500, params_.latency_log_interval_ms);
    const bool under_step = (latency_total_ms_ema_ < params_.control_step_ms);
    RCLCPP_INFO_THROTTLE(rclcpp::get_logger("StateRL"), *node_->get_clock(), interval,
                         "[推理] latency EMA [ms]: total=%.3f policy=%.3f depth=%.3f | step=%.1fms under_step=%s",
                         latency_total_ms_ema_, latency_policy_ms_ema_, latency_depth_ms_ema_,
                         params_.control_step_ms, under_step ? "yes" : "no");
    if (!params_.latency_log_file.empty())
    {
        std::ofstream ofs(params_.latency_log_file, std::ios::app);
        if (ofs.is_open())
            ofs << node_->get_clock()->now().seconds() << "," << latency_total_ms_ema_ << "," << (under_step ? 1 : 0) << "\n";
    }
}

// 将 output_dof_pos_ 与 rl_kp/rl_kd 写入 robot_command_，供 setCommand() 写入硬件接口
void StateRL::writeRobotCommandFromOutput()
{
    for (int i = 0; i < params_.num_of_dofs; ++i)
    {
        robot_command_.motor_command.q[i] = output_dof_pos_[0][i].item<double>();
        robot_command_.motor_command.dq[i] = 0.0;
        robot_command_.motor_command.kp[i] = params_.rl_kp[0][i].item<double>();
        robot_command_.motor_command.kd[i] = params_.rl_kd[0][i].item<double>();
        robot_command_.motor_command.tau[i] = 0.0;
    }
    inference_step_count_++;
}

// 单步策略推理主流程（与训练 step 观测→策略→动作 对齐）：
// 1) 构建 proprio  2) 深度编码并写回 yaw  3) 取历史、更新 buffer  4) 策略前向  5) 动作裁剪/缩放/滤波 → 目标关节角  6) 写 robot_command_
void StateRL::runModel()
{
    if (!model_loaded_)
    {
        RCLCPP_WARN_THROTTLE(rclcpp::get_logger("StateRL"), *node_->get_clock(), 2000,
                             "Policy module not loaded, skip runModel.");
        return;
    }
    publishObsFreqDebug();

    torch::NoGradGuard no_grad;
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    const auto sanitize_tensor = [&](const torch::Tensor& t) -> torch::Tensor {
        if (!t.defined()) return t;
        return torch::where(torch::isfinite(t), t, torch::zeros_like(t));
    };

    try
    {
        RunModelReindex reindex = buildReindexTensors();
        RunModelProprioResult proprio_result = buildProprio(reindex, options, sanitize_tensor);
        torch::Tensor proprio = proprio_result.proprio;
        torch::Tensor default_dof_pos = proprio_result.default_dof_pos_reindexed;

        const auto t_infer_start = std::chrono::steady_clock::now();
        torch::Tensor depth_latent = updateDepthLatentAndYaw(proprio, options, sanitize_tensor);
        torch::Tensor history_flat = getHistoryFlatAndUpdate(proprio, options, sanitize_tensor);

        const auto t_policy_start = std::chrono::steady_clock::now();
        torch::Tensor actions = runPolicyForward(proprio, history_flat, depth_latent, options, sanitize_tensor);
        const auto t_policy_end = std::chrono::steady_clock::now();
        if (params_.log_inference_latency)
        {
            const double policy_ms = std::chrono::duration<double, std::milli>(t_policy_end - t_policy_start).count();
            latency_policy_ms_ema_ = latency_ema_initialized_ ? (0.9 * latency_policy_ms_ema_ + 0.1 * policy_ms) : policy_ms;
        }

        processActions(actions, default_dof_pos, reindex.dof_reindex_inv, options, sanitize_tensor);
        logInferenceLatency(t_infer_start);
        writeRobotCommandFromOutput();
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "Exception in StateRL::runModel(): %s", e.what());
        throw;
    }
}


void StateRL::setCommand()
{
    // 将 robot_command_ 写入 ros2_control 的 command interfaces。
    // 冷启动：前 warm_up_steps 步仅推理不写指令（与 Extreme-Parkour-Onboard warm_up 一致），之后按 dry_run 决定。
    try {
        if (inference_step_count_ < params_.warm_up_steps)
        {
            if (inference_step_count_ == 1)
            {
                RCLCPP_INFO(rclcpp::get_logger("StateRL"),
                    "Warm-up: inference only for first %d steps (no motor commands)", params_.warm_up_steps);
            }
            return;
        }
        if (!warm_up_done_logged_)
        {
            warm_up_done_logged_ = true;
            RCLCPP_INFO(rclcpp::get_logger("StateRL"),
                "Warm-up done (%d steps). %s", params_.warm_up_steps,
                params_.dry_run ? "dry_run enabled (inference only)" : "Sending motor commands.");
        }
        if (params_.dry_run)
        {
            RCLCPP_WARN_THROTTLE(
                rclcpp::get_logger("StateRL"),
                *node_->get_clock(),
                2000,
                "dry_run enabled: skip sending motor commands (inference only)");
            return;
        }
        if (static_cast<int>(ctrl_interfaces_.joint_position_command_interface_.size()) < params_.num_of_dofs ||
            static_cast<int>(ctrl_interfaces_.joint_velocity_command_interface_.size()) < params_.num_of_dofs ||
            static_cast<int>(ctrl_interfaces_.joint_kp_command_interface_.size()) < params_.num_of_dofs ||
            static_cast<int>(ctrl_interfaces_.joint_kd_command_interface_.size()) < params_.num_of_dofs ||
            static_cast<int>(ctrl_interfaces_.joint_torque_command_interface_.size()) < params_.num_of_dofs)
        {
            RCLCPP_WARN_THROTTLE(
                rclcpp::get_logger("StateRL"),
                *node_->get_clock(),
                2000,
                "setCommand: command interfaces size mismatch (pos=%zu vel=%zu kp=%zu kd=%zu tau=%zu, expected=%d)",
                ctrl_interfaces_.joint_position_command_interface_.size(),
                ctrl_interfaces_.joint_velocity_command_interface_.size(),
                ctrl_interfaces_.joint_kp_command_interface_.size(),
                ctrl_interfaces_.joint_kd_command_interface_.size(),
                ctrl_interfaces_.joint_torque_command_interface_.size(),
                params_.num_of_dofs);
        }
        
        const auto is_finite = [](double v) { return std::isfinite(v); };

        // 写入每个关节的目标位置/速度/增益/力矩
        for (int i = 0; i < 12; i++)
        {
            try {
                double q = robot_command_.motor_command.q[i];
                double dq = robot_command_.motor_command.dq[i];
                double kp = robot_command_.motor_command.kp[i];
                double kd = robot_command_.motor_command.kd[i];
                double tau = robot_command_.motor_command.tau[i];

                bool sanitized = false;
                if (!is_finite(q) && i < static_cast<int>(ctrl_interfaces_.joint_position_state_interface_.size()))
                {
                    q = ctrl_interfaces_.joint_position_state_interface_[i].get().get_value();
                    sanitized = true;
                }
                if (!is_finite(q))
                {
                    q = 0.0;
                    sanitized = true;
                }
                if (!is_finite(dq))
                {
                    dq = 0.0;
                    sanitized = true;
                }
                if (!is_finite(kp) || kp < 0.0)
                {
                    kp = (params_.rl_kp.numel() >= i + 1) ? params_.rl_kp[0][i].item<double>() : 0.0;
                    if (!is_finite(kp) || kp < 0.0) kp = 0.0;
                    sanitized = true;
                }
                if (!is_finite(kd) || kd < 0.0)
                {
                    kd = (params_.rl_kd.numel() >= i + 1) ? params_.rl_kd[0][i].item<double>() : 0.0;
                    if (!is_finite(kd) || kd < 0.0) kd = 0.0;
                    sanitized = true;
                }
                if (!is_finite(tau))
                {
                    tau = 0.0;
                    sanitized = true;
                }

                if (sanitized)
                {
                    RCLCPP_WARN_THROTTLE(
                        rclcpp::get_logger("StateRL"),
                        *node_->get_clock(),
                        1000,
                        "Non-finite motor command detected, sanitized for joint %d",
                        i);
                }

                if (i < static_cast<int>(ctrl_interfaces_.joint_position_command_interface_.size())) {
                    ctrl_interfaces_.joint_position_command_interface_[i].get().set_value(q);
                }
                
                if (i < static_cast<int>(ctrl_interfaces_.joint_velocity_command_interface_.size())) {
                    ctrl_interfaces_.joint_velocity_command_interface_[i].get().set_value(dq);
                }
                
                if (i < static_cast<int>(ctrl_interfaces_.joint_kp_command_interface_.size())) {
                    ctrl_interfaces_.joint_kp_command_interface_[i].get().set_value(kp);
                }
                
                if (i < static_cast<int>(ctrl_interfaces_.joint_kd_command_interface_.size())) {
                    ctrl_interfaces_.joint_kd_command_interface_[i].get().set_value(kd);
                }
                
                if (i < static_cast<int>(ctrl_interfaces_.joint_torque_command_interface_.size())) {
                    ctrl_interfaces_.joint_torque_command_interface_[i].get().set_value(tau);
                }
            } catch (const std::exception& e) {
                RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "Exception setting command for joint %d: %s", i, e.what());
            }
        }
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "Exception in StateRL::setCommand(): %s", e.what());
        throw;
    }
}


torch::Tensor StateRL::preprocessDepthImage(const sensor_msgs::msg::Image::SharedPtr msg) {
    // 深度图预处理（与训练完全一致，见《深度图处理对齐说明.md》）：
    // 1) ROS Image -> cv::Mat（支持 16UC1/32FC1），单位统一为米
    // 2) crop：由 depth_crop_* 配置；训练为固定 [:-2, 4:-4]，实机按分辨率设等效值
    // 3) clip 到 [0, max_depth]
    // 4) resize：cv::INTER_CUBIC 到 (depth_height, depth_width)，与训练 torchvision Resize(..., BICUBIC) 一致
    // 5) 归一化：depth / max_depth - 0.5 -> [-0.5, 0.5]，顺序与训练一致（先 resize 再 normalize）
        const int target_height = params_.depth_height > 0 ? params_.depth_height : 58;
        const int target_width = params_.depth_width > 0 ? params_.depth_width : 87;

    try {
        if (!msg) {
            return torch::zeros({1, target_height, target_width}, torch::kFloat32);
        }

        // 1) 将 ROS 图像消息转换为 OpenCV 矩阵（不拷贝数据，仅包装指针；后续 clone/convert 时会拷贝）
        cv::Mat depth_image;
        if (msg->encoding == "16UC1") {
            // 16位无符号整数深度图像（通常单位为毫米）
            depth_image = cv::Mat(msg->height, msg->width, CV_16UC1, const_cast<uint8_t*>(msg->data.data()), msg->step);
        } else if (msg->encoding == "32FC1") {
            // 32位浮点数深度图像（通常单位为米）
            depth_image = cv::Mat(msg->height, msg->width, CV_32FC1, const_cast<uint8_t*>(msg->data.data()), msg->step);
        } else {
            // 不支持的编码格式
            RCLCPP_WARN(rclcpp::get_logger("StateRL"), "Unsupported depth image encoding: %s", msg->encoding.c_str());
            return torch::zeros({1, target_height, target_width}, torch::kFloat32);
        }

        // 2) 统一到米（float32）
        cv::Mat depth_m;
        if (msg->encoding == "16UC1") {
            depth_image.convertTo(depth_m, CV_32F, 0.001);  // mm -> m
        } else {
            depth_m = depth_image.clone();
        }

        // training far_clip is cfg.depth.far_clip (default 2.0 in legged_robot_config.py)
        // our config.yaml currently sets max_depth. If it's set to 1.0 by mistake, prefer 2.0 for Extreme-Parkour.
        // NOTE: we keep using params_.max_depth, but enforce sane lower bound.
        const double max_depth_m = std::max(1e-6, params_.max_depth);

        double max_val = 0.0;
        cv::minMaxLoc(depth_m, nullptr, &max_val);
        if (max_val > 20.0) {
            depth_m = depth_m * 0.001f;
        }

        cv::patchNaNs(depth_m, static_cast<float>(max_depth_m));

        // If the incoming depth frame is mostly invalid (e.g., camera dropout),
        // do NOT overwrite the cached depth tensor; keep the last good frame.
        // This avoids sudden jumps to a constant "far" image which can destabilize the recurrent depth encoder.
        {
            cv::Mat invalid_mask = depth_m <= 1e-6f;
            const int invalid_cnt = cv::countNonZero(invalid_mask);
            const int total_cnt = depth_m.rows * depth_m.cols;
            const float invalid_ratio = total_cnt > 0 ? static_cast<float>(invalid_cnt) / static_cast<float>(total_cnt) : 1.0f;
            if (invalid_ratio > 0.98f) {
                RCLCPP_WARN_THROTTLE(
                    rclcpp::get_logger("StateRL"), *node_->get_clock(), 1000,
                    "Depth frame mostly invalid (%.1f%%). Keeping last cached depth tensor.",
                    invalid_ratio * 100.0f);
                std::lock_guard<std::mutex> lock(depth_mutex_);
                if (latest_depth_tensor_.defined() && latest_depth_tensor_.numel() > 0) {
                    // 使用上一帧缓存，允许继续走深度编码，避免 depth_latent 反复归零导致姿态崩溃
                    last_depth_frame_valid_.store(true);
                    return latest_depth_tensor_.to(torch::kCPU);
                }
                // 没有缓存：标记无效，回退为零深度
                last_depth_frame_valid_.store(false);
                // fallthrough: no cached frame yet
            } else {
                last_depth_frame_valid_.store(true);
            }
        }

        // 2) 裁剪：对齐 visual_extreme_parkour.py 默认裁剪
        int top = std::max(0, params_.depth_crop_top);
        int bottom = std::max(0, params_.depth_crop_bottom);
        int left = std::max(0, params_.depth_crop_left);
        int right = std::max(0, params_.depth_crop_right);
        cv::Mat cropped = depth_m;
        if (cropped.rows > top + bottom + 1 && cropped.cols > left + right + 1)
        {
            const cv::Rect roi(left, top, cropped.cols - left - right, cropped.rows - top - bottom);
            cropped = cropped(roi);
        }

        // 3) clip 到 [0, max_depth]（与训练 far_clip=2 一致）
        cv::max(cropped, 0.0, cropped);
        cv::min(cropped, max_depth_m, cropped);

        // 4) 先 resize 再归一化，与训练一致（legged_robot.py: resize_transform 再 normalize_depth_image）
        // 训练使用 torchvision Resize(..., BICUBIC)；此处用 OpenCV INTER_CUBIC 等价
        cv::Mat resized;
        cv::resize(cropped, resized, cv::Size(target_width, target_height), 0, 0, cv::INTER_CUBIC);
        torch::Tensor depth_tensor = torch::from_blob(
            resized.data, {resized.rows, resized.cols}, torch::kFloat32).clone();
        depth_tensor = depth_tensor.unsqueeze(0);  // [1,H,W]
        depth_tensor = depth_tensor / static_cast<float>(std::max(1e-6, max_depth_m)) - 0.5f;

        torch::Tensor depth_frame = depth_tensor;

        // 缓存归一化后的深度张量（CPU），供 runModel() 在短临界区内拷贝使用
        {
            std::lock_guard<std::mutex> lock(depth_mutex_);
            const int n = std::max(1, params_.depth_buffer_len);

            // push new frame
            depth_buffer_.push_back(depth_frame);
            while (static_cast<int>(depth_buffer_.size()) > n)
            {
                depth_buffer_.pop_front();
            }
            // bootstrap: if just started, repeat first frame to fill buffer
            while (static_cast<int>(depth_buffer_.size()) < n)
            {
                depth_buffer_.push_front(depth_frame);
            }

            if (n == 1)
            {
                // [1,H,W]
                latest_depth_tensor_ = depth_buffer_.back().unsqueeze(0).clone();
            }
            else
            {
                // [1,N,H,W]
                std::vector<torch::Tensor> frames;
                frames.reserve(depth_buffer_.size());
                for (const auto& f : depth_buffer_)
                {
                    frames.push_back(f);
                }
                latest_depth_tensor_ = torch::stack(frames, 0).unsqueeze(0).clone();
            }
        }

        // 返回预处理后的张量（CPU），调用方可用于可视化/调试（推理实际读取 latest_depth_tensor_）
        return latest_depth_tensor_.to(torch::kCPU);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "Exception in preprocessDepthImage: %s", e.what());
        return torch::zeros({1, target_height, target_width}, torch::kFloat32);
    }
}
