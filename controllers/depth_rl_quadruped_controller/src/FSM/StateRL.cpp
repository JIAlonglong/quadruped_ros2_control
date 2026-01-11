//
// Created by jialong on 25-9-2.
//

// 本文件实现 RL 控制状态（StateRL）。
// 该状态在每个控制周期中完成：
// 1) 从 ros2_control 的 state_interface 读取 IMU / 关节状态 / 遥控输入；
// 2) 按训练时约定拼接观测（proprio + 可选 depth latent + 历史）；
// 3) 调用 TorchScript 策略网络推理得到动作（actions）；
// 4) 将动作映射为目标关节位置/增益，并写入 ros2_control 的 command_interface。
//
// 代码同时兼容“主线程推理”和“独立 RL 线程推理”两种模式，后者用于降低 update() 的计算开销。

#include "depth_rl_quadruped_controller/FSM/StateRL.h"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/logging.hpp>
#include <yaml-cpp/yaml.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>  
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>

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

                    obs_.depth_latent = preprocessDepthImage(msg);
                    
                    cv::Mat depth_vis = depth_m.clone();
                    cv::medianBlur(depth_vis, depth_vis, 5);
                    const float far_clip = std::max(1e-6f, static_cast<float>(params_.max_depth));
                    cv::patchNaNs(depth_vis, far_clip);
                    cv::max(depth_vis, 0.0, depth_vis);
                    cv::min(depth_vis, far_clip, depth_vis);

                    cv_bridge::CvImage img_msg;
                    img_msg.header = msg->header;
                    img_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
                    img_msg.image = depth_vis;
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

    // 历史观测缓存清空（避免切换状态时把旧历史带入策略）
    if (!params_.observations_history.empty()) {
        history_obs_buf_->clear();
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
    // - 若启用 estimator 且安全检查失败，则立即切换到 PASSIVE
    // - 遥控 command=1/2 可切到 PASSIVE/FIXEDDOWN
    if (enable_estimator_ and !estimator_->safety())
    {
        return FSMStateName::PASSIVE;
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
    params_.delta_yaw_scale = config["delta_yaw_scale"].as<double>();
    params_.foot_force_threshold = config["foot_force_threshold"].as<double>(5.0); // 默认阈值设为5.0N
    
    // 维度参数：用于拼接观测、分配张量形状
    params_.num_feet = config["num_feet"].as<int>(4);  // 默认4足机器人
    params_.mass_params_dim = config["mass_params_dim"].as<int>(4);  // 默认质量参数维度
    params_.friction_dim = config["friction_dim"].as<int>(1);  // 默认摩擦系数维度
    
    // 观测布局：这些字段必须与训练导出的模型签名一致，否则会维度对不上或语义错位
    params_.num_proprio = config["num_proprio"].as<int>(53);
    params_.num_hist_len = config["num_hist_len"].as<int>(10);
    params_.num_scan = config["num_scan"].as<int>(0);
    params_.num_priv_explicit = config["num_priv_explicit"].as<int>(0);
    params_.num_priv_latent = config["num_priv_latent"].as<int>(0);
    RCLCPP_INFO(
        rclcpp::get_logger("StateRL"),
        "观测布局配置：proprio=%d, scan=%d, priv_explicit=%d, priv_latent=%d, hist_len=%d",
        params_.num_proprio, params_.num_scan, params_.num_priv_explicit,
        params_.num_priv_latent, params_.num_hist_len);

    // 相机相关参数：当 use_camera/use_depth_cnn 打开时，会尝试加载深度特征模型做编码
    // 注意：这里的 depth_feature_dim 与训练侧保持一致（Extreme Parkour 里通常是 32 维 latent + 2 维 yaw）
        params_.use_camera = config["use_camera"].as<bool>(false);
            params_.use_depth_cnn = config["use_depth_cnn"].as<bool>(false);
    if (params_.use_camera)
    {
        params_.depth_width = config["depth_width"].as<int>(58);   // 与 Python 中的 58x87 保持一致
        params_.depth_height = config["depth_height"].as<int>(87);
        params_.max_depth = config["max_depth"].as<double>(2.0);   // 有效深度范围上限
        params_.depth_cnn_model = config["depth_cnn_model"].as<std::string>("vision_weight.pt");
        params_.depth_yaw_clip = config["depth_yaw_clip"].as<double>(0.8);
        params_.depth_yaw_alpha = config["depth_yaw_alpha"].as<double>(0.9);
        
        // Extreme Parkour 的 TorchScript 视觉骨干输出 32 维 latent + 2 维偏航
        params_.depth_feature_dim = 32;
        
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

    // 默认关节角：使用 enter() 传入的 target_pos 初始化，保证与机器人当前默认站立姿态一致
    params_.default_dof_pos = torch::from_blob(init_pos_, {12}, torch::kDouble).clone().to(torch::kFloat).unsqueeze(0);

    // params_.default_dof_pos = torch::tensor(
    //     ReadVectorFromYaml<double>(config["default_dof_pos"], params_.framework, rows, cols)).view({1, -1});
}

torch::Tensor StateRL::quatRotateInverse(const torch::Tensor& q, const torch::Tensor& v, const std::string& framework)
{
    // 坐标变换占位函数：
    // 部分项目会在这里把世界系向量旋转到机体系（或做 inverse rotate），以对齐训练坐标系。
    // 当前假设训练与在线使用坐标系一致，因此直接返回原向量。
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

        // 遥控输入映射：此处的轴向/正负号与训练脚本的约定有关（需要与 teleop 或手柄节点对齐）
        control_.x = ctrl_interfaces_.control_inputs_.ly;
        control_.y = -ctrl_interfaces_.control_inputs_.lx;
        control_.yaw = -ctrl_interfaces_.control_inputs_.rx;

        updated_ = true;
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "Exception in StateRL::getState(): %s", e.what());
        throw;
    }
}

void StateRL::runModel()
{
    // 执行一次策略推理（不写接口，只写 robot_command_ 缓存）：
    // 1) 由 robot_state_ / control_ 构建 proprio
    // 2) （可选）调用 depth encoder 得到 depth_latent 与 yaw 修正
    // 3) 更新历史缓存并拼接 policy_obs
    // 4) 调用 TorchScript policy forward 得到 actions
    // 5) 对 actions 裁剪/缩放，并映射为目标关节角（output_dof_pos_）
    // 6) 写入 robot_command_，供 setCommand() 写入硬件接口
    if (!model_loaded_)
    {
        RCLCPP_WARN_THROTTLE(rclcpp::get_logger("StateRL"), *node_->get_clock(), 2000,
                             "Policy module not loaded, skip runModel.");
        return;
    }

    // 推理阶段无需梯度信息，显式关闭可减少开销
    torch::NoGradGuard no_grad;
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    const auto sanitize_tensor = [&](const torch::Tensor& t) -> torch::Tensor {
        if (!t.defined())
        {
            return t;
        }
        return torch::where(torch::isfinite(t), t, torch::zeros_like(t));
    };

    try
    {
        const torch::Tensor dof_reindex = torch::tensor(
            {3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8},
            torch::TensorOptions().dtype(torch::kLong).device(device_));
        const torch::Tensor feet_reindex = torch::tensor(
            {1, 0, 3, 2},
            torch::TensorOptions().dtype(torch::kLong).device(device_));

        // ==================== 1. 构建 proprio 观测（基础本体观测） ====================
        // 这里把 IMU、指令、关节位姿/速度、上一动作、接触估计等拼接成训练时所需的 proprio 向量
        std::array<float, 3> ang_vel_arr{
            static_cast<float>(robot_state_.imu.gyroscope[0]),
            static_cast<float>(robot_state_.imu.gyroscope[1]),
            static_cast<float>(robot_state_.imu.gyroscope[2])
        };
        torch::Tensor ang_vel = torch::from_blob(ang_vel_arr.data(), {1, 3}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
        ang_vel = ang_vel * static_cast<float>(params_.ang_vel_scale);

        // 用四元数计算 RPY（roll/pitch/yaw），用于对齐训练时的姿态观测格式
        tf2::Quaternion base_quat_tf(robot_state_.imu.quaternion[0],
                                     robot_state_.imu.quaternion[1],
                                     robot_state_.imu.quaternion[2],
                                     robot_state_.imu.quaternion[3]);
        tf2::Matrix3x3 rot_mat(base_quat_tf);
        double roll, pitch, yaw;
        rot_mat.getRPY(roll, pitch, yaw);
        
        // IMU 观测：这里只取 roll/pitch（是否需要 yaw 取决于训练侧设计）
        torch::Tensor imu = sanitize_tensor(torch::tensor({{static_cast<float>(roll), static_cast<float>(pitch)}}, options));
        
        torch::Tensor yaw_info = torch::zeros({1, 3}, options);

        const float lin_vel_deadband = 0.1f;
        const float cmd_px_max = 0.5f;
        const float cmd_nx_max = 0.5f;

        float vx = 0.0f;
        const float ly = static_cast<float>(ctrl_interfaces_.control_inputs_.ly);
        if (ly > lin_vel_deadband)
        {
            vx = (ly - lin_vel_deadband) / (1.0f - lin_vel_deadband);
            vx = vx * cmd_px_max;
        }
        else if (ly < -lin_vel_deadband)
        {
            vx = (ly + lin_vel_deadband) / (1.0f - lin_vel_deadband);
            vx = vx * cmd_nx_max;
        }

        torch::Tensor commands = torch::tensor({{0.0f, 0.0f, vx}}, options);
        torch::Tensor env_class = torch::tensor({{1.0f, 0.0f}}, options);

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

        torch::Tensor last_actions_raw = obs_.actions.numel() == 0
            ? torch::zeros({1, params_.num_of_dofs}, options)
            : sanitize_tensor(obs_.actions.to(options));
        torch::Tensor last_actions = last_actions_raw.index_select(1, dof_reindex);

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
            for (int i = 0; i < n; ++i)
            {
                const float v = static_cast<float>(ctrl_interfaces_.foot_force_state_interface_[i].get().get_value());
                const bool contact_now = (v <= 1.5f)
                    ? (v > 1.0f)
                    : (v > static_cast<float>(params_.foot_force_threshold));
                contact_bool[0][i] = contact_now;
            }
            if (n < params_.num_feet)
            {
                contact_bool.index_put_({0, torch::indexing::Slice(n, params_.num_feet)}, false);
            }
        }

        if (!last_contact_bool_.defined() || last_contact_bool_.sizes() != contact_bool.sizes())
        {
            last_contact_bool_ = torch::zeros_like(contact_bool);
        }
        torch::Tensor contact_filt_bool = contact_bool | last_contact_bool_;
        last_contact_bool_ = contact_bool.clone();

        torch::Tensor contact_raw = torch::where(
            contact_filt_bool,
            torch::full({1, params_.num_feet}, 0.5f, options),
            torch::full({1, params_.num_feet}, -0.5f, options));

        torch::Tensor contact = contact_raw.index_select(1, feet_reindex);
        obs_.ang_vel = ang_vel.to(torch::kCPU);
        obs_.commands = commands.to(torch::kCPU);
        obs_.dof_pos = joint_pos.to(torch::kCPU);
        obs_.dof_vel = joint_vel.to(torch::kCPU);
        obs_.roll = torch::tensor(static_cast<float>(roll)).unsqueeze(0);
        obs_.pitch = torch::tensor(static_cast<float>(pitch)).unsqueeze(0);
        obs_.delta_yaw = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32));
        obs_.contact_states = contact.to(torch::kCPU);

        // 本体拼接
        torch::Tensor proprio = torch::cat(
            {ang_vel, imu, yaw_info, commands, env_class, joint_pos, joint_vel, last_actions, contact}, 1);
        proprio = sanitize_tensor(proprio.to(device_));

        // ==================== 2. 深度特征提取（depth encoder -> depth_latent + yaw） ====================
        // 若启用了相机并已加载 depth encoder：把最新深度张量与 proprio 一起输入，得到 depth latent；
        // 同时可从输出末尾取两维 yaw 修正写回 proprio 的 slice（训练时的约定）。
        torch::Tensor depth_latent = torch::zeros({1, params_.depth_feature_dim}, options);
        if (params_.use_camera && params_.use_depth_cnn && depth_model_loaded_)
        {
            try
            {
                // 为了线程安全：从共享 latest_depth_tensor_ 中短临界区拷贝一份到本地变量
                torch::Tensor latest_depth_copy;
                {
                    std::lock_guard<std::mutex> lock(depth_mutex_);
                    if (latest_depth_tensor_.defined() && latest_depth_tensor_.numel() > 0)
                    {
                        latest_depth_copy = latest_depth_tensor_.clone();
                    }
                }
                if (latest_depth_copy.defined() && latest_depth_copy.numel() > 0)
                {
                    std::vector<torch::jit::IValue> depth_inputs;
                    depth_inputs.emplace_back(latest_depth_copy.to(device_));
                    depth_inputs.emplace_back(proprio);
                    torch::Tensor depth_output = sanitize_tensor(depth_encoder_module_.forward(depth_inputs).toTensor());

                    if (depth_output.sizes().size() == 2 && depth_output.size(0) == 1 &&
                        depth_output.size(1) >= params_.depth_feature_dim + 2)
                    {
                        depth_latent = depth_output.index({0, torch::indexing::Slice(0, params_.depth_feature_dim)}).unsqueeze(0);
                        torch::Tensor yaw_from_depth =
                            sanitize_tensor(depth_output.index({0, torch::indexing::Slice(params_.depth_feature_dim,
                                                                                         params_.depth_feature_dim + 2)}).unsqueeze(0) * 1.5f);

                        const float yaw_clip = static_cast<float>(params_.depth_yaw_clip);
                        yaw_from_depth = torch::clamp(yaw_from_depth, -yaw_clip, yaw_clip);

                        const float alpha = static_cast<float>(params_.depth_yaw_alpha);
                        if (!has_depth_yaw_filtered_ ||
                            !depth_yaw_filtered_.defined() ||
                            depth_yaw_filtered_.sizes() != yaw_from_depth.sizes())
                        {
                            depth_yaw_filtered_ = yaw_from_depth.clone();
                            has_depth_yaw_filtered_ = true;
                        }
                        else
                        {
                            depth_yaw_filtered_ = depth_yaw_filtered_ * alpha +
                                                  yaw_from_depth * (1.0f - alpha);
                        }

                        proprio.index_put_({torch::indexing::Slice(), torch::indexing::Slice(6, 8)}, depth_yaw_filtered_);
                    }
                    else
                    {
                        RCLCPP_WARN_THROTTLE(rclcpp::get_logger("StateRL"), *node_->get_clock(), 2000,
                                             "Depth encoder output shape unexpected.");
                    }
                }
            }
            catch (const std::exception& e)
            {
                RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "Depth encoder forward failed: %s", e.what());
            }
        }
        // 记录 depth_latent（CPU）用于调试/可视化
        obs_.depth_latent = depth_latent.to(torch::kCPU);

        // ==================== 3. 更新历史观测缓存（ObservationBuffer） ====================
        // 把当前 proprio 写入历史环形缓存，再按 observations_history 取出并扁平化
        torch::Tensor proprio_cpu = sanitize_tensor(proprio.to(torch::kCPU));
        torch::Tensor proprio_history = proprio_cpu.clone();
        // 训练时也把proprio_history中的6-8为占零
        proprio_history.index_put_({torch::indexing::Slice(), torch::indexing::Slice(6, 8)}, 0.0f);
        history_obs_buf_->insert(proprio_history);

        std::vector<int> history_indices = params_.observations_history;
        if (history_indices.empty())
        {
            history_indices.resize(std::max(params_.num_hist_len, 1));
            std::iota(history_indices.begin(), history_indices.end(), 0);
        }
        torch::Tensor history_flat = sanitize_tensor(history_obs_buf_->getObsVec(history_indices).to(device_));

        const int policy_obs_dim =
            params_.num_proprio +
            params_.num_scan +
            params_.num_priv_explicit +
            params_.num_priv_latent +
            params_.num_hist_len * params_.num_proprio;

        torch::Tensor policy_obs = torch::zeros({1, policy_obs_dim}, options);
        policy_obs.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, params_.num_proprio)}, proprio);

        if (params_.num_hist_len > 0)
        {
            const int history_offset = policy_obs_dim - params_.num_hist_len * params_.num_proprio;
            policy_obs.index_put_(
                {torch::indexing::Slice(), torch::indexing::Slice(history_offset, policy_obs_dim)}, history_flat);
        }

        std::vector<torch::jit::IValue> policy_inputs;
        policy_inputs.emplace_back(policy_obs);
        policy_inputs.emplace_back(depth_latent);
        torch::Tensor actions = sanitize_tensor(policy_module_.forward(policy_inputs).toTensor().to(device_));

        torch::Tensor actions_clamped = actions;
        if (params_.clip_actions_upper.numel() == params_.clip_actions_lower.numel() &&
            params_.clip_actions_upper.numel() == params_.num_of_dofs)
        {
            actions_clamped = torch::max(
                torch::min(actions, params_.clip_actions_upper.to(options)),
                params_.clip_actions_lower.to(options));
        }

        for (const int index : params_.hip_scale_reduction_indices)
        {
            if (index >= 0 && index < params_.num_of_dofs)
            {
                actions_clamped[0][index] *= static_cast<float>(params_.hip_scale_reduction);
            }
        }

        const float action_alpha = static_cast<float>(params_.action_filter_alpha);
        if (!has_actions_filtered_ ||
            !actions_filtered_.defined() ||
            actions_filtered_.sizes() != actions_clamped.sizes())
        {
            actions_filtered_ = actions_clamped.clone();
            has_actions_filtered_ = true;
        }
        else
        {
            actions_filtered_ = actions_filtered_ * action_alpha +
                                actions_clamped * (1.0f - action_alpha);
        }

        torch::Tensor actions_scaled = sanitize_tensor(actions_filtered_ * static_cast<float>(params_.action_scale));
        torch::Tensor output_dof_pos_policy = sanitize_tensor(actions_scaled + default_dof_pos);
        output_dof_pos_ = sanitize_tensor(output_dof_pos_policy.index_select(1, dof_reindex));

        if (transition_percent_ < 1.5)
        {
            transition_percent_ += 1.0 / transition_duration_;
            const double phase = std::tanh(transition_percent_);
            torch::Tensor start_pos_tensor = torch::from_blob(
                const_cast<double*>(start_pos_), {1, params_.num_of_dofs}, torch::TensorOptions().dtype(torch::kFloat64))
                                             .to(options);
            output_dof_pos_ = output_dof_pos_ * phase + start_pos_tensor * (1.0 - phase);
        }

        obs_.actions = actions_filtered_.index_select(1, dof_reindex).to(torch::kCPU);

        // ==================== 6. 写入电机命令（robot_command_ -> joint interfaces） ====================
        // 这里只写 robot_command_ 缓存；真正写 ros2_control 接口在 setCommand() 完成
        for (int i = 0; i < params_.num_of_dofs; ++i)
        {
            robot_command_.motor_command.q[i] = output_dof_pos_[0][i].item<double>();
            robot_command_.motor_command.dq[i] = 0.0;
            robot_command_.motor_command.kp[i] = params_.rl_kp[0][i].item<double>();
            robot_command_.motor_command.kd[i] = params_.rl_kd[0][i].item<double>();
            robot_command_.motor_command.tau[i] = 0.0;
        }
        
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "Exception in StateRL::runModel(): %s", e.what());
        throw;
    }
}


void StateRL::setCommand() const
{
    // 将 robot_command_ 写入 ros2_control 的 command interfaces。
    // 为了提高鲁棒性，这里对接口数组大小做边界检查，并对每个关节 set_value() 做异常捕获。
    try {
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
    // 深度图预处理（尽量与训练侧保持一致）：
    // 1) ROS Image -> cv::Mat（支持 16UC1/32FC1）
        // 2) crop：对齐训练侧 crop_depth_image(depth_image[:-2, 4:-4])
        // 3) resize：对齐训练侧 torchvision.transforms.Resize(..., BICUBIC)
        // 4) normalize：depth=(depth-near)/(far-near)-0.5
        // 5) 转为 Torch Tensor：[1,H,W]，并缓存到 latest_depth_tensor_
        constexpr int kTargetHeight = 58;
        constexpr int kTargetWidth = 87;
    
    try {
        if (!msg) {
            return torch::zeros({1, kTargetHeight, kTargetWidth}, torch::kFloat32);
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
            return torch::zeros({1, kTargetHeight, kTargetWidth}, torch::kFloat32);
        }

        // 2) 统一到米（float32）
        cv::Mat depth_m;
        if (msg->encoding == "16UC1") {
            depth_image.convertTo(depth_m, CV_32F, 0.001);  // mm -> m
        } else {
            depth_m = depth_image.clone();
        }

        const double max_depth_m = std::max(1e-6, params_.max_depth);

        double max_val = 0.0;
        cv::minMaxLoc(depth_m, nullptr, &max_val);
        if (max_val > 20.0) {
            depth_m = depth_m * 0.001f;
        }

        cv::patchNaNs(depth_m, static_cast<float>(max_depth_m));

        cv::Mat cropped = depth_m;
        if (cropped.rows > 2 && cropped.cols > 8) {
            const cv::Rect roi(4, 0, cropped.cols - 8, cropped.rows - 2);
            cropped = cropped(roi);
        }

        cv::Mat resized;
        cv::resize(cropped, resized, cv::Size(kTargetWidth, kTargetHeight), 0.0, 0.0, cv::INTER_CUBIC);

        cv::max(resized, 0.0f, resized);
        cv::min(resized, static_cast<float>(max_depth_m), resized);
        resized = resized / static_cast<float>(max_depth_m) - 0.5f;
        
        // 4) 转换为 PyTorch 张量，并添加 batch 维度，得到 [1,H,W]
        torch::Tensor depth_tensor = torch::from_blob(resized.data, {kTargetHeight, kTargetWidth}, torch::kFloat32)
                                         .unsqueeze(0)
                                         .clone();

        // 缓存归一化后的深度张量（CPU），供 runModel() 在短临界区内拷贝使用
        {
            std::lock_guard<std::mutex> lock(depth_mutex_);
            latest_depth_tensor_ = depth_tensor.clone();
        }

        // 返回预处理后的张量（CPU），调用方可用于可视化/调试（推理实际读取 latest_depth_tensor_）
        return latest_depth_tensor_.to(torch::kCPU);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "Exception in preprocessDepthImage: %s", e.what());
        return torch::zeros({1, kTargetHeight, kTargetWidth}, torch::kFloat32);
    }
}
