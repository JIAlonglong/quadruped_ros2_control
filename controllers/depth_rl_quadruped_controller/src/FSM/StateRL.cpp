//
// Created by biao on 24-10-6.
//

#include "depth_rl_quadruped_controller/FSM/StateRL.h"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/logging.hpp>
#include <yaml-cpp/yaml.h>

template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node& node)
{
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
    throw std::invalid_argument("Unsupported framework: " + framework);
}

StateRL::StateRL(CtrlInterfaces& ctrl_interfaces,
                 CtrlComponent& ctrl_component,
                 const std::vector<double>& target_pos) :
    FSMState(FSMStateName::RL, "rl", ctrl_interfaces),
    node_(ctrl_component.node_),
    enable_estimator_(ctrl_component.enable_estimator_),
    estimator_(ctrl_component.estimator_)
{
    if (!node_->has_parameter("robot_pkg")) {
        node_->declare_parameter("robot_pkg", robot_pkg_);
    }
    if (!node_->has_parameter("model_folder")) {
        node_->declare_parameter("model_folder", model_folder_);
    }
    if (!node_->has_parameter("use_rl_thread")) {
        node_->declare_parameter("use_rl_thread", use_rl_thread_);
    }

    robot_pkg_ = node_->get_parameter("robot_pkg").as_string();
    model_folder_ = node_->get_parameter("model_folder").as_string();
    use_rl_thread_ = node_->get_parameter("use_rl_thread").as_bool();

    RCLCPP_INFO(node_->get_logger(), "Using robot model from %s", robot_pkg_.c_str());
    const std::string package_share_directory = ament_index_cpp::get_package_share_directory(robot_pkg_);
    const std::string model_path = package_share_directory + "/config/" + model_folder_;

    for (int i = 0; i < 12; i++)
    {
        init_pos_[i] = target_pos[i];
    }

    // read params from yaml
    loadYaml(model_path);

    if (!params_.observations_history.empty())
    {
        history_obs_buf_ = std::make_shared<ObservationBuffer>(1, params_.num_observations,
                                                               params_.observations_history.size());
    }

    RCLCPP_INFO(node_->get_logger(), "Model loading: %s", params_.model_name.c_str());
    model_ = torch::jit::load(model_path + "/" + params_.model_name);


    // for (const auto &param: model_.parameters()) {
    //     std::cout << "Parameter dtype: " << param.dtype() << std::endl;
    // }


    if (use_rl_thread_)
    {
        rl_thread_ = std::thread([&]{
            while (true)
            {
                try
                {
                    executeAndSleep(
                        [&]
                        {
                            if (running_)
                            {
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
        setThreadPriority(60, rl_thread_);
    }
}

void StateRL::enter()
{
    // 初始化观测（确保形状与后续计算一致）
    // 基础观测
    obs_.lin_vel = torch::zeros({1, 3}, torch::kFloat32);  // 线速度 [1,3]（x,y,z）
    obs_.ang_vel = torch::zeros({1, 3}, torch::kFloat32);  // 角速度 [1,3]（roll,pitch,yaw轴）
    obs_.gravity_vec = torch::tensor({{0.0, 0.0, -1.0}}, torch::kFloat32);  // 重力向量 [1,3]（指向-z方向）
    obs_.commands = torch::zeros({1, 3}, torch::kFloat32);  // 控制命令 [1,3]（如x速度、y速度、yaw角速度）
    obs_.base_quat = torch::tensor({{0.0, 0.0, 0.0, 1.0}}, torch::kFloat32);  // 四元数 [1,4]（x,y,z,w，初始姿态）
    obs_.dof_pos = params_.default_dof_pos;  // 关节位置 [1, K]（K为关节数，复用默认姿态）
    obs_.dof_vel = torch::zeros({1, params_.num_of_dofs}, torch::kFloat32);  // 关节速度 [1, K]
    obs_.actions = torch::zeros({1, params_.num_of_dofs}, torch::kFloat32);  // 上一步动作 [1, K]

    // 新增：extreme_parkout 扩展观测
    obs_.roll = torch::tensor(0.0f, torch::kFloat32).unsqueeze(0);  // 横滚角 [1]（标量，批量维度为1）
    obs_.pitch = torch::tensor(0.0f, torch::kFloat32).unsqueeze(0);  // 俯仰角 [1]
    obs_.delta_yaw = torch::tensor(0.0f, torch::kFloat32).unsqueeze(0);  // 偏航角偏差 [1]
    obs_.delta_next_yaw = torch::tensor(0.0f, torch::kFloat32).unsqueeze(0);  // 下一个偏航角偏差 [1]
    obs_.contact_states = torch::zeros({1, params_.num_feet}, torch::kFloat32);  // 足部接触状态 [1, F]（F为足数，如4）
    obs_.env_class = torch::tensor(0.0f, torch::kFloat32).unsqueeze(0);  // 环境类别 [1]（初始为0）

    // 私有观测初始化（若需要）
    obs_.mass_params = torch::zeros({1, params_.mass_params_dim}, torch::kFloat32);  // 质量参数 [1, M]
    obs_.friction_coeffs = torch::zeros({1, params_.friction_dim}, torch::kFloat32);  // 摩擦系数 [1, C]
    obs_.motor_strength = torch::ones({1, 2}, torch::kFloat32);  // 电机强度 [1,2]（初始为1.0，与Python中-1偏移对应）

    // Init output
    output_torques = torch::zeros({1, params_.num_of_dofs});
    output_dof_pos_ = params_.default_dof_pos;

    // Init control
    control_.x = 0.0;
    control_.y = 0.0;
    control_.yaw = 0.0;

    // history
    if (!params_.observations_history.empty()) {
        history_obs_buf_->clear();
    }

    running_ = true;
}

void StateRL::run(const rclcpp::Time&/*time*/, const rclcpp::Duration&/*period*/)
{
    getState();
    if (!use_rl_thread_)
    {
        runModel();
    }
    setCommand();
}

void StateRL::exit()
{
    running_ = false;
}

FSMStateName StateRL::checkChange()
{
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

torch::Tensor StateRL::computeObservation()
{
    std::vector<torch::Tensor> obs_buf;
    std::vector<torch::Tensor> priv_explicit_list;
    std::vector<torch::Tensor> priv_latent_list;


    for (const std::string& observation : params_.observations)
    {
        if (observation == "lin_vel")
        {
            obs_buf.push_back(obs_.lin_vel * params_.lin_vel_scale);
        }
        else if (observation == "ang_vel")
        {
            obs_buf.push_back(
                quatRotateInverse(obs_.base_quat, obs_.ang_vel, params_.framework) * params_.ang_vel_scale);
        }
       else if (observation == "commands")
        {
            obs_buf.push_back(torch::zeros({obs_.commands.size(0), 2}, obs_.commands.options()));
            obs_buf.push_back(obs_.commands.index({torch::indexing::Slice(), 0}).unsqueeze(1) * params_.commands_scale);
        }
        else if (observation == "dof_pos")
        {
            obs_buf.push_back((obs_.dof_pos - params_.default_dof_pos) * params_.dof_pos_scale);
        }
        else if (observation == "dof_vel")
        {
            obs_buf.push_back(obs_.dof_vel * params_.dof_vel_scale);
        }
        else if (observation == "actions")
        {
            obs_buf.push_back(obs_.actions);
        }
        // 新增：extreme_parkout
        else if (observation == "roll")
        {
            obs_buf.push_back(obs_.roll.unsqueeze(1));
        }
        else if (observation == "pitch")
        {
            obs_buf.push_back(obs_.pitch.unsqueeze(1));
        }
        else if (observation == "delta_yaw")
        {
            obs_buf.push_back(torch::zeros_like(obs_.delta_yaw.unsqueeze(1)));
            obs_buf.push_back(obs_.delta_yaw.unsqueeze(1) * params_.delta_yaw_scale);
        }
        else if (observation == "delta_next_yaw")
        {
            obs_buf.push_back(obs_.delta_next_yaw.unsqueeze(1) * params_.delta_yaw_scale);
        }
        else if (observation == "env_class")
        {
            torch::Tensor env_not_17 = (obs_.env_class != 17.0).to(torch::kFloat32).unsqueeze(1);
            torch::Tensor env_is_17 = (obs_.env_class == 17.0).to(torch::kFloat32).unsqueeze(1);
            obs_buf.push_back(env_not_17);
            obs_buf.push_back(env_is_17);
        }
        else if (observation == "contact_states")
        {
            torch::Tensor contact_norm = obs_.contact_states - 0.5;  
            obs_buf.push_back(contact_norm);
        }
        // 
    }

    // 拼接基础观测量
    torch::Tensor obs = cat(obs_buf, 1);
    // 显式私有观测
    priv_explicit_list.push_back(obs_.lin_vel * params_.lin_vel_scale);  // 线速度缩放
    priv_explicit_list.push_back(torch::zeros_like(obs_.lin_vel));  // 占位符1
    priv_explicit_list.push_back(torch::zeros_like(obs_.lin_vel));  // 占位符2
    torch::Tensor priv_explicit = torch::cat(priv_explicit_list, 1);
    // 潜在私有观测
    priv_latent_list.push_back(obs_.mass_params);
    priv_latent_list.push_back(obs_.friction_coeffs);
    priv_latent_list.push_back((obs_.motor_strength[0] - 1.0).unsqueeze(1));
    priv_latent_list.push_back((obs_.motor_strength[1] - 1.0).unsqueeze(1));
    torch::Tensor priv_latent = torch::cat(priv_latent_list, 1);

    // 整合所有观测（基础观测 + 私有观测）
    std::vector<torch::Tensor> all_obs;
    all_obs.push_back(obs);
    all_obs.push_back(priv_explicit);
    all_obs.push_back(priv_latent);

    // 若使用相机
    // if (params_.use_camera) {
        // obs_list.push_back(obs_.depth_latent);
    // }

    // 最终拼接所有观测
    torch::Tensor final_obs = torch::cat(all_obs, 1);

    // 观测裁剪（对应 Python 中的 clamp）
    final_obs = torch::clamp(final_obs, -params_.clip_obs, params_.clip_obs);
    
    // 强制补全
    if (final_obs.size(1) < params_.num_observations) {
        int padding_size = params_.num_observations - final_obs.size(1);
        torch::Tensor padding = torch::zeros({final_obs.size(0), padding_size}, final_obs.options());
        final_obs = torch::cat({final_obs, padding}, 1);
        // 警告只出现一次
        static bool warning_printed = false;
        if (!warning_printed) {
            RCLCPP_WARN(rclcpp::get_logger("StateRL"), "obs size: %d, padding size: %d", obs.size(1), padding_size);
            warning_printed = true;
        }
    }

    return final_obs;
}

void StateRL::loadYaml(const std::string& config_path)
{
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(config_path + "/config.yaml");
    }
    catch ([[maybe_unused]] YAML::BadFile& e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("StateRL"), "The file '%s' does not exist", config_path.c_str());
        return;
    }

    params_.model_name = config["model_name"].as<std::string>();

    params_.model_name = config["model_name"].as<std::string>();
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
    if (config["clip_actions_lower"].IsNull() && config["clip_actions_upper"].IsNull())
    {
        params_.clip_actions_upper = torch::tensor({}).view({1, -1});
        params_.clip_actions_lower = torch::tensor({}).view({1, -1});
    }
    else
    {
        params_.clip_actions_upper = torch::tensor(
            ReadVectorFromYaml<double>(config["clip_actions_upper"], params_.framework, rows, cols)).view({1, -1});
        params_.clip_actions_lower = torch::tensor(
            ReadVectorFromYaml<double>(config["clip_actions_lower"], params_.framework, rows, cols)).view({1, -1});
    }
    params_.action_scale = config["action_scale"].as<double>();
    params_.hip_scale_reduction = config["hip_scale_reduction"].as<double>();
    params_.hip_scale_reduction_indices = ReadVectorFromYaml<int>(config["hip_scale_reduction_indices"]);
    params_.num_of_dofs = config["num_of_dofs"].as<int>();
    params_.lin_vel_scale = config["lin_vel_scale"].as<double>();
    params_.ang_vel_scale = config["ang_vel_scale"].as<double>();
    params_.dof_pos_scale = config["dof_pos_scale"].as<double>();
    params_.dof_vel_scale = config["dof_vel_scale"].as<double>();
    // 是否使用相机
    params_.use_camera = config["use_camera"].as<bool>(false);
    // params_.commands_scale = torch::tensor(ReadVectorFromYaml<double>(config["commands_scale"])).view({1, -1});
    params_.commands_scale = torch::tensor({params_.lin_vel_scale, params_.lin_vel_scale, params_.ang_vel_scale});
    params_.rl_kp = torch::tensor(ReadVectorFromYaml<double>(config["rl_kp"], params_.framework, rows, cols)).view({
        1, -1
    });
    params_.rl_kd = torch::tensor(ReadVectorFromYaml<double>(config["rl_kd"], params_.framework, rows, cols)).view({
        1, -1
    });
    params_.torque_limits = torch::tensor(
        ReadVectorFromYaml<double>(config["torque_limits"], params_.framework, rows, cols)).view({1, -1});

    params_.default_dof_pos = torch::from_blob(init_pos_, {12}, torch::kDouble).clone().to(torch::kFloat).unsqueeze(0);

    // params_.default_dof_pos = torch::tensor(
    //     ReadVectorFromYaml<double>(config["default_dof_pos"], params_.framework, rows, cols)).view({1, -1});
}

torch::Tensor StateRL::quatRotateInverse(const torch::Tensor& q, const torch::Tensor& v, const std::string& framework)
{
    torch::Tensor q_w;
    torch::Tensor q_vec;
    if (framework == "isaacsim")
    {
        q_w = q.index({torch::indexing::Slice(), 0});
        q_vec = q.index({torch::indexing::Slice(), torch::indexing::Slice(1, 4)});
    }
    else if (framework == "isaacgym")
    {
        q_w = q.index({torch::indexing::Slice(), 3});
        q_vec = q.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    }
    const c10::IntArrayRef shape = q.sizes();

    const torch::Tensor a = v * (2.0 * torch::pow(q_w, 2) - 1.0).unsqueeze(-1);
    const torch::Tensor b = cross(q_vec, v, -1) * q_w.unsqueeze(-1) * 2.0;
    const torch::Tensor c = q_vec * bmm(q_vec.view({shape[0], 1, 3}), v.view({shape[0], 3, 1})).squeeze(-1) * 2.0;
    return a - b + c;
}

torch::Tensor StateRL::forward()
{
    torch::autograd::GradMode::set_enabled(false);
    torch::Tensor clamped_obs = computeObservation();
    torch::Tensor actions;

    if (!params_.observations_history.empty())
    {
        history_obs_buf_->insert(clamped_obs);
        history_obs_ = history_obs_buf_->getObsVec(params_.observations_history);
        actions = model_.forward({history_obs_}).toTensor();
    }
    else
    {
        actions = model_.forward({clamped_obs}).toTensor();
    }

    if (params_.clip_actions_upper.defined() && params_.clip_actions_lower.defined())
    {
        actions = torch::clamp(actions, 
                              params_.clip_actions_lower.expand_as(actions), 
                              params_.clip_actions_upper.expand_as(actions));
    }
    return actions;
}

void StateRL::getState()
{
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

    control_.x = ctrl_interfaces_.control_inputs_.ly;
    control_.y = -ctrl_interfaces_.control_inputs_.lx;
    control_.yaw = -ctrl_interfaces_.control_inputs_.rx;

    updated_ = true;
}

void StateRL::runModel()
{
    if (enable_estimator_)
    {
        obs_.lin_vel = torch::from_blob(estimator_->getVelocity().data(), {3}, torch::kDouble).clone().
            to(torch::kFloat).unsqueeze(0);
    }
    obs_.ang_vel = torch::tensor(robot_state_.imu.gyroscope).unsqueeze(0);
    obs_.commands = torch::tensor({{control_.x, control_.y, control_.yaw}});
    obs_.base_quat = torch::tensor(robot_state_.imu.quaternion).unsqueeze(0);
    obs_.dof_pos = torch::tensor(robot_state_.motor_state.q).narrow(0, 0, params_.num_of_dofs).unsqueeze(0);
    obs_.dof_vel = torch::tensor(robot_state_.motor_state.dq).narrow(0, 0, params_.num_of_dofs).unsqueeze(0);

    const torch::Tensor clamped_actions = forward();

    for (const int i : params_.hip_scale_reduction_indices)
    {
        clamped_actions[0][i] *= params_.hip_scale_reduction;
    }

    obs_.actions = clamped_actions;

    const torch::Tensor actions_scaled = clamped_actions * params_.action_scale;
    // torch::Tensor output_torques = params_.rl_kp * (actions_scaled + params_.default_dof_pos - obs_.dof_pos) - params_.rl_kd * obs_.dof_vel;
    // output_torques = clamp(output_torques, -(params_.torque_limits), params_.torque_limits);

    output_dof_pos_ = actions_scaled + params_.default_dof_pos;

    for (int i = 0; i < params_.num_of_dofs; ++i)
    {
        robot_command_.motor_command.q[i] = output_dof_pos_[0][i].item<double>();
        robot_command_.motor_command.dq[i] = 0;
        robot_command_.motor_command.kp[i] = params_.rl_kp[0][i].item<double>();
        robot_command_.motor_command.kd[i] = params_.rl_kd[0][i].item<double>();
        robot_command_.motor_command.tau[i] = 0;
    }
}

void StateRL::setCommand() const
{
    for (int i = 0; i < 12; i++)
    {
        ctrl_interfaces_.joint_position_command_interface_[i].get().
                                                                            set_value(
                                                                                robot_command_.motor_command.q[i]);
        ctrl_interfaces_.joint_velocity_command_interface_[i].get().set_value(
            robot_command_.motor_command.dq[i]);
        ctrl_interfaces_.joint_kp_command_interface_[i].get().set_value(
            robot_command_.motor_command.kp[i]);
        ctrl_interfaces_.joint_kd_command_interface_[i].get().set_value(
            robot_command_.motor_command.kd[i]);
        ctrl_interfaces_.joint_torque_command_interface_[i].get().
                                                                          set_value(
                                                                              robot_command_.motor_command.tau[i]);
    }
}
