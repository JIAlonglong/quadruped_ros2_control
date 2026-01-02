//
// Created by tlab-uav on 24-10-4.
//

#include "RlQuadrupedController.h"
#include <rclcpp/logging.hpp>

namespace depth_rl_quadruped_controller
{
    using config_type = controller_interface::interface_configuration_type;

    controller_interface::InterfaceConfiguration LeggedGymController::command_interface_configuration() const
    {
        controller_interface::InterfaceConfiguration conf = {config_type::INDIVIDUAL, {}};

        conf.names.reserve(joint_names_.size() * command_interface_types_.size());
        for (const auto& joint_name : joint_names_)
        {
            for (const auto& interface_type : command_interface_types_)
            {
                // 使用命名空间前缀（若存在）拼接完整的关节名，避免出现重复“/=”的错误格式
                const std::string full_joint_name = !command_prefix_.empty()
                    ? command_prefix_ + "/" + joint_name
                    : joint_name;
                conf.names.push_back(full_joint_name + "/" + interface_type);
            }
        }

        return conf;
    }

    controller_interface::InterfaceConfiguration LeggedGymController::state_interface_configuration() const
    {
        controller_interface::InterfaceConfiguration conf = {config_type::INDIVIDUAL, {}};

        conf.names.reserve(joint_names_.size() * state_interface_types_.size());
        for (const auto& joint_name : joint_names_)
        {
            for (const auto& interface_type : state_interface_types_)
            {
                // 统一按“关节/接口”格式注册状态接口
                conf.names.push_back(joint_name + "/" + interface_type);
            }
        }

        for (const auto& interface_type : imu_interface_types_)
        {
            // IMU 传感器同样按“传感器名/接口”格式填充
            conf.names.push_back(imu_name_ + "/" + interface_type);
        }

        for (const auto& interface_type : foot_force_interface_types_)
        {
            // 足端力传感器接口也遵循相同命名规则
            conf.names.push_back(foot_force_name_ + "/" + interface_type);
        }

        return conf;
    }

    controller_interface::return_type LeggedGymController::
    update(const rclcpp::Time& time, const rclcpp::Duration& period)
    {
        if (ctrl_component_.enable_estimator_)
        {
            if (ctrl_component_.robot_model_ == nullptr)
            {
                return controller_interface::return_type::OK;
            }
            // 加载机器人模型
            ctrl_component_.robot_model_->update();
            // 估计机器人状态
            ctrl_component_.estimator_->update();
        }
        // 机器人状态机
        if (mode_ == FSMMode::NORMAL)
        {
            current_state_->run(time, period);
            next_state_name_ = current_state_->checkChange();
            if (next_state_name_ != current_state_->state_name)
            {
                mode_ = FSMMode::CHANGE;
                next_state_ = getNextState(next_state_name_);
                RCLCPP_INFO(get_node()->get_logger(), "Switched from %s to %s",
                            current_state_->state_name_string.c_str(), next_state_->state_name_string.c_str());
            }
        }
        else if (mode_ == FSMMode::CHANGE)
        {
            current_state_->exit();
            current_state_ = next_state_;

            current_state_->enter();
            mode_ = FSMMode::NORMAL;
        }

        return controller_interface::return_type::OK;
    }

    controller_interface::CallbackReturn LeggedGymController::on_init()
    {
        try
        {
            // auto_declare: 自动声明参数并且获取参数
            joint_names_ = auto_declare<std::vector<std::string>>("joints", joint_names_);
            feet_names_ = auto_declare<std::vector<std::string>>("feet_names", feet_names_);
            command_interface_types_ =
                auto_declare<std::vector<std::string>>("command_interfaces", command_interface_types_);
            state_interface_types_ =
                auto_declare<std::vector<std::string>>("state_interfaces", state_interface_types_);

            command_prefix_ = auto_declare<std::string>("command_prefix", command_prefix_);
            base_name_ = auto_declare<std::string>("base_name", base_name_);

            // imu sensor
            imu_name_ = auto_declare<std::string>("imu_name", imu_name_);
            imu_interface_types_ = auto_declare<std::vector<std::string>>("imu_interfaces", state_interface_types_);

            // foot_force_sensor
            foot_force_name_ = auto_declare<std::string>("foot_force_name", foot_force_name_);
            foot_force_interface_types_ =
                auto_declare<std::vector<std::string>>("foot_force_interfaces", foot_force_interface_types_);
            feet_force_threshold_ = auto_declare<double>("feet_force_threshold", feet_force_threshold_);

            // pose parameters
            down_pos_ = auto_declare<std::vector<double>>("down_pos", down_pos_);
            stand_pos_ = auto_declare<std::vector<double>>("stand_pos", stand_pos_);
            stand_kp_ = auto_declare<double>("stand_kp", stand_kp_);
            stand_kd_ = auto_declare<double>("stand_kd", stand_kd_);

            // 验证stand_pos_参数长度
            if (stand_pos_.size() != 12) {
                RCLCPP_FATAL(get_node()->get_logger(), "Invalid stand_pos_ size: %zu (expected 12)", stand_pos_.size());
                return controller_interface::CallbackReturn::ERROR;
            }

            get_node()->get_parameter("update_rate", ctrl_interfaces_.frequency_);
            RCLCPP_INFO(get_node()->get_logger(), "Controller Update Rate: %d Hz", ctrl_interfaces_.frequency_);

            // 当四足机器人每只脚都有对应的力传感器，启用估计器
            if (foot_force_interface_types_.size() == 4)
            {
                RCLCPP_INFO(get_node()->get_logger(), "Enable Estimator");
                ctrl_component_.enable_estimator_ = true;
                ctrl_component_.estimator_ = std::make_shared<Estimator>(ctrl_interfaces_, ctrl_component_);
            }
            ctrl_component_.node_ = get_node();
            // 添加参数加载完成调试日志
            RCLCPP_INFO(get_node()->get_logger(), "All parameters loaded successfully");
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
            return controller_interface::CallbackReturn::ERROR;
        }

        return CallbackReturn::SUCCESS;
    }

    controller_interface::CallbackReturn LeggedGymController::on_configure(
        const rclcpp_lifecycle::State& previous_state)
    {
        // 添加状态转换调试日志
        RCLCPP_INFO(get_node()->get_logger(), "Transitioning from state %s to configuring", previous_state.label().c_str());
        robot_description_subscription_ = get_node()->create_subscription<std_msgs::msg::String>(
            "/robot_description", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local(),
            [this](const std_msgs::msg::String::SharedPtr msg)
            {
                RCLCPP_INFO(get_node()->get_logger(), "Received robot description, size: %zu bytes", msg->data.size());
                if (ctrl_component_.enable_estimator_)
                {
                    try {
                        ctrl_component_.robot_model_ = std::make_shared<QuadrupedRobot>(
                            ctrl_interfaces_, msg->data, feet_names_, base_name_);
                        RCLCPP_INFO(get_node()->get_logger(), "Successfully created QuadrupedRobot instance");
                        robot_model_loaded_ = true; // 添加此行设置标志位
                    } catch (const std::exception& e) {
                        RCLCPP_ERROR(get_node()->get_logger(), "Failed to create QuadrupedRobot: %s", e.what());
                    }
                }
            });

        // 添加模型加载等待机制 (3秒超时)
        rclcpp::Rate rate(10); // 10Hz轮询频率
        const int max_attempts = 30; // 最多尝试30次 (3秒)
        int attempt = 0;
        while (rclcpp::ok() && !robot_model_loaded_ && attempt < max_attempts)
        {
            RCLCPP_INFO(get_node()->get_logger(), "等待机器人模型加载 (尝试 %d/%d)", attempt+1, max_attempts);
            rate.sleep();
            attempt++;
        }

        // 超时检查
        if (!robot_model_loaded_)
        {
            RCLCPP_ERROR(get_node()->get_logger(), "等待机器人模型加载超时");
            return CallbackReturn::ERROR;
        }

        control_input_subscription_ = get_node()->create_subscription<control_input_msgs::msg::Inputs>(
            "/control_input", 10, [this](const control_input_msgs::msg::Inputs::SharedPtr msg)
            {
                // Handle message
                ctrl_interfaces_.control_inputs_.command = msg->command;
                ctrl_interfaces_.control_inputs_.lx = msg->lx;
                ctrl_interfaces_.control_inputs_.ly = msg->ly;
                ctrl_interfaces_.control_inputs_.rx = msg->rx;
                ctrl_interfaces_.control_inputs_.ry = msg->ry;
            });

        return CallbackReturn::SUCCESS;
    }


    controller_interface::CallbackReturn LeggedGymController::on_activate(
        const rclcpp_lifecycle::State& previous_state)
    {
        // 添加激活状态调试日志
        RCLCPP_INFO(get_node()->get_logger(), "Activating controller from state %s", previous_state.label().c_str());
         // clear out vectors in case of restart
        ctrl_interfaces_.clear();

        // assign command interfaces
        for (auto& interface : command_interfaces_)
        {
            std::string interface_name = interface.get_interface_name();
            if (const size_t pos = interface_name.find('/'); pos != std::string::npos)
            {
                command_interface_map_[interface_name.substr(pos + 1)]->push_back(interface);
            }
            else
            {
                command_interface_map_[interface_name]->push_back(interface);
            }
        }

        // assign state interfaces
        for (auto& interface : state_interfaces_)
        {
            if (interface.get_prefix_name() == imu_name_)
            {
                RCLCPP_INFO_STREAM(rclcpp::get_logger("depth_rl_quadruped_controller"), "IMU Interface: " << interface.get_interface_name() << ", Type: " << interface.get_value());
                ctrl_interfaces_.imu_state_interface_.emplace_back(interface);
            }
            else if (interface.get_prefix_name() == foot_force_name_)
            {
                RCLCPP_INFO_STREAM(rclcpp::get_logger("depth_rl_quadruped_controller"), "Foot Force Interface: " << interface.get_interface_name() << ", Type: " << interface.get_value());
                ctrl_interfaces_.foot_force_state_interface_.emplace_back(interface);
            }
            else
            {
                RCLCPP_INFO_STREAM(rclcpp::get_logger("depth_rl_quadruped_controller"), "Other State Interface: " << interface.get_interface_name() << ", Type: " << interface.get_value());
                state_interface_map_[interface.get_interface_name()]->push_back(interface);
            }
        }

        // Create FSM List
        state_list_.passive = std::make_shared<StatePassive>(ctrl_interfaces_);
        state_list_.fixedDown = std::make_shared<StateFixedDown>(ctrl_interfaces_, down_pos_, stand_kp_, stand_kd_);
        state_list_.fixedStand = std::make_shared<StateFixedStand>(ctrl_interfaces_, stand_pos_, stand_kp_, stand_kd_);
        // 这里执行失败
        RCLCPP_INFO(get_node()->get_logger(), "Initializing StateRL with stand_pos: [%f, %f, %f]", stand_pos_[0], stand_pos_[1], stand_pos_[2]);
        state_list_.rl = std::make_shared<StateRL>(ctrl_interfaces_, ctrl_component_, std::vector<double>(stand_pos_.begin(), stand_pos_.end()));
        RCLCPP_INFO(get_node()->get_logger(), "StateRL initialized successfully");
        // Initialize FSM
        current_state_ = state_list_.passive;
        RCLCPP_INFO(get_node()->get_logger(), "Set initial state to: %s", current_state_->state_name_string.c_str());

        try {
            current_state_->enter();
            RCLCPP_INFO(get_node()->get_logger(), "Successfully entered initial state: %s", current_state_->state_name_string.c_str());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_node()->get_logger(), "Exception during state enter(): %s", e.what());
            throw;
        }
        mode_ = FSMMode::NORMAL;
        next_state_ = current_state_;
        next_state_name_ = current_state_->state_name;
        mode_ = FSMMode::NORMAL;
        // 添加激活完成调试日志
        RCLCPP_INFO(get_node()->get_logger(), "Controller activated successfully, current state: %s", current_state_->state_name_string.c_str());

        return CallbackReturn::SUCCESS;
    }

    controller_interface::CallbackReturn LeggedGymController::on_deactivate(
        const rclcpp_lifecycle::State& /*previous_state*/)
    {
        release_interfaces();
        return CallbackReturn::SUCCESS;
    }

    controller_interface::CallbackReturn
    LeggedGymController::on_cleanup(const rclcpp_lifecycle::State& previous_state)
    {
        return ControllerInterface::on_cleanup(previous_state);
    }

    controller_interface::CallbackReturn
    LeggedGymController::on_shutdown(const rclcpp_lifecycle::State& previous_state)
    {
        return ControllerInterface::on_shutdown(previous_state);
    }

    controller_interface::CallbackReturn LeggedGymController::on_error(const rclcpp_lifecycle::State& previous_state)
    {
        return ControllerInterface::on_error(previous_state);
    }

    std::shared_ptr<FSMState> LeggedGymController::getNextState(const FSMStateName stateName) const
    {
        switch (stateName)
        {
        case FSMStateName::INVALID:
            return state_list_.invalid;
        case FSMStateName::PASSIVE:
            return state_list_.passive;
        case FSMStateName::FIXEDDOWN:
            return state_list_.fixedDown;
        case FSMStateName::FIXEDSTAND:
            return state_list_.fixedStand;
        case FSMStateName::RL:
            return state_list_.rl;
        default:
            return state_list_.invalid;
        }
    }
}

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(depth_rl_quadruped_controller::LeggedGymController, controller_interface::ControllerInterface);