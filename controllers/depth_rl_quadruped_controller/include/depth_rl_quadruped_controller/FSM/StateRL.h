//
// Created by biao on 24-10-6.
//

#ifndef STATERL_H
#define STATERL_H

#include <common/ObservationBuffer.h>
#include <depth_rl_quadruped_controller/control/CtrlComponent.h>
#include <torch/script.h>

#include "controller_common/FSM/FSMState.h"

struct CtrlComponent;

template <typename Functor>
void executeAndSleep(Functor f, const double frequency)
{
    using clock = std::chrono::high_resolution_clock;
    const auto start = clock::now();

    // Execute wrapped function
    f();

    // Compute desired duration rounded to clock decimation
    const std::chrono::duration<double> desiredDuration(1.0 / frequency);
    const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

    // Sleep
    const auto sleepTill = start + dt;
    std::this_thread::sleep_until(sleepTill);
}

inline void setThreadPriority(int priority, std::thread& thread)
{
    sched_param sched{};
    sched.sched_priority = priority;

    if (priority != 0)
    {
        if (pthread_setschedparam(thread.native_handle(), SCHED_FIFO, &sched) != 0)
        {
            std::cerr << "WARNING: Failed to set threads priority (one possible reason could be "
                "that the user and the group permissions are not set properly.)"
                << std::endl;
        }
    }
}


template <typename T>
struct RobotCommand
{
    struct MotorCommand
    {
        std::vector<T> q = std::vector<T>(32, 0.0);
        std::vector<T> dq = std::vector<T>(32, 0.0);
        std::vector<T> tau = std::vector<T>(32, 0.0);
        std::vector<T> kp = std::vector<T>(32, 0.0);
        std::vector<T> kd = std::vector<T>(32, 0.0);
    } motor_command;
};

template <typename T>
struct RobotState
{
    struct IMU
    {
        std::vector<T> quaternion = {1.0, 0.0, 0.0, 0.0}; // w, x, y, z
        std::vector<T> gyroscope = {0.0, 0.0, 0.0};
        std::vector<T> accelerometer = {0.0, 0.0, 0.0};
    } imu;

    struct MotorState
    {
        std::vector<T> q = std::vector<T>(32, 0.0);
        std::vector<T> dq = std::vector<T>(32, 0.0);
        std::vector<T> ddq = std::vector<T>(32, 0.0);
        std::vector<T> tauEst = std::vector<T>(32, 0.0);
        std::vector<T> cur = std::vector<T>(32, 0.0);
    } motor_state;
};

struct Control
{
    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
};

struct ModelParams
{
    std::string model_name;
    std::string framework;
    int decimation;
    int num_observations;
    bool use_camera;
    std::vector<std::string> observations;
    std::vector<int> observations_history;
    int num_feet;
    int mass_params_dim;                  // 质量参数维度（对应priv_latent中的质量参数）
    int friction_dim;                     // 摩擦系数维度（对应priv_latent中的摩擦参数）
    double damping;
    double stiffness;
    double action_scale;
    double hip_scale_reduction;
    std::vector<int> hip_scale_reduction_indices;
    int num_of_dofs;
    double lin_vel_scale;
    double ang_vel_scale;
    double dof_pos_scale;
    double dof_vel_scale;
    double delta_yaw_scale;
    double clip_obs;
    torch::Tensor clip_actions_upper;
    torch::Tensor clip_actions_lower;
    torch::Tensor torque_limits;
    torch::Tensor rl_kd;
    torch::Tensor rl_kp;
    torch::Tensor commands_scale;
    torch::Tensor default_dof_pos;
    // 相机参数
    bool use_camera;
    int depth_width;
    int depth_height;
    double max_depth;
    int depth_feature_dim;
    bool use_depth_cnn;
    std::string depth_cnn_model;
};

struct Observations
{
    // =========================================================================
    // 基础观测（对应 Python 中 obs_buf 的核心组成部分）
    // =========================================================================
    torch::Tensor lin_vel;         // 基底盘线速度（三维），形状 [N, 3]
    torch::Tensor ang_vel;         // 基底盘角速度（三维），形状 [N, 3]
    torch::Tensor gravity_vec;     // 重力向量（三维，机体坐标系），形状 [N, 3]
    torch::Tensor commands;        // 控制命令（如 x速度、y速度、yaw角速度），形状 [N, 3]
    // torch::Tensor base_quat;       // 基底盘姿态四元数（x,y,z,w 或 w,x,y,z），形状 [N, 4]
    torch::Tensor imu_obs;
    torch::Tensor dof_pos;         // 关节位置（每个关节的角度），形状 [N, K]（K为关节数量）
    torch::Tensor dof_vel;         // 关节速度（每个关节的角速度），形状 [N, K]
    torch::Tensor actions;         // 上一步动作（关节控制量），形状 [N, K]

    // =========================================================================
    // 扩展观测（extreme_parkout 项目新增，对应 Python 中补充的观测项）
    // =========================================================================
    torch::Tensor roll;            // 横滚角（IMU 姿态角），形状 [N]（标量，拼接时需 unsqueeze(1)）
    torch::Tensor pitch;           // 俯仰角（IMU 姿态角），形状 [N]（标量，拼接时需 unsqueeze(1)）
    torch::Tensor delta_yaw;       // 当前目标偏航角与实际偏航角的偏差，形状 [N]（标量）
    torch::Tensor delta_next_yaw;  // 下一个目标偏航角与实际偏航角的偏差，形状 [N]（标量）
    torch::Tensor contact_states;  // 足部接触状态（滤波后），形状 [N, F]（F为足数，如4足机器人为4）
    torch::Tensor env_class;       // 环境类别标识（如17代表特定地形），形状 [N]（标量）

    // =========================================================================
    // 私有观测（对应 Python 中的 priv_explicit 和 priv_latent）
    // =========================================================================
    torch::Tensor mass_params;     // 机器人质量参数，形状 [N, M]（M为质量参数维度）
    torch::Tensor friction_coeffs; // 地面摩擦系数，形状 [N, C]（C为摩擦参数维度）
    torch::Tensor motor_strength;  // 电机强度参数（两组），形状 [N, 2]
};

class StateRL final : public FSMState
{
public:
    explicit StateRL(CtrlInterfaces& ctrl_interfaces,
                     CtrlComponent& ctrl_component,
                     const std::vector<double>& target_pos);

    void enter() override;

    void run(const rclcpp::Time& time,
             const rclcpp::Duration& period) override;

    void exit() override;

    FSMStateName checkChange() override;

    torch::Tensor preprocessDepthImage();

private:
    torch::Tensor computeObservation();

    void loadYaml(const std::string& config_path);

    static torch::Tensor quatRotateInverse(const torch::Tensor& q, const torch::Tensor& v,
                                           const std::string& framework);

    /**
    * @brief Forward the RL model to get the action
    */
    torch::Tensor forward();

    void getState();

    void runModel();

    void setCommand() const;

    std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node_;
    std::string robot_pkg_ = "go2_description";
    std::string model_folder_ = "legged_gym";

    bool enable_estimator_;
    std::shared_ptr<Estimator>& estimator_;

    // Parameters
    ModelParams params_;
    Observations obs_;
    Control control_;
    double init_pos_[12] = {};

    RobotState<double> robot_state_;
    RobotCommand<double> robot_command_;

    // history buffer
    std::shared_ptr<ObservationBuffer> history_obs_buf_;
    torch::Tensor history_obs_;

    // rl module
    torch::jit::script::Module model_;
    bool use_rl_thread_ = true;
    std::thread rl_thread_;
    bool running_ = false;
    bool updated_ = false;

    // output buffer
    torch::Tensor output_torques;
    torch::Tensor output_dof_pos_;

    // depth
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_image_sub_;
    cv::Mat depth_image_;
    std::mutex depth_mutex_;  // 线程安全锁
};


#endif //STATERL_H
