import yaml
import json
import os
import sys

# ANSI 颜色代码
class Colors:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== {msg} ==={Colors.ENDC}")

def print_ok(msg):
    print(f"{Colors.OKGREEN}[✔ 对齐] {msg}{Colors.ENDC}")

def print_warn(msg, detail=""):
    print(f"{Colors.WARNING}[WARN 警告] {msg}{Colors.ENDC}")
    if detail:
        print(f"    -> {detail}")

def print_fail(msg, detail=""):
    print(f"{Colors.FAIL}[✘ 不匹配] {msg}{Colors.ENDC}")
    if detail:
        print(f"    -> {detail}")

def check_alignment(cpp_config_path, python_config_path):
    print_header("极限界 (Extreme Parkour) 配置对齐检查程序")
    print(f"C++ 配置文件:    {cpp_config_path}")
    print(f"Python 配置文件: {python_config_path}")

    # 加载配置
    try:
        with open(cpp_config_path, 'r') as f:
            cpp_cfg = yaml.safe_load(f)
    except Exception as e:
        print_fail(f"无法加载 C++ 配置: {e}")
        return

    try:
        with open(python_config_path, 'r') as f:
            py_cfg = json.load(f)
    except Exception as e:
        print_fail(f"无法加载 Python 配置: {e}")
        return

    mismatches = 0
    warnings = 0

    # ================= 1. 关键缩放系数 (Scales) =================
    print_header("1. 观测缩放系数 (Observation Scales)")
    
    # 检查函数
    def check_scale(name, cpp_key, py_key_chain, tolerance=1e-5):
        nonlocal mismatches
        cpp_val = float(cpp_cfg.get(cpp_key, 0.0))
        
        # 获取 Python 值
        py_val = py_cfg
        for k in py_key_chain:
            py_val = py_val.get(k, {})
        py_val = float(py_val if not isinstance(py_val, dict) else 0.0)

        if abs(cpp_val - py_val) > tolerance:
            print_fail(f"{name}: C++={cpp_val} != Python={py_val}", 
                      f"影响: 导致神经网络输入数值范围错误，严重影响策略表现。")
            mismatches += 1
        else:
            print_ok(f"{name}: {cpp_val}")

    check_scale("线速度缩放 (lin_vel_scale)", 'lin_vel_scale', ['normalization', 'obs_scales', 'lin_vel'])
    check_scale("角速度缩放 (ang_vel_scale)", 'ang_vel_scale', ['normalization', 'obs_scales', 'ang_vel'])
    check_scale("关节位置缩放 (dof_pos_scale)", 'dof_pos_scale', ['normalization', 'obs_scales', 'dof_pos'])
    check_scale("关节速度缩放 (dof_vel_scale)", 'dof_vel_scale', ['normalization', 'obs_scales', 'dof_vel'])

    # ================= 2. 控制参数 (Control) =================
    print_header("2. 控制器参数 (Control Parameters)")

    # KP
    cpp_kp = cpp_cfg.get('rl_kp', [])
    cpp_kp_val = float(cpp_kp[0]) if isinstance(cpp_kp, list) and cpp_kp else 0.0
    py_kp = float(py_cfg.get('control', {}).get('stiffness', {}).get('joint', 0.0))
    
    if abs(cpp_kp_val - py_kp) > 0.1:
        print_fail(f"刚度系数 (KP): C++={cpp_kp_val} != Python={py_kp}")
        mismatches += 1
    else:
        print_ok(f"刚度系数 (KP): {cpp_kp_val}")

    # KD
    cpp_kd = cpp_cfg.get('rl_kd', [])
    cpp_kd_val = float(cpp_kd[0]) if isinstance(cpp_kd, list) and cpp_kd else 0.0
    py_kd = float(py_cfg.get('control', {}).get('damping', {}).get('joint', 0.0))
    
    if abs(cpp_kd_val - py_kd) > 0.01:
        print_fail(f"阻尼系数 (KD): C++={cpp_kd_val} != Python={py_kd}")
        mismatches += 1
    else:
        print_ok(f"阻尼系数 (KD): {cpp_kd_val}")

    # Action Scale
    check_scale("动作缩放 (action_scale)", 'action_scale', ['control', 'action_scale'])
    
    # Clip Actions
    check_scale("动作截断 (clip_actions)", 'clip_actions', ['normalization', 'clip_actions'])

    # ================= 3. 运行逻辑与阈值 =================
    print_header("3. 运行逻辑与阈值 (Operation Logic)")

    # 足部接触力阈值
    cpp_force = float(cpp_cfg.get('foot_force_threshold', 0.0))
    EXPECTED_FORCE_THR = 25.0 
    
    if abs(cpp_force - EXPECTED_FORCE_THR) > 5.0:
        print_fail(f"足底接触力阈值 (foot_force_threshold): C++={cpp_force} (推荐: {EXPECTED_FORCE_THR})",
                   "严重警告: 阈值过高会导致机器人误判为‘脚已离地’，从而提前收腿或动作异常！这是‘提前抬脚’的常见原因。")
        mismatches += 1
    else:
        print_ok(f"足底接触力阈值: {cpp_force} (匹配标准值)")

    # 指令速度 (Forward Command Speed)
    cpp_cmd = float(cpp_cfg.get('forward_command_speed', 0.0))
    print(f"    [信息] C++ 设置的前进指令速度: {cpp_cmd} m/s")
    if cpp_cmd < 0.4:
        print_warn(f"前进指令速度较小 ({cpp_cmd} m/s)。", "如果训练时是用 0.5 m/s 或更高，这可能导致策略困惑。")
        warnings += 1

    # ================= 4. 频率与步长 (Frequency & Decimation) =================
    print_header("4. 频率与步长 (Frequency check)")
    
    cpp_decimation = int(cpp_cfg.get('decimation', 4))
    py_dt = float(py_cfg.get('sim', {}).get('dt', 0.005))
    py_decimation = int(py_cfg.get('control', {}).get('decimation', 4))
    
    policy_dt = py_dt * py_decimation 
    print(f"    [信息] 策略训练 DT: {policy_dt:.3f}s (频率: {1/policy_dt:.1f}Hz)")
    
    print_warn("请确认底层控制频率 (Frequency):")
    print(f"    如果 ros2_control 运行在 500Hz: 500/{cpp_decimation} = {500/cpp_decimation}Hz。")
    print(f"    如果 ros2_control 运行在 200Hz: 200/{cpp_decimation} = {200/cpp_decimation}Hz。")
    print(f"    目标控制频率应接近: {1/policy_dt:.1f}Hz")
    print("    -> 频率不匹配会导致历史观测 (History Buffer) 的时间跨度错误，严重影响动态表现。")

    # ================= 总结 =================
    print_header("检查总结 (Summary)")
    if mismatches == 0:
        print(f"{Colors.OKGREEN}{Colors.BOLD}完美！未发现明显的配置不匹配。{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}发现 {mismatches} 个配置不匹配，请务必修复！{Colors.ENDC}")
    
    if warnings > 0:
        print(f"{Colors.WARNING}发现 {warnings} 个警告，建议人工复核。{Colors.ENDC}")

if __name__ == "__main__":
    check_alignment(
        '/home/jialonglong/Quadruped_ws/src/quadruped_ros2_control/descriptions/peak_lab/depth_go2_description/config/extreme_parkour_common/config.yaml',
        '/home/jialonglong/Quadruped_ws/external/extreme-parkour/external/original_parkour/Extreme-Parkour-Onboard/traced/config.json'
    )
