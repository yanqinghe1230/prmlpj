import argparse
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import pybullet as p

# 复用 evaluate.py 中的策略与评估器
from evaluate import BCPolicyWrapper, PolicyEvaluator, KukaButtonEnv


class ConfigurableKukaButtonEnv(KukaButtonEnv):
    """
    在 evaluate.KukaButtonEnv 基础上增加：
    - 可配置初始关节角 `initial_positions`
    - 可动态重建按钮的碰撞几何（位置与尺寸）
    """

    def __init__(
        self,
        render: bool = False,
        dt: float = 1.0 / 240.0,
        max_steps: int = 500,
        initial_positions: Optional[List[float]] = None,
        button_position: Optional[Tuple[float, float, float]] = None,
        button_size: Optional[Tuple[float, float, float]] = None,
    ):
        super().__init__(render=render, dt=dt, max_steps=max_steps)

        # 自定义初始关节角（用于 reset）
        self.custom_initial_positions = initial_positions

        # 按需更新按钮几何
        if button_position is not None or button_size is not None:
            pos = list(button_position) if button_position is not None else self.button_position
            size = list(button_size) if button_size is not None else self.button_size
            self._rebuild_button(pos, size)

    def _rebuild_button(self, position: List[float], size: List[float]):
        # 删除旧按钮并创建新按钮
        try:
            p.removeBody(self.button_id)
        except Exception:
            pass
        button_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        button_visual_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=size, rgbaColor=[1, 0, 0, 1]
        )
        self.button_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=button_collision_shape,
            baseVisualShapeIndex=button_visual_shape,
            basePosition=position,
        )
        self.button_position = position
        self.button_size = size

    def reset(self):
        self.step_count = 0
        default_positions = [0, 0, 0, -np.pi / 4, 0, np.pi / 4, 0]
        initial_positions = (
            self.custom_initial_positions
            if self.custom_initial_positions is not None
            else default_positions
        )
        for i, joint_id in enumerate(self.controllable_joints):
            p.resetJointState(self.robot_id, joint_id, float(initial_positions[i]))
        for _ in range(10):
            p.stepSimulation()
        return self.get_observation()


def _fmt_vec3(v: List[float], prec: int = 3) -> str:
    if v is None: return "Default"
    return f"({float(v[0]):.{prec}f}, {float(v[1]):.{prec}f}, {float(v[2]):.{prec}f})"

def _fmt_list(xs: List[float], prec: int = 3, max_len: int = 7) -> str:
    if xs is None: return "Default"
    vals = ", ".join([f"{float(x):.{prec}f}" for x in xs[:max_len]])
    if len(xs) > max_len:
        vals += ", ..."
    return f"[{vals}]"


def run_single_experiment(
    exp_name: str,
    configs: List[Dict[str, Any]],
    policy: BCPolicyWrapper,
    episodes: int,
    dt: float,
    max_steps: int,
):
    print(f"\n{'='*120}")
    print(f"评估组: {exp_name} (共 {len(configs)} 个配置)")
    print(f"参数: {episodes} episodes/config, max_steps={max_steps}")
    print(f"{'='*120}")
    
    # 表头
    # ID | Success | Steps | Dist(mm) | Failures | Config Detail
    header = f"{'ID':<4} | {'Success':<8} | {'Steps':<6} | {'Dist(mm)':<8} | {'Failure Modes':<25} | {'Config Detail'}"
    print(header)
    print("-" * len(header))

    for idx, cfg in enumerate(configs):
        env = ConfigurableKukaButtonEnv(
            render=False,
            dt=dt,
            max_steps=max_steps,
            initial_positions=cfg.get("initial_positions"),
            button_position=cfg.get("button_position"),
            button_size=cfg.get("button_size"),
        )
        evaluator = PolicyEvaluator(policy, env)
        # 运行评估，不保存轨迹
        stats = evaluator.evaluate(
            num_episodes=episodes, render_first=0, save_failed=False
        )
        # 重要：关闭 bullet 连接，防止串扰
        env.close()

        # 格式化输出
        succ_rate = f"{stats['success_rate']*100:.1f}%"
        avg_steps = f"{stats['avg_steps']:.1f}"
        avg_dist = f"{stats['avg_final_distance']*1000:.1f}"
        
        # 简化失败模式显示 (例如: {'timeout': 5, 'miss': 1} -> "timeout:5, miss:1")
        fails = stats['failure_modes']
        fail_items = [f"{k}:{v}" for k,v in sorted(fails.items(), key=lambda x: -x[1])]
        fail_str = ", ".join(fail_items) if fails else "-"
        # 截断过长的失败信息
        if len(fail_str) > 25: fail_str = fail_str[:22] + "..."
        
        # 构建配置描述字符串，只显示非默认项
        details = []
        if cfg.get("note"):
            details.append(f"[{cfg['note']}]")
        if cfg.get("initial_positions") is not None:
            details.append(f"Joints={_fmt_list(cfg['initial_positions'])}")
        if cfg.get("button_position") is not None:
            details.append(f"Pos={_fmt_vec3(cfg['button_position'])}")
        if cfg.get("button_size") is not None:
             details.append(f"Size={_fmt_vec3(cfg['button_size'])}")
        config_str = " ".join(details)
        if not config_str: config_str = "Default Config"

        print(f"{idx+1:<4} | {succ_rate:<8} | {avg_steps:<6} | {avg_dist:<8} | {fail_str:<25} | {config_str}")


def build_experiments(noise_deg: float = 5.0) -> Dict[str, List[Dict[str, Any]]]:
    """
    构建三组独立实验：
    1. 初始姿态变动
    2. 按钮位置变动
    3. 按钮尺寸变动
    """
    default_init = np.array([0, 0, 0, -np.pi / 4, 0, np.pi / 4, 0], dtype=np.float32)
    default_pos = np.array([0.5, 0.5, 0.05], dtype=np.float32)
    default_size = np.array([0.05, 0.05, 0.01], dtype=np.float32)
    
    rng = np.random.default_rng(42)
    experiments = {}

    # === Exp 1: Initial Joint Variations ===
    exp1 = []
    # 1. 基准
    exp1.append({"initial_positions": default_init.tolist(), "note": "Baseline"})
    # 2. 随机噪声
    for i in range(2):
        noise_rad = np.deg2rad(noise_deg)
        noise = rng.uniform(-noise_rad, noise_rad, size=7)
        exp1.append({
            "initial_positions": (default_init + noise).tolist(),
            "note": f"Noise ±{noise_deg:.0f}deg ({i+1})"
        })
    # 3. 更大噪声
    for i in range(2): # 生成 2 个较大随机
        noise_rad = np.deg2rad(noise_deg * 2.0)
        noise = rng.uniform(-noise_rad, noise_rad, size=7)
        exp1.append({
            "initial_positions": (default_init + noise).tolist(),
            "note": f"Large Noise ±{noise_deg*2:.0f}deg ({i+1})"
        })
    experiments["Exp 1: 初始关节鲁棒性测试 (Initial Joint Robustness)"] = exp1

    # === Exp 2: Button Position Variations ===
    exp2 = []
    exp2.append({"button_position": default_pos.tolist(), "note": "Baseline"})
    offsets = [
        ([0.05, 0.0, 0.0], "Shift X+5cm"),
        ([-0.05, 0.0, 0.0], "Shift X-5cm"),
        ([0.0, 0.05, 0.0], "Shift Y+5cm"),
        ([0.0, -0.05, 0.0], "Shift Y-5cm"),
        ([0.03, 0.03, 0.0], "Shift XY+3cm"),
    ]
    for off, note in offsets:
        exp2.append({
            "button_position": (default_pos + np.array(off)).tolist(),
            "note": note
        })
    experiments["Exp 2: 按钮位置泛化测试 (Button Position Generalization)"] = exp2

    # === Exp 3: Button Size Variations ===
    exp3 = []
    exp3.append({"button_size": default_size.tolist(), "note": "Baseline"})
    sizes = [
        ([0.03, 0.03, 0.01], "Smaller (3cm)"),  # 原 5cm
        ([0.07, 0.07, 0.01], "Larger (7cm)"),
        ([0.05, 0.05, 0.005], "Thinner (0.5cm)"), # 原 1cm
        ([0.05, 0.05, 0.02], "Thicker (2cm)"),
    ]
    for sz, note in sizes:
        exp3.append({
            "button_size": sz,
            "note": note
        })
    experiments["Exp 3: 按钮尺寸泛化测试 (Button Size Generalization)"] = exp3

    return experiments


def main():
    parser = argparse.ArgumentParser(description="Kuka 按钮任务泛化评估 (Print-only Mode)")
    parser.add_argument("--model", default="best_model.pth", help="模型路径")
    parser.add_argument("--episodes", type=int, default=15, help="每个配置测试的 episode 数") 
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="仿真步长") 
    parser.add_argument("--max-steps", type=int, default=800, help="单次最大步数")
    parser.add_argument("--noise-deg", type=float, default=8.0, help="关节噪声基准幅度(度)")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"正在加载模型: {args.model} (Device: {device}) ...")
    try:
        policy = BCPolicyWrapper(args.model, device=device)
    except Exception as e:
        print(f"加载策略失败: {e}")
        return

    # 构建并运行实验
    all_experiments = build_experiments(noise_deg=args.noise_deg)
    
    # Kuka环境初始化时会有一些输出，先打印个分隔符
    print("\n" + "*"*80)
    print("开始泛化测试")
    print("*"*80)

    for exp_name, configs in all_experiments.items():
        run_single_experiment(
            exp_name=exp_name,
            configs=configs,
            policy=policy,
            episodes=args.episodes,
            dt=args.dt,
            max_steps=args.max_steps
        )

    print("\n" + "*"*80)
    print("所有测试完成 (All Tests Completed)")
    print("*"*80 + "\n")


if __name__ == "__main__":
    main()
