import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import pybullet as p
import pybullet_data
import time

from kuka_env import KukaButtonEnv

class KukaButtonEnv:
    """KUKA机械臂按按钮任务的仿真环境（简化版，完整版见上面的代码）"""
    
    def __init__(self, render=True, dt=1./60., max_steps=500):
        self.render_mode = render
        self.dt = dt
        self.max_steps = max_steps
        self.step_count = 0
        self.button_position = [0.5, 0.5, 0.05]
        self.button_size = [0.05, 0.05, 0.01]
        self._init_pybullet()
    
    def _init_pybullet(self):
        if self.render_mode:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.dt)
        
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
        
        button_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.button_size)
        button_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=self.button_size, rgbaColor=[1, 0, 0, 1])
        self.button_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=button_collision_shape,
            baseVisualShapeIndex=button_visual_shape,
            basePosition=self.button_position
        )
        
        self.num_joints = p.getNumJoints(self.robot_id)
        self.controllable_joints = []
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.controllable_joints.append(i)
        
        self.end_effector_link_id = self._get_end_effector_link_id()
    
    def _get_end_effector_link_id(self):
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[12].decode('utf-8') == "lbr_iiwa_link_7":
                return i
        return self.num_joints - 1
    
    def reset(self):
        self.step_count = 0
        initial_positions = [0, 0, 0, -np.pi/4, 0, np.pi/4, 0]
        for i, joint_id in enumerate(self.controllable_joints):
            p.resetJointState(self.robot_id, joint_id, initial_positions[i])
        for _ in range(10):
            p.stepSimulation()
        return self.get_observation()
    
    def get_observation(self):
        joint_states = p.getJointStates(self.robot_id, self.controllable_joints)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        end_effector_state = p.getLinkState(self.robot_id, self.end_effector_link_id, computeLinkVelocity=1)
        end_effector_position = list(end_effector_state[0])
        end_effector_orientation = list(end_effector_state[1])
        
        return {
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'end_effector_position': end_effector_position,
            'end_effector_orientation': end_effector_orientation
        }
    
    def step(self, action):
        self.step_count += 1
        self._apply_action(action)
        p.stepSimulation()
        
        observation = self.get_observation()
        success = self.check_button_pressed()
        distance = self.get_distance_to_button()
        collision = self.check_collision()
        timeout = self.step_count >= self.max_steps
        
        reward = 100.0 if success else -distance * 10
        done = success or collision or timeout
        
        info = {
            'success': success,
            'distance_to_button': distance,
            'collision': collision,
            'timeout': timeout
        }
        
        if self.render_mode:
            time.sleep(self.dt)
        
        return observation, reward, done, info
    
    def _apply_action(self, action):
        max_velocity = 1.0
        action = np.clip(action, -max_velocity, max_velocity)
        
        for i, joint_id in enumerate(self.controllable_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_id,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=action[i],
                force=100
            )
    
    def check_button_pressed(self):
        contact_points = p.getContactPoints(bodyA=self.robot_id, bodyB=self.button_id)
        for contact in contact_points:
            if contact[3] == self.end_effector_link_id:
                end_effector_state = p.getLinkState(self.robot_id, self.end_effector_link_id)
                if end_effector_state[0][2] >= 0:
                    return True
        return self.get_distance_to_button() < 0.015
    
    def get_distance_to_button(self):
        end_effector_state = p.getLinkState(self.robot_id, self.end_effector_link_id)
        end_effector_position = np.array(end_effector_state[0])
        button_position = np.array(self.button_position)
        return np.linalg.norm(end_effector_position - button_position)
    
    def check_collision(self):
        contact_points = p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id)
        if len(contact_points) > 0:
            return True
        end_effector_state = p.getLinkState(self.robot_id, self.end_effector_link_id)
        if end_effector_state[0][2] < 0:
            return True
        return False
    
    def close(self):
        p.disconnect(self.physics_client)


# ============================================================================
# 策略定义
# ============================================================================

class MLPPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)


class LSTMPolicy(nn.Module):
    """LSTM策略"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            state_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state_history):
        """
        Args:
            state_history: (batch, seq_len, state_dim)
        """
        lstm_out, _ = self.lstm(state_history)
        last_hidden = lstm_out[:, -1, :]  # 取最后一个时间步
        action = self.fc(last_hidden)
        return action


class BCPolicyWrapper:
    def __init__(self, model_path, device='cuda', history_len=10):
        self.device = device
        self.history_len = history_len
        self.history_buffer = []  # 用于 LSTM 的历史状态 buffer
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        self.state_mean = checkpoint['state_mean']
        self.state_std = checkpoint['state_std']
        self.action_mean = checkpoint['action_mean']
        self.action_std = checkpoint['action_std']
        
        state_dim = len(self.state_mean)
        action_dim = len(self.action_mean)
        
        # 自动检测模型类型
        state_dict = checkpoint['model_state_dict']
        if any('lstm' in k for k in state_dict.keys()):
            self.is_lstm = True
            print(f"检测到 LSTM 模型 (history_len={history_len})")
            self.model = LSTMPolicy(state_dim, action_dim, hidden_dim=128, num_layers=2).to(device)
        else:
            self.is_lstm = False
            print("检测到 MLP 模型")
            self.model = MLPPolicy(state_dim, action_dim, hidden_dims=[256, 256, 128]).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"策略已加载: State dim={state_dim}, Action dim={action_dim}")
    
    def reset(self):
        """重置策略状态 (清空历史 buffer)"""
        self.history_buffer = []

    def get_action(self, observation):
        state = np.concatenate([
            observation['joint_positions'],
            observation['joint_velocities'],
            observation['end_effector_position'],
            observation['end_effector_orientation']
        ]).astype(np.float32)
        
        # 归一化
        state_normalized = (state - self.state_mean) / self.state_std
        
        if self.is_lstm:
            # 更新历史 buffer
            self.history_buffer.append(state_normalized)
            if len(self.history_buffer) > self.history_len:
                self.history_buffer.pop(0)
            
            # 准备输入: 如果历史不足 history_len，用第一帧重复填充
            current_history = list(self.history_buffer)
            while len(current_history) < self.history_len:
                current_history.insert(0, current_history[0])
            
            input_tensor = np.array(current_history) # (seq_len, state_dim)
            input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(self.device) # (1, seq_len, state_dim)
            
            with torch.no_grad():
                action_normalized = self.model(input_tensor)
        else:
            state_tensor = torch.from_numpy(state_normalized).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_normalized = self.model(state_tensor)
        
        action = action_normalized.cpu().numpy().squeeze()
        action = action * self.action_std + self.action_mean
        
        return action


# ============================================================================
# 评估器
# ============================================================================

class PolicyEvaluator:
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env
        self.results = []
    
    def run_episode(self, render=False, save_trajectory=False):
        observation = self.env.reset()
        if hasattr(self.policy, 'reset'):
            self.policy.reset() # 重置策略内部状态 (如 LSTM buffer)

        trajectory = []
        
        for step in range(self.env.max_steps):
            action = self.policy.get_action(observation)
            observation, reward, done, info = self.env.step(action)
            
            if save_trajectory:
                trajectory.append({
                    'observation': observation,
                    'action': action.tolist(),
                    'distance': info['distance_to_button']
                })
            
            if done:
                failure_mode = self._classify_failure(info, trajectory)
                return {
                    'success': info['success'],
                    'steps': step + 1,
                    'failure_mode': failure_mode if not info['success'] else None,
                    'trajectory': trajectory if save_trajectory else None,
                    'final_distance': info['distance_to_button']
                }
        
        return {
            'success': False,
            'steps': self.env.max_steps,
            'failure_mode': 'timeout',
            'trajectory': trajectory if save_trajectory else None,
            'final_distance': info['distance_to_button']
        }
    
    def _classify_failure(self, info, trajectory):
        if info['success']:
            return None
        
        if info['collision']:
            return 'collision'
        
        if info['timeout']:
            return 'timeout'
        
        final_distance = info['distance_to_button']
        if final_distance > 0.1:
            return 'miss'
        elif final_distance > 0.03:
            return 'near_miss'
        
        if len(trajectory) > 10:
            recent_distances = [t['distance'] for t in trajectory[-10:]]
            if max(recent_distances) - min(recent_distances) > 0.05:
                return 'oscillation'
        
        return 'unknown'
    
    def evaluate(self, num_episodes=50, render_first=5, save_failed=False):
        print(f"\n{'='*70}")
        print(f"开始评估: {num_episodes} episodes")
        print(f"{'='*70}\n")
        
        self.results = []
        failure_modes = defaultdict(int)
        success_count = 0
        
        for episode in range(num_episodes):
            render = (episode < render_first)
            save_traj = (not render and save_failed)
            
            result = self.run_episode(render=render, save_trajectory=save_traj)
            self.results.append(result)
            
            if result['success']:
                success_count += 1
                status = "✓ 成功"
            else:
                failure_modes[result['failure_mode']] += 1
                status = f"✗ 失败 ({result['failure_mode']})"
                
                if save_failed and result['trajectory']:
                    self._save_failed_trajectory(episode, result)
            
            print(f"Episode {episode+1:3d}/{num_episodes}: {status:25s} "
                  f"| 步数: {result['steps']:3d} "
                  f"| 距离: {result['final_distance']:.4f}m")
        
        success_rate = success_count / num_episodes
        
        statistics = {
            'success_rate': success_rate,
            'success_count': success_count,
            'total_episodes': num_episodes,
            'avg_steps': np.mean([r['steps'] for r in self.results]),
            'avg_final_distance': np.mean([r['final_distance'] for r in self.results]),
            'failure_modes': dict(failure_modes)
        }
        
        self._print_summary(statistics)
        #self._plot_results(statistics)
        
        return statistics
    
    def _save_failed_trajectory(self, episode_id, result):
        save_dir = Path('failed_trajectories')
        save_dir.mkdir(exist_ok=True)
        
        filename = save_dir / f"failed_ep{episode_id}_{result['failure_mode']}.json"
        
        simplified_traj = [{
            'joint_positions': t['observation']['joint_positions'],
            'end_effector_position': t['observation']['end_effector_position'],
            'action': t['action'],
            'distance': t['distance']
        } for t in result['trajectory'][::5]]
        
        with open(filename, 'w') as f:
            json.dump(simplified_traj, f, indent=2)
    
    def _print_summary(self, stats):
        print(f"\n{'='*70}")
        print(f"评估摘要")
        print(f"{'='*70}")
        print(f"成功率: {stats['success_rate']*100:.1f}% ({stats['success_count']}/{stats['total_episodes']})")
        print(f"平均步数: {stats['avg_steps']:.1f}")
        print(f"平均最终距离: {stats['avg_final_distance']*1000:.1f}mm")
        
        if stats['failure_modes']:
            print(f"\n失败模式分布:")
            for mode, count in sorted(stats['failure_modes'].items(), key=lambda x: x[1], reverse=True):
                percentage = count / stats['total_episodes'] * 100
                print(f"  {mode:15s}: {count:3d} ({percentage:5.1f}%)")
        print(f"{'='*70}\n")
    
    def _plot_results(self, stats):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 成功率
        axes[0, 0].bar(['成功', '失败'], 
                      [stats['success_count'], stats['total_episodes'] - stats['success_count']],
                      color=['green', 'red'])
        axes[0, 0].set_ylabel('Episodes')
        axes[0, 0].set_title(f"成功率: {stats['success_rate']*100:.1f}%", fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 失败模式
        if stats['failure_modes']:
            modes = list(stats['failure_modes'].keys())
            counts = list(stats['failure_modes'].values())
            axes[0, 1].barh(modes, counts, color='coral')
            axes[0, 1].set_xlabel('Count')
            axes[0, 1].set_title('失败模式分布')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 步数分布
        steps = [r['steps'] for r in self.results]
        axes[1, 0].hist(steps, bins=20, edgecolor='black', color='skyblue')
        axes[1, 0].set_xlabel('步数')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title(f'步数分布 (均值: {stats["avg_steps"]:.1f})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 距离分布
        distances = [r['final_distance']*1000 for r in self.results]  # 转为mm
        axes[1, 1].hist(distances, bins=20, edgecolor='black', color='lightgreen')
        axes[1, 1].set_xlabel('最终距离 (mm)')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].set_title(f'最终距离分布')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evaluation_results.png', dpi=150)
        print("结果图已保存到 evaluation_results.png")


# ============================================================================
# 主程序
# ============================================================================

def main():
    # 配置
    MODEL_PATH = 'best_model.pth'
    NUM_EPISODES = 50
    RENDER_FIRST = 3  # 前3个episode显示GUI
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("BC策略评估")
    print("="*70)
    
    # 加载策略
    print("\n1. 加载策略...")
    policy = BCPolicyWrapper(MODEL_PATH, device=DEVICE)
    
    # 创建环境
    print("\n2. 初始化环境...")
    env = KukaButtonEnv(render=True, dt=1./60., max_steps=1000)
    
    # 创建评估器
    print("\n3. 创建评估器...")
    evaluator = PolicyEvaluator(policy, env)
    
    # 运行评估
    print("\n4. 开始评估...")
    statistics = evaluator.evaluate(
        num_episodes=NUM_EPISODES,
        render_first=RENDER_FIRST,
        save_failed=False
    )
    
    '''
    # 保存结果
    print("\n5. 保存结果...")
    with open('evaluation_statistics.json', 'w') as f:
        json.dump(statistics, f, indent=2)
    '''
    # 关闭环境
    env.close()
    
    print("\n✓ 评估完成！")
    print("结果文件:")
    #print("  - evaluation_results.png: 可视化")
    #print("  - evaluation_statistics.json: 统计数据")
    #print("  - failed_trajectories/: 失败案例")


if __name__ == '__main__':
    main()