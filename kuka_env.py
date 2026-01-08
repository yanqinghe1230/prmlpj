import pybullet as p
import pybullet_data
import numpy as np
import time

class KukaButtonEnv:
    
    def __init__(self, render=True, dt=1./120., max_steps=500):
        """
        Args:
            render: 是否显示GUI
            dt: 仿真步长
            max_steps: 单个episode最大步数
        """
        self.render_mode = render
        self.dt = dt
        self.max_steps = max_steps
        self.step_count = 0
        
        # 按钮参数
        self.button_position = [0.5, 0.5, 0.05]
        self.button_size = [0.05, 0.05, 0.01]
        self.success_threshold = 0.02  # 2cm内算成功
        
        # 初始化环境
        self._init_pybullet()
        
    def _init_pybullet(self):
        """初始化PyBullet"""
        # 连接物理引擎
        if self.render_mode:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.dt)
        
        # 加载机械臂
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
        
        # 创建按钮
        button_collision_shape = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=self.button_size
        )
        button_visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=self.button_size,
            rgbaColor=[1, 0, 0, 1]
        )
        self.button_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=button_collision_shape,
            baseVisualShapeIndex=button_visual_shape,
            basePosition=self.button_position
        )
        
        # 获取关节信息
        self.num_joints = p.getNumJoints(self.robot_id)
        self.controllable_joints = []
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_type = joint_info[2]
            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                self.controllable_joints.append(i)
        
        # 获取末端执行器链接ID
        self.end_effector_link_id = self._get_end_effector_link_id()
        
        print(f"环境初始化完成：")
        print(f"  可控关节数: {len(self.controllable_joints)}")
        print(f"  末端链接ID: {self.end_effector_link_id}")
        print(f"  按钮位置: {self.button_position}")
    
    def _get_end_effector_link_id(self):
        """获取末端执行器链接ID"""
        end_effector_link_name = "lbr_iiwa_link_7"
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            link_name = joint_info[12].decode('utf-8')
            if link_name == end_effector_link_name:
                return i
        
        # 如果找不到，返回最后一个链接
        return self.num_joints - 1
    
    def reset(self):
        """重置环境到初始状态"""
        self.step_count = 0
        
        # 重置机械臂到初始位置（可以是随机的或固定的）
        initial_positions = [0, 0, 0, -np.pi/4, 0, np.pi/4, 0]
        
        for i, joint_id in enumerate(self.controllable_joints):
            p.resetJointState(self.robot_id, joint_id, initial_positions[i])
        
        # 稳定几步
        for _ in range(10):
            p.stepSimulation()
        
        return self.get_observation()
    
    def get_observation(self):
        """获取当前观测"""
        # 获取关节状态
        joint_states = p.getJointStates(self.robot_id, self.controllable_joints)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        # 获取末端执行器状态
        end_effector_state = p.getLinkState(
            self.robot_id,
            self.end_effector_link_id,
            computeLinkVelocity=1
        )
        end_effector_position = list(end_effector_state[0])
        end_effector_orientation = list(end_effector_state[1])
        
        return {
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'end_effector_position': end_effector_position,
            'end_effector_orientation': end_effector_orientation
        }
    
    def step(self, action):
        """
        执行一步动作
        
        Args:
            action: numpy array (7,) 关节速度命令
        
        Returns:
            observation: 观测
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        self.step_count += 1
        
        # 应用动作（速度控制）
        self._apply_action(action)
        
        # 仿真一步
        p.stepSimulation()
        
        # 获取新的观测
        observation = self.get_observation()
        
        # 检查成功条件
        success = self.check_button_pressed()
        distance = self.get_distance_to_button()
        collision = self.check_collision()
        timeout = self.step_count >= self.max_steps
        
        # 计算奖励（可选）
        reward = self._compute_reward(success, distance, collision)
        
        # 判断episode是否结束
        done = success or collision or timeout
        
        # 额外信息
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
        """
        应用关节速度控制
        
        Args:
            action: numpy array (7,) 关节速度
        """
        # 限制速度范围（安全考虑）
        max_velocity = 1.0  # rad/s
        action = np.clip(action, -max_velocity, max_velocity)
        
        # 使用速度控制
        for i, joint_id in enumerate(self.controllable_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_id,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=action[i],
                force=100  # 最大力矩
            )
    
    def check_button_pressed(self):
        """检查是否成功按下按钮"""
        # 方法1: 检测接触
        contact_points = p.getContactPoints(
            bodyA=self.robot_id,
            bodyB=self.button_id
        )
        
        for contact in contact_points:
            if contact[3] == self.end_effector_link_id:
                # 获取末端位置
                end_effector_state = p.getLinkState(
                    self.robot_id,
                    self.end_effector_link_id
                )
                end_effector_position = end_effector_state[0]
                
                # 确保末端在桌面以上
                if end_effector_position[2] >= 0:
                    return True
        
        # 方法2: 基于距离（备选）
        distance = self.get_distance_to_button()
        if distance < 0.015:  # 1.5cm内算成功
            return True
        
        return False
    
    def get_distance_to_button(self):
        """计算末端执行器到按钮的距离"""
        end_effector_state = p.getLinkState(
            self.robot_id,
            self.end_effector_link_id
        )
        end_effector_position = np.array(end_effector_state[0])
        button_position = np.array(self.button_position)
        
        distance = np.linalg.norm(end_effector_position - button_position)
        return distance
    
    def check_collision(self):
        """检查是否发生不良碰撞（比如自碰撞）"""
        # 检查自碰撞
        contact_points = p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id)
        if len(contact_points) > 0:
            return True
        
        # 检查末端是否穿过桌面
        end_effector_state = p.getLinkState(
            self.robot_id,
            self.end_effector_link_id
        )
        if end_effector_state[0][2] < 0: 
            return True
        
        return False
    
    def _compute_reward(self, success, distance, collision):
        """
        计算奖励（如果需要强化学习）
        对于模仿学习，这个不是必需的
        """
        if success:
            return 100.0
        elif collision:
            return -50.0
        else:
            # 基于距离的稀疏奖励
            return -distance * 10
    
    def render(self):
        """渲染（PyBullet GUI模式下自动渲染）"""
        if not self.render_mode:
            print("Warning: render() called but environment is in DIRECT mode")
    
    def close(self):
        """关闭环境"""
        p.disconnect(self.physics_client)


# ============================================================================
# 测试环境
# ============================================================================

def test_environment():
    """测试环境是否正常工作"""
    print("测试环境...")
    
    # 创建环境
    env = KukaButtonEnv(render=True)
    
    # 重置
    obs = env.reset()
    print(f"\n初始观测:")
    print(f"  关节位置: {obs['joint_positions']}")
    print(f"  末端位置: {obs['end_effector_position']}")
    print(f"  到按钮距离: {env.get_distance_to_button():.4f}m")
    
    # 运行随机动作
    print("\n运行100步随机动作...")
    for step in range(100):
        # 随机动作
        action = np.random.uniform(-0.1, 0.1, size=7)
        
        obs, reward, done, info = env.step(action)
        
        if step % 20 == 0:
            print(f"  Step {step}: 距离={info['distance_to_button']:.4f}m")
        
        if done:
            print(f"\nEpisode结束 (step {step}):")
            print(f"  成功: {info['success']}")
            print(f"  碰撞: {info['collision']}")
            print(f"  超时: {info['timeout']}")
            break
    
    env.close()
    print("\n环境测试完成！")


if __name__ == "__main__":
    test_environment()