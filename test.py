import pybullet as p
import pybullet_data
import time
import json

def init():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

    button_position = [0.5, 0.5, 0.05]  
    button_size = [0.05, 0.05, 0.01]  
    button_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=button_size)
    button_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=button_size, rgbaColor=[1, 0, 0, 1])
    button_id = p.createMultiBody(
        baseMass=0, 
        baseCollisionShapeIndex=button_collision_shape,
        baseVisualShapeIndex=button_visual_shape,
        basePosition=button_position
    )
    
    # 获取机械臂末端的链接ID
    end_effector_link_name = "lbr_iiwa_link_7"  # KUKA机械臂末端链接名称
    num_joints = p.getNumJoints(robot_id)
    end_effector_link_id = -1
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[12].decode('utf-8') == end_effector_link_name:
            end_effector_link_id = i
            break

    return robot_id, button_id, end_effector_link_id

# 成功判定函数
def check_button_pressed(robot_id, button_id, end_effector_link_id):
    contact_points = p.getContactPoints(bodyA=robot_id, bodyB=button_id)
    for contact in contact_points:
        if contact[3] == end_effector_link_id:  # 检查是否是末端链接
            # 获取末端执行器的位姿
            end_effector_state = p.getLinkState(robot_id, end_effector_link_id)
            end_effector_position = end_effector_state[0]  # 末端位置
            if end_effector_position[2] >= 0:  # 确保末端Z轴位置不低于0
                return True
    return False

# 修改仿真循环，限制机械臂末端的Z轴位置

def start_simulate(robot_id, button_id, end_effector_link_id):
    expert_trajectory = []

    while True:
        p.stepSimulation()

        # 获取当前机械臂的关节状态
        num_joints = p.getNumJoints(robot_id)
        joint_states = p.getJointStates(robot_id, range(num_joints))
        joint_positions = [state[0] for state in joint_states]  # 关节角度
        joint_velocities = [state[1] for state in joint_states]  # 关节速度

        # 获取末端执行器的位姿
        end_effector_state = p.getLinkState(robot_id, end_effector_link_id)
        end_effector_position = end_effector_state[0]  # 末端位置
        end_effector_orientation = end_effector_state[1]  # 末端方向（四元数）

        # 确保末端不会穿过桌子
        if end_effector_position[2] < 0:
            print("机械臂末端试图穿过桌面！限制动作。")
            # 将关节状态恢复到上一个安全状态
            for i in range(num_joints):
                p.resetJointState(robot_id, i, joint_positions[i])

        # 记录当前时间步的数据
        expert_trajectory.append({
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "end_effector_position": end_effector_position,
            "end_effector_orientation": end_effector_orientation
        })

        # 检测是否成功按下按钮
        if check_button_pressed(robot_id, button_id, end_effector_link_id):
            print("按钮被末端按下！任务成功！")
            break

        time.sleep(1. / 120.)

    # 保存轨迹数据到文件
    with open("expert_trajectory.json", "w") as f:
        json.dump(expert_trajectory, f, indent=4)
    print("专家轨迹已保存到 expert_trajectory.json")

if __name__ == "__main__":
    robot_id, button_id, end_effector_link_id = init()
    start_simulate(robot_id, button_id, end_effector_link_id)