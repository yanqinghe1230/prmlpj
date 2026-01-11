import json
import numpy as np
from scipy.ndimage import gaussian_filter1d

def clean_trajectory(input_file, output_file, 
                     similarity_threshold=1e-3,
                     velocity_threshold=1e-5,
                     downsample_factor=2,
                     sigma=2,
                     smooth_joints=True):
    """
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
        similarity_threshold: 状态相似度阈值（欧氏距离）
        velocity_threshold: 速度阈值，低于此值认为静止
        downsample_factor: 降采样倍率（2表示保留一半的点）
        sigma: 高斯滤波标准差
        smooth_joints: 是否平滑关节数据（推荐True）
    """
    with open(input_file, 'r') as f:
        trajectory = json.load(f)
    
    print(f"原始轨迹: {len(trajectory)} 步")
    
    if downsample_factor > 1:
        trajectory = trajectory[::downsample_factor]
        print(f"降采样后: {len(trajectory)} 步 (因子={downsample_factor})")
    
    trajectory = remove_static_segments(trajectory, velocity_threshold)
    print(f"移除静止片段后: {len(trajectory)} 步")
    
    trajectory = remove_similar_states(trajectory, similarity_threshold)
    print(f"移除相似状态后: {len(trajectory)} 步")
    
    trajectory = smooth_trajectory(trajectory, sigma, smooth_joints)
    print(f"平滑后: {len(trajectory)} 步")
    
    with open(output_file, 'w') as f:
        json.dump(trajectory, f, indent=4)
    
    print(f"\n清洗完成！保存到 {output_file}")
    
    print_trajectory_stats(trajectory)
    
    return trajectory


def remove_static_segments(trajectory, velocity_threshold=1e-5):
    """移除几乎静止的时间步"""
    filtered = []
    
    for state in trajectory:
        vel = np.array(state['joint_velocities'])
        vel_norm = np.linalg.norm(vel)
        
        # 保留速度大于阈值的状态
        if vel_norm > velocity_threshold:
            filtered.append(state)
    
    # 如果过滤太激进导致轨迹太短，返回原始轨迹
    if len(filtered) < 50:
        print("  警告: 速度阈值过高，保留原始轨迹")
        return trajectory
    
    return filtered


def remove_similar_states(trajectory, threshold=1e-3):
    """
    移除相似的连续状态
    使用欧氏距离
    """
    if len(trajectory) == 0:
        return trajectory
    
    cleaned = [trajectory[0]]
    
    for state in trajectory[1:]:
        if not states_are_similar_euclidean(state, cleaned[-1], threshold):
            cleaned.append(state)
    
    return cleaned


def states_are_similar_euclidean(state1, state2, threshold=1e-3):
    """
    使用欧氏距离判断状态相似度
    """
    vec1 = np.concatenate([
        state1['joint_positions'],
        state1['joint_velocities'],
        state1['end_effector_position'],
        state1['end_effector_orientation']
    ])
    
    vec2 = np.concatenate([
        state2['joint_positions'],
        state2['joint_velocities'],
        state2['end_effector_position'],
        state2['end_effector_orientation']
    ])
    
    # 计算欧氏距离
    distance = np.linalg.norm(vec1 - vec2)
    
    return distance < threshold


def smooth_trajectory(trajectory, sigma=2, smooth_joints=True):
    """
    使用高斯滤波平滑轨迹
    现在会平滑所有关节数据
    """
    if len(trajectory) < 3:
        return trajectory
    
    # 提取所有数据
    joint_positions = np.array([s['joint_positions'] for s in trajectory])
    joint_velocities = np.array([s['joint_velocities'] for s in trajectory])
    ee_positions = np.array([s['end_effector_position'] for s in trajectory])
    ee_orientations = np.array([s['end_effector_orientation'] for s in trajectory])
    
    # 平滑关节数据
    if smooth_joints:
        joint_positions_smooth = gaussian_filter1d(joint_positions, sigma=sigma, axis=0)
        joint_velocities_smooth = gaussian_filter1d(joint_velocities, sigma=sigma, axis=0)
    else:
        joint_positions_smooth = joint_positions
        joint_velocities_smooth = joint_velocities
    
    # 平滑末端执行器数据
    ee_positions_smooth = gaussian_filter1d(ee_positions, sigma=sigma, axis=0)
    ee_orientations_smooth = gaussian_filter1d(ee_orientations, sigma=sigma, axis=0)
    
    # 四元数归一化（重要！）
    for i in range(len(ee_orientations_smooth)):
        quat = ee_orientations_smooth[i]
        ee_orientations_smooth[i] = quat / np.linalg.norm(quat)
    
    # 重构轨迹
    smoothed_trajectory = []
    for i in range(len(trajectory)):
        smoothed_state = {
            'joint_positions': joint_positions_smooth[i].tolist(),
            'joint_velocities': joint_velocities_smooth[i].tolist(),
            'end_effector_position': ee_positions_smooth[i].tolist(),
            'end_effector_orientation': ee_orientations_smooth[i].tolist()
        }
        smoothed_trajectory.append(smoothed_state)
    
    return smoothed_trajectory


def print_trajectory_stats(trajectory):
    """打印轨迹统计信息"""
    velocities = np.array([s['joint_velocities'] for s in trajectory])
    vel_norms = np.linalg.norm(velocities, axis=1)
    
    positions = np.array([s['joint_positions'] for s in trajectory])
    pos_diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    
    print("\n=== 轨迹统计 ===")
    print(f"时间步数: {len(trajectory)}")
    print(f"持续时间: {len(trajectory) / 60:.2f}秒 (假设60Hz)")
    print(f"平均速度范数: {vel_norms.mean():.6f}")
    print(f"最大速度范数: {vel_norms.max():.6f}")
    print(f"平均位置变化: {pos_diffs.mean():.6f}")
    print(f"最大位置变化: {pos_diffs.max():.6f}")
    print(f"接近静止的帧 (<1e-4): {np.sum(vel_norms < 1e-4)} ({np.sum(vel_norms < 1e-4)/len(trajectory)*100:.1f}%)")


if __name__ == "__main__":
    input_file = "./trajectory/trajectory_1.json"
    output_file = "cleaned_trajectory.json"
    
    trajectory = clean_trajectory(
        input_file, 
        output_file,
        similarity_threshold=1e-3,    
        velocity_threshold=5e-5,       
        downsample_factor=2,          
        sigma=2,                       
        smooth_joints=True    
    )