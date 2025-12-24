import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_trajectory(file_path):
    """
    Visualizes the trajectory in 3D space.

    Args:
        file_path (str): Path to the JSON file containing the trajectory.
    """
    # Load the trajectory data
    with open(file_path, 'r') as f:
        trajectory = json.load(f)

    # Extract end-effector positions
    positions = [state['end_effector_position'] for state in trajectory]
    x = [pos[0] for pos in positions]
    y = [pos[1] for pos in positions]
    z = [pos[2] for pos in positions]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='End-Effector Trajectory', color='b')
    ax.scatter(x, y, z, c='r', s=10, label='Trajectory Points')

    # Label the axes
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Trajectory Visualization')
    ax.legend()

    # Save the plot as an image
    plt.savefig('trajectory_visualization_smooth.png')

    # Show the plot
    #plt.show()

if __name__ == "__main__":
    file_path = "cleaned_trajectory.json"  # Replace with your trajectory file path
    visualize_trajectory(file_path)