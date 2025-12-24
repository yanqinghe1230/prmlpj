import json
from scipy.ndimage import gaussian_filter1d

def clean_trajectory(input_file, output_file):
    """
    Cleans the trajectory data by removing consecutive duplicate states.

    Args:
        input_file (str): Path to the input JSON file containing the trajectory.
        output_file (str): Path to the output JSON file to save the cleaned trajectory.
    """
    # Load the trajectory data from the input file
    with open(input_file, 'r') as f:
        trajectory = json.load(f)

    # Ensure the trajectory is a list
    if not isinstance(trajectory, list):
        raise ValueError("The trajectory data must be a list of states.")

    # Clean the trajectory by removing consecutive similar states
    cleaned_trajectory = []
    for state in trajectory:
        if not cleaned_trajectory or not states_are_similar(state, cleaned_trajectory[-1]):
            cleaned_trajectory.append(state)

    # Smooth the cleaned trajectory
    smoothed_trajectory = smooth_trajectory(cleaned_trajectory, sigma=2)

    # Save the smoothed trajectory to the output file
    with open(output_file, 'w') as f:
        json.dump(smoothed_trajectory, f, indent=4)

    print(f"Smoothed trajectory saved to {output_file}")

def states_are_similar(state1, state2, threshold=1e-4):
    """
    Compares two states and determines if they are similar within a given threshold.

    Args:
        state1 (dict): The first state to compare.
        state2 (dict): The second state to compare.
        threshold (float): The maximum allowed difference to consider states as similar.

    Returns:
        bool: True if states are similar, False otherwise.
    """
    for key in state1:
        if key in state2:
            value1 = state1[key]
            value2 = state2[key]

            # Compare lists element-wise
            if isinstance(value1, list) and isinstance(value2, list):
                for v1, v2 in zip(value1, value2):
                    if abs(v1 - v2) > threshold:
                        return False
            # Compare scalar values
            elif isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                if abs(value1 - value2) > threshold:
                    return False
    return True

def smooth_trajectory(trajectory, sigma=1):
    """
    Smooths the trajectory using a Gaussian filter.

    Args:
        trajectory (list): The trajectory data to smooth.
        sigma (float): The standard deviation for Gaussian kernel.

    Returns:
        list: The smoothed trajectory.
    """
    smoothed_trajectory = []

    # Extract and smooth each component of the trajectory
    positions = [state['end_effector_position'] for state in trajectory]
    x = [pos[0] for pos in positions]
    y = [pos[1] for pos in positions]
    z = [pos[2] for pos in positions]

    x_smooth = gaussian_filter1d(x, sigma)
    y_smooth = gaussian_filter1d(y, sigma)
    z_smooth = gaussian_filter1d(z, sigma)

    for i, state in enumerate(trajectory):
        smoothed_state = state.copy()
        smoothed_state['end_effector_position'] = [x_smooth[i], y_smooth[i], z_smooth[i]]
        smoothed_trajectory.append(smoothed_state)

    return smoothed_trajectory

if __name__ == "__main__":
    input_file = "expert_trajectory.json"  # Replace with your input file path
    output_file = "cleaned_trajectory.json"  # Replace with your desired output file path
    
    tra_before = json.load(open(input_file, "r"))

    print(f"时间步数量: {len(tra_before)}")
    clean_trajectory(input_file, output_file)
    tra_after=json.load(open(output_file, "r"))
    print(f"时间步数量：{len(tra_after)}")