import json

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

    # Save the cleaned trajectory to the output file
    with open(output_file, 'w') as f:
        json.dump(cleaned_trajectory, f, indent=4)

def states_are_similar(state1, state2, threshold=1e-5):
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

if __name__ == "__main__":
    input_file = "./trajectory/trajectory1.json"  # Replace with your input file path
    output_file = "cleaned_trajectory.json"  # Replace with your desired output file path
    
    tra_before = json.load(open(input_file, "r"))

    print(f"时间步数量: {len(tra_before)}")
    clean_trajectory(input_file, output_file)
    tra_after=json.load(open(output_file, "r"))
    print(f"时间步数量：{len(tra_after)}")
    print(f"Cleaned trajectory saved to {output_file}")