import json
import os
from pathlib import Path

def filter_trajectory_files(directory, max_steps=1000):
    """
    Deletes all JSON files in the directory that have more than max_steps time steps.

    Args:
        directory (str): The directory containing trajectory JSON files.
        max_steps (int): The maximum allowed number of time steps. Files exceeding this will be deleted.
    """
    trajectory_dir = Path(directory)
    
    if not trajectory_dir.exists():
        print(f"Directory {directory} does not exist.")
        return
    
    deleted_count = 0
    kept_count = 0
    
    # Iterate through all JSON files in the directory
    for json_file in trajectory_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                trajectory = json.load(f)
            
            # Check if the trajectory is a list
            if isinstance(trajectory, list):
                num_steps = len(trajectory)
                
                if num_steps > max_steps:
                    os.remove(json_file)
                    print(f"Deleted: {json_file.name} (时间步数: {num_steps})")
                    deleted_count += 1
                else:
                    print(f"Kept: {json_file.name} (时间步数: {num_steps})")
                    kept_count += 1
            else:
                print(f"Skipped: {json_file.name} (数据格式不是列表)")
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    print(f"\n总结: 删除了 {deleted_count} 个文件，保留了 {kept_count} 个文件。")

if __name__ == "__main__":
    directory = "./cleaned_trajectory"  # Replace with your trajectory directory path
    max_steps = 1000  # Maximum allowed time steps
    
    filter_trajectory_files(directory, max_steps)