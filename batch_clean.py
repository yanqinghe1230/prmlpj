import json
import os
from pathlib import Path
from new_clean import clean_trajectory
def batch_clean_trajectories(input_directory, output_directory):
    """
    Cleans all trajectory JSON files in the input directory by removing consecutive duplicate states
    and saves the cleaned trajectories to the output directory.

    Args:
        input_directory (str): Path to the directory containing input trajectory JSON files.
        output_directory (str): Path to the directory to save cleaned trajectory JSON files.
    """
    input_dir = Path(input_directory)
    output_dir = Path(output_directory)
    
    if not input_dir.exists():
        print(f"Input directory {input_directory} does not exist.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for json_file in input_dir.glob("*.json"):
        try:
            input_file_path = json_file
            output_file_path = output_dir / json_file.name
            
            clean_trajectory(input_file_path, output_file_path)
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")

if __name__ == "__main__":
    input_directory = "./trajectory"  # Replace with your input trajectory directory path
    output_directory = "./cleaned_trajectory"  # Replace with your desired output directory path
    
    batch_clean_trajectories(input_directory, output_directory)