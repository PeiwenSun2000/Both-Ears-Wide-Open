import os
import subprocess
import pandas as pd

def generate_csv(folder_path, csv_path):
    """
    Generate a CSV file containing all file paths in the folder, ensuring specific quote format.
    """
    print(folder_path)
    paths = []
    for f in os.listdir(folder_path):
        full_path = os.path.join(folder_path, f)
        if os.path.exists(full_path):
            # Ensure the path is enclosed in double quotes
            if full_path[-4:] == ".wav":
                paths.append(f'"{full_path[:-4]}"')
    # Write to CSV file, ensuring the header is also in double quotes
    with open(csv_path, 'w') as f:
        f.write('"path"\n')  # Write header
        f.write('\n'.join(paths))  # Write paths

def run_visualization(exp_name):
    """
    Run the visualization_video.sh script.
    """
    subprocess.run(['bash', 'scripts/visualization_video.sh', exp_name, exp_name])

def calculate_distances(folder1, folder2):
    """
    Calculate distances between folders using cal_avg_mean_itd.py and cal_avg_idt_fad.py.
    """
    subprocess.run(['python', 'cal_avg_mean_itd.py'])
    subprocess.run(['python', 'cal_avg_idt_fad.py'])

def main(folder1, folder2):
    # Generate CSV files
    csv1 = 'temp_1.csv'
    csv2 = 'temp_2.csv'
    generate_csv(folder1, csv1)
    generate_csv(folder2, csv2)

    # Run visualization for the second folder
    run_visualization('exp1')

    # Calculate distances between folders
    calculate_distances(folder1, folder2)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Automate the process of generating PKL files and calculating distances.")
    parser.add_argument('--folder1', type=str, required=True, help='Path to the first data folder')
    parser.add_argument('--folder2', type=str, required=True, help='Path to the second data folder')
    args = parser.parse_args()

    main(args.folder1, args.folder2) 