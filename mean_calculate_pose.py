import os
import argparse

def parse_pose_file(file_path):
    """Parses a pose_eval.txt file to extract metrics."""
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                parts = line.split(',')
                rpe_trans = float(parts[0].split(':')[1].strip())
                rpe_rot = float(parts[1].split(':')[1].strip())
                ate = float(parts[2].split(':')[1].strip())
                return rpe_trans, rpe_rot, ate
    return 0, 0, 0  # Default values if file is empty or not in expected format

def process_folders(root_directory):
    """Processes each folder in the root directory and calculates mean metrics."""
    results = []
    all_rpe_trans = []
    all_rpe_rot = []
    all_ate = []

    for folder in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder, "pose_eval.txt")
        if os.path.exists(folder_path):
            rpe_trans, rpe_rot, ate = parse_pose_file(folder_path)
            results.append((folder, rpe_trans, rpe_rot, ate))

            all_rpe_trans.append(rpe_trans)
            all_rpe_rot.append(rpe_rot)
            all_ate.append(ate)

    overall_mean_rpe_trans = sum(all_rpe_trans) / len(all_rpe_trans) if all_rpe_trans else 0
    overall_mean_rpe_rot = sum(all_rpe_rot) / len(all_rpe_rot) if all_rpe_rot else 0
    overall_mean_ate = sum(all_ate) / len(all_ate) if all_ate else 0

    return results, overall_mean_rpe_trans, overall_mean_rpe_rot, overall_mean_ate

def write_results_to_file(output_path, results, overall_means):
    """Writes the folder-wise and overall mean metrics to a results file."""
    with open(output_path, 'w') as file:
        file.write("Folder-wise Mean Metrics:\n")
        for folder, rpe_trans, rpe_rot, ate in results:
            file.write(f"{folder} - RPE_trans: {rpe_trans:.4f}, RPE_rot: {rpe_rot:.4f}, ATE: {ate:.4f}\n")
        
        file.write("\nOverall Mean Metrics:\n")
        file.write(f"RPE_trans: {overall_means[0]:.4f}\n")
        file.write(f"RPE_rot: {overall_means[1]:.4f}\n")
        file.write(f"ATE: {overall_means[2]:.4f}\n")

def main():
    parser = argparse.ArgumentParser(description="Process pose_eval.txt files in subfolders.")
    parser.add_argument("root_directory", help="Path to the root directory containing subfolders with pose_eval.txt files")
    args = parser.parse_args()

    root_directory = args.root_directory
    output_file_path = os.path.join(root_directory, "pose_mean_metrics.txt")

    results, overall_mean_rpe_trans, overall_mean_rpe_rot, overall_mean_ate = process_folders(root_directory)
    write_results_to_file(output_file_path, results, (overall_mean_rpe_trans, overall_mean_rpe_rot, overall_mean_ate))

    print(f"Results written to {output_file_path}")

if __name__ == "__main__":
    main()

