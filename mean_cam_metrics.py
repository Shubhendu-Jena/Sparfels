import argparse
import re

def parse_metrics(file_path):
    """
    Parse the metrics file and return a dictionary of metrics.
    """
    metrics = {}
    overall_metrics = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Match folder-wise metrics
            match = re.match(r"(scan\d+) - RPE_trans: ([\d.]+), RPE_rot: ([\d.]+), ATE: ([\d.]+)", line)
            if match:
                scan, rpe_trans, rpe_rot, ate = match.groups()
                metrics[scan] = {
                    "RPE_trans": float(rpe_trans),
                    "RPE_rot": float(rpe_rot),
                    "ATE": float(ate)
                }
    return metrics

def calculate_mean_metrics(metrics1, metrics2):
    """
    Calculate the mean metrics for each scan and overall mean metrics.
    """
    mean_metrics = {}
    total_rpe_trans = 0
    total_rpe_rot = 0
    total_ate = 0
    count = 0

    # Compute folder-wise mean metrics
    for scan in metrics1.keys():
        if scan in metrics2:
            rpe_trans = (metrics1[scan]["RPE_trans"] + metrics2[scan]["RPE_trans"]) / 2
            rpe_rot = (metrics1[scan]["RPE_rot"] + metrics2[scan]["RPE_rot"]) / 2
            ate = (metrics1[scan]["ATE"] + metrics2[scan]["ATE"]) / 2

            mean_metrics[scan] = {
                "RPE_trans": rpe_trans,
                "RPE_rot": rpe_rot,
                "ATE": ate
            }

            # Update totals
            total_rpe_trans += rpe_trans
            total_rpe_rot += rpe_rot
            total_ate += ate
            count += 1

    # Compute overall mean metrics
    overall_mean_metrics = {
        "RPE_trans": total_rpe_trans / count if count > 0 else 0,
        "RPE_rot": total_rpe_rot / count if count > 0 else 0,
        "ATE": total_ate / count if count > 0 else 0
    }

    return mean_metrics, overall_mean_metrics


def calculate_overall_mean(overall1, overall2):
    """
    Calculate the overall mean metrics.
    """
    return {
        "RPE_trans": (overall1["RPE_trans"] + overall2["RPE_trans"]) / 2,
        "RPE_rot": (overall1["RPE_rot"] + overall2["RPE_rot"]) / 2,
        "ATE": (overall1["ATE"] + overall2["ATE"]) / 2
    }

def save_mean_metrics(output_path, mean_metrics, overall_mean):
    """
    Save the calculated mean metrics to a file.
    """
    with open(output_path, 'w') as file:
        file.write("Folder-wise Mean Metrics:\n")
        for scan, metrics in mean_metrics.items():
            file.write(f"{scan} - RPE_trans: {metrics['RPE_trans']:.4f}, RPE_rot: {metrics['RPE_rot']:.4f}, ATE: {metrics['ATE']:.4f}\n")
        file.write("\nOverall Mean Metrics:\n")
        file.write(f"RPE_trans: {overall_mean['RPE_trans']:.4f}, RPE_rot: {overall_mean['RPE_rot']:.4f}, ATE: {overall_mean['ATE']:.4f}\n")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Calculate the mean of metrics from two input files.")
    parser.add_argument(
        "--files", 
        nargs=2, 
        required=True, 
        metavar=("FILE1", "FILE2"), 
        help="Paths to the two input files."
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Path to the output file where mean metrics will be saved."
    )
    
    args = parser.parse_args()
    
    # Parse the input files
    metrics1 = parse_metrics(args.files[0])
    metrics2 = parse_metrics(args.files[1])

    # Calculate folder-wise mean metrics
    mean_metrics, overall_mean_metrics = calculate_mean_metrics(metrics1, metrics2)

    # Calculate overall mean metrics
    #overall_mean = calculate_overall_mean(overall1, overall2)

    # Save results to the output file
    save_mean_metrics(args.output, mean_metrics, overall_mean_metrics)
    print(f"Mean metrics saved to {args.output}")

if __name__ == "__main__":
    main()

