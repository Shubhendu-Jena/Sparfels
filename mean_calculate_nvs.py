import os

def parse_metrics_file(file_path):
    psnr_values = []
    ssim_values = []
    lpips_values = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                parts = line.split(',')
                psnr = float(parts[2].split(':')[1].strip())
                ssim = float(parts[3].split(':')[1].strip())
                lpips = float(parts[4].split(':')[1].strip())

                psnr_values.append(psnr)
                ssim_values.append(ssim)
                lpips_values.append(lpips)

    mean_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
    mean_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0
    mean_lpips = sum(lpips_values) / len(lpips_values) if lpips_values else 0

    return mean_psnr, mean_ssim, mean_lpips

def process_folders(root_directory):
    results = []
    all_psnr = []
    all_ssim = []
    all_lpips = []

    for folder in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder, "test", "ours_1000", "metrics.txt")
        if os.path.exists(folder_path):
            mean_psnr, mean_ssim, mean_lpips = parse_metrics_file(folder_path)
            results.append((folder, mean_psnr, mean_ssim, mean_lpips))

            # Accumulate metrics for overall mean
            all_psnr.append(mean_psnr)
            all_ssim.append(mean_ssim)
            all_lpips.append(mean_lpips)

    # Calculate overall mean across all folders
    overall_mean_psnr = sum(all_psnr) / len(all_psnr) if all_psnr else 0
    overall_mean_ssim = sum(all_ssim) / len(all_ssim) if all_ssim else 0
    overall_mean_lpips = sum(all_lpips) / len(all_lpips) if all_lpips else 0

    return results, overall_mean_psnr, overall_mean_ssim, overall_mean_lpips

def write_results_to_file(output_path, results, overall_means):
    with open(output_path, 'w') as file:
        file.write("Folder-wise Mean Metrics:\n")
        for folder, psnr, ssim, lpips in results:
            file.write(f"{folder} - PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, LPIPS: {lpips:.4f}\n")
        
        file.write("\nOverall Mean Metrics:\n")
        file.write(f"PSNR: {overall_means[0]:.2f}, SSIM: {overall_means[1]:.4f}, LPIPS: {overall_means[2]:.4f}\n")

# Main execution
root_directory = "."
output_file_path = os.path.join(root_directory, "mean_metrics.txt")

# Process folders and compute metrics
results, overall_mean_psnr, overall_mean_ssim, overall_mean_lpips = process_folders(root_directory)

# Write results to file
write_results_to_file(output_file_path, results, (overall_mean_psnr, overall_mean_ssim, overall_mean_lpips))

print(f"Results written to {output_file_path}")

