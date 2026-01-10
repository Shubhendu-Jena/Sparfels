import argparse
import pandas as pd

# Function to read and process the text file
def read_and_process_file(file_path):
    try:
        # Read the text file into a pandas DataFrame
        df = pd.read_csv(file_path, sep='\t', comment=None)  # Assuming the file uses tab as the separator
        
        # Convert the 'Scan' column to numeric, to ensure proper handling
        df['Scan'] = pd.to_numeric(df['Scan'], errors='coerce')
        
        # Remove rows where 'Scan' is NaN (this handles cases like 'Mean' rows)
        df = df.dropna(subset=['Scan'])
        
        # Convert relevant columns to numeric
        df['average_relative_error'] = pd.to_numeric(df['average_relative_error'], errors='coerce')
        df['average_thresh_inlier'] = pd.to_numeric(df['average_thresh_inlier'], errors='coerce')
        df['average_normal_consistency'] = pd.to_numeric(df['average_normal_consistency'], errors='coerce')
        
        # Return the cleaned DataFrame
        return df
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

# Function to compute the mean of each column per scan value
def compute_mean_per_scan(df):
    # Group by 'Scan' and calculate the mean for 'average_relative_error' and 'average_thresh_inlier'
    mean_per_scan = df.groupby('Scan')[['average_relative_error', 'average_thresh_inlier', 'average_normal_consistency']].mean()
    
    # Calculate the overall means
    overall_mean = {
        'average_relative_error': df['average_relative_error'].mean(),
        'average_thresh_inlier': df['average_thresh_inlier'].mean(),
        'average_normal_consistency': df['average_normal_consistency'].mean()
    }
    
    # Append the mean row (labeled 'Mean') to the dataframe
    mean_row = pd.DataFrame(overall_mean, index=['Mean'])
    
    # Combine the groupby result with the overall mean row
    result = pd.concat([mean_per_scan, mean_row])
    
    return result

# Main script logic to take file names as input from command line
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process two files and compute means for each scan.')
    
    # Add arguments for file paths
    parser.add_argument('file1', type=str, help='Path to the first input file')
    parser.add_argument('file2', type=str, help='Path to the second input file')
    parser.add_argument('output', type=str, help='Path to the output file')

    # Parse the arguments
    args = parser.parse_args()

    # Read and process both files
    df1 = read_and_process_file(args.file1)
    df2 = read_and_process_file(args.file2)

    # Check if the dataframes are valid
    if df1 is None or df2 is None:
        print("One or both input files failed to process. Exiting.")
        return

    # Combine both dataframes
    combined_df = pd.concat([df1, df2])

    # Compute the mean per scan for the combined DataFrame, including the "Mean" row
    mean_per_scan = compute_mean_per_scan(combined_df)

    # Output the result to the console
    print(mean_per_scan)

    # Save the result to the specified output file
    try:
        # Ensure the column names are correct, and save as tab-separated file
        mean_per_scan.to_csv(args.output, sep='\t', index_label='Scan')
        print(f"Results saved to {args.output}")
    except Exception as e:
        print(f"Error saving the output file: {e}")

# Run the script
if __name__ == "__main__":
    main()

