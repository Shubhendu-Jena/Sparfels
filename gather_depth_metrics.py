import pandas as pd 
import argparse
import glob 
from joblib import Parallel, delayed
import numpy as np
import re
import os

def get_scan_number(path):
    return  re.search(r'scan(\d+)', path).group(1)
def load_json  (path):
    #print(path)
    #print(pd.read_json(path) )
    df = pd.read_json(path)[['average_relative_error','average_thresh_inlier','average_normal_consistency']].mean(0)
    #df.index = [scan_number]
    return  df
def extract_scan_number(scan_path):
    match = re.search(r"scan(\d+)", scan_path)
    return int(match.group(1)) if match else float('inf')

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--ckpt", type=int, default=1000)
    parser.add_argument(
        "--alignment", type=str, default="lsq_scale_shift"
    )
    args = parser.parse_args()
    path= f"{args.model_path}/scan*/train/ours_{args.ckpt}/depth_metrics_{args.alignment}.json"

    scenes = glob.glob(path)
    scenes = sorted(scenes, key=extract_scan_number)
    #print(scenes)
    #load_json(f"{args.model_path}/scan24/train/ours_{args.ckpt}/depth_metrics.json" )
    results_meshes = Parallel(n_jobs=10)(delayed(load_json)(scenepath ) for scenepath in scenes)
    dataset_results = pd.DataFrame(results_meshes)
    dataset_results.index = Parallel(n_jobs=10)(delayed(get_scan_number)(scenepath ) for scenepath in scenes)
    #print(results_meshes[0].mean(0))
    #print(dataset_results)
    robust_mean = np.mean(dataset_results.drop(index = '37')['average_relative_error'].values ) 
    dataset_results['average_relative_error'] = dataset_results['average_relative_error'].map('{:,.3f}'.format)
    dataset_results['average_thresh_inlier'] = dataset_results['average_thresh_inlier'].map('{:,.3f}'.format)
    dataset_results['average_normal_consistency'] = dataset_results['average_normal_consistency'].map('{:,.3f}'.format)
    pd.options.display.float_format =  '{:,.3f}'.format

    print(dataset_results.to_csv().strip('\n'))

    # Convert columns to numeric values (if they are strings)
    dataset_results['average_relative_error'] = pd.to_numeric(dataset_results['average_relative_error'], errors='coerce')  # Use 'coerce' to handle invalid parsing gracefully
    dataset_results['average_thresh_inlier'] = pd.to_numeric(dataset_results['average_thresh_inlier'], errors='coerce')
    dataset_results['average_normal_consistency'] = pd.to_numeric(dataset_results['average_normal_consistency'], errors='coerce')
    mean_values = dataset_results.mean()
    # Append the mean values as a new row
    dataset_results.loc['Mean'] = mean_values

    dataset_results.to_csv(os.path.join(args.model_path, 'depth_normal_median.txt'), sep='\t', index=True, index_label='Scan')
    #results_meshes = [load_json(scenepath ) for scenepath in scenes ]
    print(np.mean(results_meshes, axis = 0))
    print('robust_mean', robust_mean)
    import json
    # jsonfile = json.load(open (f"{args.model_path}/scan83/train/ours_{args.ckpt}/depth_metrics.json", 'r') )

