# Author: Vojtěch Eichler
# Simple script to unify metrics from multiple log files into a single CSV

import os
import re
import pandas as pd
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Unify metrics from multiple log files into a single CSV')
    parser.add_argument('--input_dir', type=str, default='baselines',
                      help='Directory containing the log files')
    parser.add_argument('--output_file', type=str, default='baselines/unified_metrics.csv',
                      help='Path to save the unified metrics CSV')
    parser.add_argument('--log_pattern', type=str, default='run.log',
                      help='Pattern to match log files')
    return parser.parse_args()

def parse_metric_line(line):
    """Parse a line containing a metric value."""
    # Match both timestamp formats:
    # 1. [DD/MM/YYYY HH:MM:SS] format
    # 2. YYYY-MM-DD HH:MM:SS format
    pattern = r'(?:\[\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\]|\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?(NDCG|MAP|Recall|P)@(\d+):\s*(\d+\.\d+)'
    match = re.search(pattern, line)
    if match:
        metric, k, value = match.groups()
        return metric, int(k), float(value)
    return None

def process_log_file(file_path):
    """Process a single log file and return a list of (metric, k, value) tuples."""
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            result = parse_metric_line(line)
            if result:
                results.append(result)
    return results

def main():
    args = parse_args()

    # Get all log files from input directory
    input_dir = Path(args.input_dir)
    log_files = list(input_dir.rglob(args.log_pattern))

    # Process each log file
    all_results = []
    for log_file in log_files:
        model_name = log_file.parent.name
        results = process_log_file(log_file)
        for metric, k, value in results:
            all_results.append({
                'model': model_name,
                'metric': metric,
                'k': k,
                'value': value
            })

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(all_results)
    df = df.sort_values(['model', 'metric', 'k'])

    # Save as CSV
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df.to_csv(args.output_file, index=False)
    print(f"Unified metrics saved to {args.output_file}")

    # Print summary
    print("\nSummary of metrics by model:")
    summary = df.groupby(['model', 'metric'])['value'].mean().unstack()
    print(summary)

if __name__ == '__main__':
    main()