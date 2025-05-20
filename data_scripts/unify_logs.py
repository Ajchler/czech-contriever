# Author: Vojtěch Eichler

import os
import re
import pandas as pd
from pathlib import Path

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
    # Get all run.log files from baselines directory
    baseline_dir = Path('baselines')
    log_files = list(baseline_dir.rglob('run.log'))

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
    output_file = 'baselines/unified_metrics.csv'
    df.to_csv(output_file, index=False)
    print(f"Unified metrics saved to {output_file}")

    # Print summary
    print("\nSummary of metrics by model:")
    summary = df.groupby(['model', 'metric'])['value'].mean().unstack()
    print(summary)

if __name__ == '__main__':
    main()