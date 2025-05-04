#!/usr/bin/env python
\"\"\"CLI to generate error summaries.

Usage:
    python analyze_classification_errors.py <experiment_dir> [-o OUTPUT]
\"\"\"
import argparse
from pathlib import Path
from src.analysis.error_analysis_utils import run_enhanced_analysis

def main():
    p = argparse.ArgumentParser()
    p.add_argument('experiment_dir', type=Path)
    p.add_argument('-o', '--output', type=Path, default=None,
                   help='Directory to save CSVs (default = experiment dir)')
    args = p.parse_args()
    outdir = args.output or args.experiment_dir
    outdir.mkdir(parents=True, exist_ok=True)

    res = run_enhanced_analysis(args.experiment_dir)
    res['error_patterns'].to_csv(outdir / 'error_patterns.csv', index=False)
    res['hard_cases'].to_csv(outdir / 'hard_cases.csv', index=False)
    print(f'✓ Wrote CSVs to {outdir}')

if __name__ == '__main__':
    main()
