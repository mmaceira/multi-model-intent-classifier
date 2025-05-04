#!/usr/bin/env python3
"""
Analyze classification errors in the Reuters dataset.

This script runs enhanced qualitative analysis on the predictions from 
multiple models to identify error patterns, consistently misclassified examples,
and generate visualizations for better understanding of model performance.

Usage:
    python analyze_classification_errors.py [experiment_dir] [output_dir]

Example:
    python analyze_classification_errors.py experiment_with_07_classes error_analysis_results
"""

import os
import sys
from pathlib import Path
from src.enhanced_qualitative_analysis import run_enhanced_qualitative_analysis

def main():
    # Get command line arguments or use defaults
    if len(sys.argv) >= 3:
        experiment_dir = sys.argv[1]
        output_dir = sys.argv[2]
    else:
        # Default paths
        experiment_dir = "experiment_with_07_classes"
        output_dir = "error_analysis_results"
        print(f"Using default paths: experiment_dir={experiment_dir}, output_dir={output_dir}")
    
    # Make sure experiment directory exists
    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory '{experiment_dir}' not found")
        sys.exit(1)
    
    # Make sure predictions directory exists
    predictions_dir = os.path.join(experiment_dir, "predictions")
    if not os.path.exists(predictions_dir):
        print(f"Error: Predictions directory '{predictions_dir}' not found")
        sys.exit(1)
    
    # Run the analysis
    print(f"Analyzing classification errors from {experiment_dir}...")
    print(f"Results will be saved to {output_dir}")
    
    try:
        results = run_enhanced_qualitative_analysis(experiment_dir, output_dir)
        
        # Print a summary of the findings
        if not results['error_patterns'].empty:
            print("\nTop 5 common error patterns:")
            for _, row in results['error_patterns'].head(5).iterrows():
                print(f"  {row['error_type']}: {row['total_count']} occurrences")
        
        if not results['misclassified_examples'].empty:
            print(f"\nFound {len(results['misclassified_examples'])} examples misclassified by multiple models")
            print("Top 3 most consistently misclassified examples:")
            for _, row in results['misclassified_examples'].head(3).iterrows():
                print(f"  Example ID {row['id']} (true: {row['true_label']})")
                print(f"    Misclassified by {row['misclassified_count']} models")
                print(f"    Text: {row['text'][:100]}..." if len(row['text']) > 100 else f"    Text: {row['text']}")
        
        print(f"\nDetailed analysis complete! Check {output_dir} for full results.")
        print(f"  - HTML report: {os.path.join(output_dir, 'detailed_error_report.html')}")
        print(f"  - CSV files with error patterns and misclassified examples")
        print(f"  - Visualizations of error distributions")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 