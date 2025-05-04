"""Unified error-analysis utilities.

Every helper here is **pure** (no I/O) so the module can be reused from
CLI scripts, notebooks and the API alike.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from src.utils.error_analysis_utils import load_all_prediction_files, analyze_text_features, visualize_error_distribution, generate_detailed_error_report

def load_all_prediction_files(experiment_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Load every CSV inside <experiment_dir>/predictions/ into a dict."""
    exp = Path(experiment_dir)
    pred_dir = exp / 'predictions'
    if not pred_dir.exists():
        raise FileNotFoundError(f'{pred_dir} does not exist')
    dfs: Dict[str, pd.DataFrame] = {}
    for csv in pred_dir.glob('*.csv'):
        model = csv.stem
        df = pd.read_csv(csv)
        df['model'] = model
        dfs[model] = df
    return dfs

def analyse_error_patterns(pred_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return dataframe with a row per distinct (true -> pred) error."""
    frames = []
    for name, df in pred_dfs.items():
        errs = df[df.true_label != df.pred_label].copy()
        errs['error_type'] = errs.true_label + ' -> ' + errs.pred_label
        frames.append(errs)
    if not frames:
        return pd.DataFrame(columns=['error_type', 'total_count'])
    merged = pd.concat(frames, ignore_index=True)
    return (merged.groupby('error_type', as_index=False)
                  .size()
                  .rename(columns={'size': 'total_count'})
                  .sort_values('total_count', ascending=False))

def consistently_misclassified(pred_dfs: Dict[str, pd.DataFrame], min_models: int = 2):
    """Docs misclassified by >= min_models models in exactly the same way."""
    combined = None
    for name, df in pred_dfs.items():
        wrong = df[df.true_label != df.pred_label][['id', 'text', 'true_label', 'pred_label']].copy()
        wrong[name] = True
        combined = wrong if combined is None else combined.merge(wrong, how='outer')
    combined = combined.fillna(False)
    mask = combined.drop(columns=['id', 'text', 'true_label', 'pred_label']).sum(1) >= min_models
    return combined[mask]

def analyze_text_features(predictions_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Analyze text features that might contribute to classification errors.
    
    Returns a dataframe with text statistics for correct and incorrect predictions.
    """
    all_rows = []
    
    for model_name, df in predictions_dict.items():
        # Add features
        df['is_correct'] = df['true_label'] == df['pred_label']
        df['text_length'] = df['text'].apply(lambda x: len(str(x)))
        df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
        
        # Group by correct/incorrect
        for is_correct in [True, False]:
            subset = df[df['is_correct'] == is_correct]
            if len(subset) > 0:
                all_rows.append({
                    'model': model_name,
                    'is_correct': is_correct,
                    'count': len(subset),
                    'avg_text_length': subset['text_length'].mean(),
                    'avg_word_count': subset['word_count'].mean(),
                })
    
    return pd.DataFrame(all_rows)

def visualize_error_distribution(predictions_dict: Dict[str, pd.DataFrame], output_dir: Path):
    """Create visualizations of error distributions across models and classes.
    
    Saves visualizations to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine all predictions for comparison
    model_results = []
    class_error_rates = []
    
    for model_name, df in predictions_dict.items():
        # Calculate overall accuracy
        accuracy = (df['true_label'] == df['pred_label']).mean()
        model_results.append({'model': model_name, 'accuracy': accuracy})
        
        # Calculate per-class error rates
        for class_name in df['true_label'].unique():
            class_df = df[df['true_label'] == class_name]
            error_rate = (class_df['true_label'] != class_df['pred_label']).mean()
            class_error_rates.append({
                'model': model_name,
                'class': class_name,
                'error_rate': error_rate,
                'count': len(class_df)
            })
    
    if not model_results:
        return
    
    # Plot overall model comparison
    model_df = pd.DataFrame(model_results)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='accuracy', data=model_df)
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_accuracy_comparison.png', dpi=300)
    plt.close()
    
    # Plot per-class error rates
    class_df = pd.DataFrame(class_error_rates)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='class', y='error_rate', hue='model', data=class_df)
    plt.title('Error Rate by Class and Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'error_rate_by_class.png', dpi=300)
    plt.close()

def generate_detailed_error_report(predictions_dict: Dict[str, pd.DataFrame], output_dir: Path):
    """Generate an HTML report with detailed analysis of classification errors.
    
    Includes example text snippets and patterns.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get consistently misclassified examples
    misclass_df = consistently_misclassified(predictions_dict)
    
    # Get common error patterns
    error_patterns_df = analyse_error_patterns(predictions_dict)
    
    # Create HTML report
    html = []
    html.append('<html><head><title>Detailed Error Analysis</title>')
    html.append('<style>body{font-family:Arial;max-width:1200px;margin:0 auto;padding:20px}')
    html.append('table{border-collapse:collapse;width:100%;margin-bottom:20px}')
    html.append('th,td{border:1px solid #ddd;padding:8px}')
    html.append('th{background-color:#f2f2f2;text-align:left}')
    html.append('tr:nth-child(even){background-color:#f9f9f9}')
    html.append('h1,h2,h3{color:#333}</style></head><body>')
    
    html.append('<h1>Detailed Classification Error Analysis</h1>')
    
    # Common error patterns
    html.append('<h2>Common Error Patterns</h2>')
    if not error_patterns_df.empty:
        html.append('<table><tr><th>Error Type</th><th>Count</th></tr>')
        for _, row in error_patterns_df.head(10).iterrows():
            html.append(f'<tr><td>{row["error_type"]}</td><td>{row["total_count"]}</td></tr>')
        html.append('</table>')
    else:
        html.append('<p>No error patterns found.</p>')
    
    # Consistently misclassified examples
    html.append('<h2>Consistently Misclassified Examples</h2>')
    if not misclass_df.empty:
        html.append('<table><tr><th>Text</th><th>True Label</th><th>Predicted Label</th><th>Models</th></tr>')
        for _, row in misclass_df.head(20).iterrows():
            models = [name for name in predictions_dict.keys() if row.get(name, False)]
            html.append(f'<tr><td>{row["text"]}</td><td>{row["true_label"]}</td>')
            html.append(f'<td>{row["pred_label"]}</td><td>{", ".join(models)}</td></tr>')
        html.append('</table>')
    else:
        html.append('<p>No consistently misclassified examples found.</p>')
    
    # Model-specific analyses
    html.append('<h2>Model-Specific Error Analysis</h2>')
    for model_name, df in predictions_dict.items():
        errors = df[df['true_label'] != df['pred_label']]
        html.append(f'<h3>{model_name}</h3>')
        
        # Error count by class
        error_by_class = errors.groupby('true_label').size().reset_index(name='count')
        html.append('<h4>Error Count by True Class</h4>')
        html.append('<table><tr><th>Class</th><th>Error Count</th></tr>')
        for _, row in error_by_class.sort_values('count', ascending=False).iterrows():
            html.append(f'<tr><td>{row["true_label"]}</td><td>{row["count"]}</td></tr>')
        html.append('</table>')
        
        # Sample errors
        html.append('<h4>Sample Errors</h4>')
        html.append('<table><tr><th>Text</th><th>True Label</th><th>Predicted Label</th></tr>')
        for _, row in errors.head(5).iterrows():
            html.append(f'<tr><td>{row["text"]}</td><td>{row["true_label"]}</td><td>{row["pred_label"]}</td></tr>')
        html.append('</table>')
    
    html.append('</body></html>')
    
    # Write the report
    with open(output_dir / 'detailed_error_report.html', 'w') as f:
        f.write('\n'.join(html))

def run_enhanced_analysis(experiment_dir: str | Path, output_dir: str | Path | None = None, verbose: bool = True):
    """Run enhanced qualitative analysis on all models in an experiment.
    
    Provides more detailed insights into classification errors.
    
    Parameters
    ----------
    experiment_dir : str or Path
        Folder that contains a `predictions/` sub-dir with CSVs.
    output_dir : str or Path, optional
        Where analysis outputs will be saved. If None, uses experiment_dir.
    verbose : bool, optional
        Log progress to stdout.
    """
    if verbose:
        print(f"Running enhanced qualitative analysis...")
    
    # Load all prediction files
    predictions_dict = load_all_prediction_files(experiment_dir)
    
    if verbose:
        print(f"Loaded {len(predictions_dict)} model prediction files")
    
    # Create output directory
    output_dir = Path(output_dir) if output_dir else Path(experiment_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Perform analyses
    if verbose:
        print("Analyzing error patterns...")
    error_patterns = analyse_error_patterns(predictions_dict)
    error_patterns.to_csv(output_dir / 'common_error_patterns.csv', index=False)
    
    if verbose:
        print("Identifying consistently misclassified examples...")
    misclass_examples = consistently_misclassified(predictions_dict, min_models=len(predictions_dict))
    if not misclass_examples.empty:
        misclass_examples.to_csv(output_dir / 'consistently_misclassified.csv', index=False)
    
    if verbose:
        print("Analyzing text features...")
    text_features = analyze_text_features(predictions_dict)
    text_features.to_csv(output_dir / 'text_feature_analysis.csv', index=False)
    
    if verbose:
        print("Creating visualizations...")
    visualize_error_distribution(predictions_dict, output_dir)
    
    if verbose:
        print("Generating detailed error report...")
    generate_detailed_error_report(predictions_dict, output_dir)
    
    if verbose:
        print(f"✓ Enhanced qualitative analysis complete. Results saved to {output_dir.resolve()}")
    
    return {
        'error_patterns': error_patterns,
        'hard_cases': misclass_examples,
        'text_features': text_features
    }