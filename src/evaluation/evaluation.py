"""
Main evaluation module that orchestrates the evaluation workflow.

This module provides the main entry point for running evaluations on text classification models.
It coordinates the various evaluation components and generates comprehensive reports.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Any
from IPython.display import display

import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .metrics import compute_metrics, analyze_text_features
from .visualization import (
    plot_label_distribution,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_model_comparisons,
    plot_top_misclassifications,
    visualize_error_distribution,
    generate_detailed_error_report,
    plot_top_error_types,
)
from .utils import (
    ensure_dir,
    setup_logging,
    load_all_prediction_files,
    analyse_error_patterns,
    consistently_misclassified,
)

def analyze_top_errors(
    predictions_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 20
) -> None:
    """Analyze and save the top N errors for a model.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame containing predictions and text data with 'true_label' and 'pred_label' columns
    output_path : Path
        Path to save the error analysis results
    top_n : int, optional
        Number of top errors to analyze, by default 20
    """
    # Handle both column naming conventions
    true_col = 'true_label' if 'true_label' in predictions_df.columns else 'y_true'
    pred_col = 'pred_label' if 'pred_label' in predictions_df.columns else 'y_pred'
    
    # Create a DataFrame with true and predicted labels
    error_df = predictions_df[predictions_df[true_col] != predictions_df[pred_col]].copy()

    # Count occurrences of each (true_label, pred_label) pair
    error_counts = (
        error_df.groupby([true_col, pred_col])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(top_n)
    )

    # Save to CSV
    error_counts.to_csv(output_path, index=False)

def run_evaluations(
    model_names: List[str],
    *,
    artefacts_root: str | Path = "artefacts",
    output_dir: str | Path = "results",
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Compute metrics from persisted predictions.
    
    This function orchestrates the evaluation workflow by:
    1. Reading model predictions for both train and test sets
    2. Computing various metrics (accuracy, F1, etc.)
    3. Generating visualizations (confusion matrices, etc.)
    4. Comparing models based on their performance
    5. Computing overfitting metrics
    
    Parameters
    ----------
    model_names : List[str]
        List of model identifiers (must correspond to sub-folders inside
        *artefacts_root*).
    artefacts_root : str | Path, optional
        Directory containing model artifacts, by default "artefacts".
    output_dir : str | Path, optional
        Where to write evaluation artifacts (reports, matrices, etc.),
        by default "results".
    verbose : bool, optional
        Whether to log progress, by default True.
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping model names to their evaluation metrics.
    """
    logger = setup_logging(verbose)
    logger.info("Starting evaluation process")
    logger.info(f"Output directory for plots and results: {output_dir}")
    
    # Initialize results dictionary and ensure output directory exists
    results: Dict[str, Dict[str, Any]] = {}
    artefacts_root = Path(artefacts_root)
    output_dir = ensure_dir(output_dir)

    # Load all predictions
    predictions_dict = load_all_prediction_files(artefacts_root)
    
    if not predictions_dict:
        logger.error("No prediction files found")
        return results

    # Get unique classes from the first model's test predictions
    classes = sorted(predictions_dict[list(predictions_dict.keys())[0]]['test']['true_label'].unique())
    is_multiclass = len(classes) > 2

    # Process each model
    for name, model_predictions in predictions_dict.items():
        if name not in model_names:
            continue
            
        model_out_dir = ensure_dir(output_dir / name)
        logger.info(f"Processing model {name}")
        logger.info(f"Output directories for model {name}:")
        logger.info(f"  - Training results: {model_out_dir / 'train'}")
        logger.info(f"  - Test results: {model_out_dir / 'test'}")
        
        # Dictionary to store all results for this model
        model_results = {}
        
        # Process each split (train and test)
        for split_name, df in model_predictions.items():
            split_out_dir = ensure_dir(model_out_dir / split_name)
            
            # Extract true and predicted labels
            y_true = df['true_label'].values
            y_pred = df['pred_label'].values
            
            # Compute basic metrics
            metrics = {
                f'{split_name}_accuracy': accuracy_score(y_true, y_pred),
                f'{split_name}_macro_f1': f1_score(y_true, y_pred, average="macro"),
                f'{split_name}_weighted_f1': f1_score(y_true, y_pred, average="weighted"),
            }
            
            # Save classification report
            report_dict = classification_report(y_true, y_pred, output_dict=True)
            pd.DataFrame(report_dict).T.to_csv(split_out_dir / f"{split_name}_report.csv")
            
            # Store metrics
            model_results.update(metrics)
            
            # Additional analyses for test set
            if split_name == "test" and "text" in df.columns:
                logger.info(f"Processing top misclassifications for {name}")
                plot_top_misclassifications(
                    df,
                    split_out_dir / "top_misclassifications.csv"
                )
                
                # Analyze top 20 errors
                logger.info(f"Analyzing top 20 errors for {name}")
                analyze_top_errors(
                    df,
                    split_out_dir / f"{split_name}_top_20_errors.csv"
                )
                # Plot and save top n error types (default 10)
                plot_top_error_types(df, split_out_dir / "top_10_error_types.png", n=10)
        
        # Compute overfitting metrics if both train and test predictions are available
        if 'train' in model_predictions and 'test' in model_predictions:
            for metric in ['accuracy', 'macro_f1', 'weighted_f1']:
                train_metric = model_results.get(f'train_{metric}')
                test_metric = model_results.get(f'test_{metric}')
                if train_metric is not None and test_metric is not None:
                    model_results[f'{metric}_diff'] = train_metric - test_metric
        
        # Store model results
        results[name] = model_results

    # Create a comprehensive summary table
    summary_df = pd.DataFrame(results).T
    
    # Add "best" indicators for test metrics
    for metric in ["test_accuracy", "test_macro_f1", "test_weighted_f1"]:
        if metric in summary_df.columns:
            best_idx = summary_df[metric].idxmax()
            summary_df[f"{metric}_best"] = False
            summary_df.loc[best_idx, f"{metric}_best"] = True
    
    # Save summary metrics
    summary_df.to_csv(output_dir / "summary_metrics.csv")
    
    # Generate visualizations for each model
    logger.info("Generating visualizations for each model...")
    for name, model_predictions in predictions_dict.items():
        if name not in model_names:
            continue
            
        model_out_dir = ensure_dir(output_dir / name)
        logger.info(f"Generating visualizations for model {name}")
        
        # Process each split (train and test)
        for split_name, df in model_predictions.items():
            split_out_dir = ensure_dir(model_out_dir / split_name)
            logger.info(f"  - Processing {split_name} set visualizations")
            
            # Create a predictions dict with just this model and split
            single_model_predictions = {name: {split_name: df}}
            
            # Generate split-specific visualizations
            plot_label_distribution(single_model_predictions, split_out_dir)
            plot_confusion_matrix(single_model_predictions, split_out_dir)
            plot_precision_recall_curves(single_model_predictions, split_out_dir)
            visualize_error_distribution(single_model_predictions, split_out_dir)
            generate_detailed_error_report(single_model_predictions, split_out_dir)
    
    # Generate model comparison visualizations if multiple models
    if len(model_names) > 1:
        plot_model_comparisons(predictions_dict, output_dir)
        
    # Perform error analysis
    logger.info("Performing error analysis...")
    
    # Analyze error patterns
    error_patterns = analyse_error_patterns(predictions_dict)
    error_patterns.to_csv(output_dir / 'common_error_patterns.csv', index=False)
    
    # Identify consistently misclassified examples
    misclass_examples = consistently_misclassified(predictions_dict, min_models=len(predictions_dict))
    if not misclass_examples.empty:
        misclass_examples.to_csv(output_dir / 'consistently_misclassified.csv', index=False)
    
    # Analyze text features
    if "text" in predictions_dict[list(predictions_dict.keys())[0]]['test'].columns:
        text_features = analyze_text_features(predictions_dict)
        text_features.to_csv(output_dir / 'text_features_analysis.csv', index=False)

    return results

def display_detailed_results(results: Dict[str, Dict[str, Any]]) -> None:
    """Display detailed evaluation results in a formatted way.
    
    This function displays:
    1. Test metrics summary
    2. Train metrics summary (if available)
    3. Train/Test differences for overfitting analysis (if available)
    
    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Dictionary mapping model names to their evaluation metrics.
    """
    print("\n=== Summary of Test Metrics ===")
    test_metrics = pd.DataFrame({
        model: {k: v for k, v in metrics.items() if k.startswith('test_')}
        for model, metrics in results.items()
    }).T
    display(test_metrics)

    # Show train metrics if available
    train_cols = [col for col in next(iter(results.values())).keys() if col.startswith('train_')]
    if train_cols:
        print("\n=== Summary of Train Metrics ===")
        train_metrics = pd.DataFrame({
            model: {k: v for k, v in metrics.items() if k.startswith('train_')}
            for model, metrics in results.items()
        }).T
        display(train_metrics)
        
        # Show potential overfitting metrics
        diff_cols = [col for col in next(iter(results.values())).keys() if col.endswith('_diff')]
        if diff_cols:
            print("\n=== Train/Test Differences (Overfitting Analysis) ===")
            diff_metrics = pd.DataFrame({
                model: {k: v for k, v in metrics.items() if k.endswith('_diff')}
                for model, metrics in results.items()
            }).T
            display(diff_metrics)
            
            # Interpretation guideline
            print("\nInterpretation guide:")
            print("- Positive values indicate potential overfitting (model performs better on training data)")
            print("- Values close to zero indicate good generalization")
            print("- Negative values might indicate underfitting or data leakage issues")
