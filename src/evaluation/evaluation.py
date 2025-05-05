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
)
from .utils import (
    ensure_dir,
    setup_logging,
    load_all_prediction_files,
    analyse_error_patterns,
    consistently_misclassified,
)

def run_evaluations(
    model_names: List[str],
    *,
    y_true: Sequence[Any],
    y_train_true: Optional[Sequence[Any]] = None,
    artefacts_root: str | Path = "artefacts",
    output_dir: str | Path = "results",
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Compute metrics from persisted predictions.
    
    This function orchestrates the evaluation workflow by:
    1. Reading model predictions and probabilities
    2. Computing various metrics (accuracy, F1, ROC AUC, etc.)
    3. Generating visualizations (confusion matrices, ROC curves, etc.)
    4. Comparing training vs test performance
    5. Computing overfitting metrics
    6. Additional analyses (misclassifications, calibration) when data is available
    
    Parameters
    ----------
    model_names : List[str]
        List of model identifiers (must correspond to sub-folders inside
        *artefacts_root*).
    y_true : Sequence[Any]
        Ground-truth labels for the test set – must be in the same order
        that was used when generating the predictions.
    y_train_true : Optional[Sequence[Any]], optional
        Ground-truth labels for the training set – if provided,
        training metrics will also be computed, by default None.
    artefacts_root : str | Path, optional
        Directory containing model artifacts, by default "artefacts".
    output_dir : str | Path, optional
        Where to write evaluation artifacts (reports, matrices, ROC curves),
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
    
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    classes = np.unique(y_true)
    is_multiclass = len(classes) > 2
    
    # Prepare binarized versions of labels for ROC and PR curves
    y_true_bin = label_binarize(y_true, classes=classes) if is_multiclass else y_true
    
    if y_train_true is not None:
        y_train_true = np.asarray(y_train_true)
        y_train_true_bin = label_binarize(y_train_true, classes=classes) if is_multiclass else y_train_true

    # Initialize results dictionary and ensure output directory exists
    results: Dict[str, Dict[str, Any]] = {}
    artefacts_root = Path(artefacts_root)
    output_dir = ensure_dir(output_dir)

    # Process each model
    for name in model_names:
        model_dir = artefacts_root / name
        model_out_dir = ensure_dir(output_dir / name)
        train_out_dir = ensure_dir(model_out_dir / "train")
        test_out_dir = ensure_dir(model_out_dir / "test")
        logger.info(f"Output directories for model {name}:")
        logger.info(f"  - Training results: {train_out_dir}")
        logger.info(f"  - Test results: {test_out_dir}")

        # Dictionary to store all results for this model
        model_results = {}

        # Define splits to process
        splits = [
            {
                "name": "test",
                "y_true": y_true,
                "y_true_bin": y_true_bin,
                "pred_file": "test_predictions.csv",
                "prob_file": "test_prob.npy",
                "prefix": "test_",
                "out_dir": test_out_dir
            }
        ]
        
        if y_train_true is not None:
            splits.append({
                "name": "train",
                "y_true": y_train_true,
                "y_true_bin": y_train_true_bin,
                "pred_file": "train_predictions.csv",
                "prob_file": "train_prob.npy",
                "prefix": "train_",
                "out_dir": train_out_dir
            })

        # Process each split
        for split in splits:
            # Load predictions and compute metrics
            metrics = compute_metrics(
                model_dir=model_dir,
                split=split,
                classes=classes,
                is_multiclass=is_multiclass,
                logger=logger
            )
            
            if metrics is None:
                continue
                
            y_pred, y_prob, split_metrics = metrics
            
            # Generate visualizations
            logger.info(f"Processing distribution analysis for {name} on {split['name']} set")
            plot_label_distribution(
                split["y_true"], 
                y_pred, 
                classes,
                split["name"].title(),
                name,
                split["out_dir"] / f"{split['name']}_label_distribution.png"
            )
            
            logger.info(f"Processing confusion matrix for {name} on {split['name']} set")
            plot_confusion_matrix(
                split["y_true"],
                y_pred,
                classes,
                split["name"].title(),
                name,
                split["out_dir"] / f"{split['name']}_confusion.png"
            )
            
            # Store metrics
            model_results.update(split_metrics)
            
            # Process probabilities if available
            if y_prob is not None:
                # Generate ROC curves
                logger.info(f"Processing ROC curves for {name} on {split['name']} set")
                plot_roc_curves(
                    split["y_true_bin"],
                    y_prob,
                    classes,
                    split["name"].title(),
                    name,
                    split["out_dir"] / f"{split['name']}_roc_curves.png",
                    is_multiclass
                )
                
                # Generate precision-recall curves
                logger.info(f"Processing precision-recall curves for {name} on {split['name']} set")
                plot_precision_recall_curves(
                    split["y_true_bin"],
                    y_prob,
                    classes,
                    split["name"].title(),
                    name,
                    split["out_dir"] / f"{split['name']}_pr_curves.png",
                    is_multiclass
                )
            
            # Additional analyses for test set
            if split["name"] == "test":
                # Top misclassifications if text data is available
                if "text" in pd.read_csv(model_dir / split["pred_file"]).columns:
                    logger.info(f"Processing top misclassifications for {name}")
                    plot_top_misclassifications(
                        pd.read_csv(model_dir / split["pred_file"])["text"].values,
                        split["y_true"],
                        y_pred,
                        split["out_dir"] / "top_misclassifications.csv"
                    )

        # Store model results
        results[name] = model_results

    # Create a comprehensive summary table
    summary_df = pd.DataFrame(results).T
    
    # Add "best" indicators
    for metric in ["test_accuracy", "test_macro_f1", "test_weighted_f1", "test_roc_auc"]:
        if metric in summary_df.columns:
            best_idx = summary_df[metric].idxmax()
            summary_df[f"{metric}_best"] = False
            summary_df.loc[best_idx, f"{metric}_best"] = True
    
    # Save summary metrics
    summary_df.to_csv(output_dir / "summary_metrics.csv")
    
    # Generate model comparison visualizations if multiple models
    if len(model_names) > 1:
        plot_model_comparisons(summary_df, model_names, output_dir)

    # Load all prediction files for error analysis
    predictions_dict = load_all_prediction_files(artefacts_root)
    
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
    text_features = analyze_text_features(predictions_dict)
    text_features.to_csv(output_dir / 'text_feature_analysis.csv', index=False)
    
    # Create error distribution visualizations
    visualize_error_distribution(predictions_dict, output_dir)
    
    # Generate detailed error report
    generate_detailed_error_report(predictions_dict, output_dir)
    
    # Add error analysis results to the main results dictionary
    for model_name in results:
        results[model_name].update({
            'error_patterns': error_patterns,
            'hard_cases': misclass_examples,
            'text_features': text_features
        })

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