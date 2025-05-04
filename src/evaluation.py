"""\
Evaluation module for text classification models.

This module provides functions for evaluating text classification models,
including computing metrics, generating visualizations, and analyzing
errors. It supports various evaluation tasks like classification reports,
confusion matrices, ROC curves, and error analysis.

Functions:
- run_evaluations: Run comprehensive model evaluation
- _persist_report: Save classification report
- _persist_confusion: Save confusion matrix
- _bootstrap_f1: Perform bootstrap test for F1 score
- _persist_roc_curves: Save ROC curves
- _collect_worst_errors: Collect confident misclassifications
- collect_evaluation_results: Organize evaluation results from multiple models into overall and per-class metrics
- compare_models: Create a DataFrame comparing models based on selected metrics
- print_evaluation_results: Format and print the evaluation results in a readable way

Created: 2025-05-03
"""

import os
import logging
from typing import Optional, Union
from src.utils.model_storage import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import pandas as pd

def _snug(s: str) -> str:
    """Remove whitespace from a string.
    
    Args:
        s: Input string
        
    Returns:
        String with whitespace removed
    """
    return ''.join(s.split())

def run_evaluations(model_or_path: Union[Any, str],
                    X_test: List[str], y_test: np.ndarray,
                    X_train: Optional[List[str]] = None,
                    y_train: Optional[np.ndarray] = None,
                    output_dir: str = "results",
                    model_name: str = "unknown") -> Dict[str, Any]:
    """Run comprehensive evaluation of a text classification model.
    
    This function performs a complete evaluation of a text classifier,
    including computing metrics, generating visualizations, and analyzing
    errors. It saves all results to the specified output directory.
    
    Args:
        model_or_path: *Either* a fitted model **or** a path to a
            ``.joblib`` file saved by `run_trainings`
        X_test: List of test documents
        y_test: Array of test labels
        X_train, y_train: (optional) supply to compute train metrics
        output_dir: Directory to save results (default: 'results')
        model_name: Name of the model for logging purposes
        
    Returns:
        Dictionary containing evaluation results
        
    Example:
        >>> results = run_evaluations(model, X_test, y_test)
        >>> print(f"Macro F1: {results['macro_f1']:.3f}")
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"🔍 Evaluating model: {model_name}")
    
    # Load the model if a string path was supplied
    if isinstance(model_or_path, str):
        logging.info(f"  • Loading model from {model_or_path}")
        model = load_model(model_or_path)
    else:
        model = model_or_path

    # Evaluate both splits
    splits = {"test": (X_test, y_test)}
    if X_train is not None and y_train is not None:
        splits["train"] = (X_train, y_train)

    # Prepare the results dictionary with top-level metrics
    results = {}
    
    for split, (X_split, y_split) in splits.items():
        logging.info(f"  • Evaluating on {split} set ({len(X_split)} samples)...")
        
        # Make predictions
        y_pred = model.predict(X_split)
        
        # Calculate metrics
        accuracy = accuracy_score(y_split, y_pred)
        logging.info(f"    ✓ {split.capitalize()} accuracy: {accuracy:.4f}")
        
        # Calculate classification report
        report_dict = classification_report(y_split, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_split, y_pred)
        
        # Extract scores
        macro_f1 = report_dict['macro avg']['f1-score']
        weighted_f1 = report_dict['weighted avg']['f1-score']
        
        logging.info(f"    ✓ {split.capitalize()} macro F1: {macro_f1:.4f}, weighted F1: {weighted_f1:.4f}")
        
        # Save results
        report_path = os.path.join(output_dir, f"{split}_report.csv")
        confusion_path = os.path.join(output_dir, f"{split}_confusion.png")
        
        _persist_report(report_df, report_path)
        _persist_confusion(cm, confusion_path)
        logging.info(f"    ✓ Saved {split} reports to {output_dir}")
        
        # Store metrics in the split-specific results
        results[split] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "report": report_dict
        }
    
    # For compatibility with the notebook, add top-level metrics using test split
    if 'test' in results:
        results['accuracy'] = results['test']['accuracy']
        results['macro_f1'] = results['test']['macro_f1'] 
        results['weighted_f1'] = results['test']['weighted_f1']
        results['f1_macro'] = results['test']['macro_f1']  # For backward compatibility
    
    logging.info(f"  ✓ Evaluation of {model_name} completed")
    
    return results

def _persist_report(report: pd.DataFrame, path: str) -> None:
    """Save classification report to CSV file.
    
    Args:
        report: Classification report as DataFrame
        path: Path to save the report
    """
    report.to_csv(path)

def _persist_confusion(cm: np.ndarray, path: str) -> None:
    """Save confusion matrix visualization.
    
    Args:
        cm: Confusion matrix
        path: Path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(path)
    plt.close()

def _bootstrap_f1(y_true: np.ndarray, y_pred: np.ndarray,
                 n_iterations: int = 1000) -> float:
    """Perform bootstrap test for F1 score significance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_iterations: Number of bootstrap iterations
        
    Returns:
        p-value for the F1 score
    """
    observed_f1 = f1_score(y_true, y_pred, average='macro')
    null_f1s = []
    
    for _ in range(n_iterations):
        y_shuffled = np.random.permutation(y_pred)
        null_f1 = f1_score(y_true, y_shuffled, average='macro')
        null_f1s.append(null_f1)
    
    p_value = np.mean(null_f1s >= observed_f1)
    return p_value

def _persist_roc_curves(model: Any, X_test: List[str], y_test: np.ndarray,
                       path: str) -> None:
    """Save ROC curves for each class.
    
    Args:
        model: Trained classifier with predict_proba method
        X_test: Test documents
        y_test: Test labels
        path: Path to save the visualization
    """
    y_prob = model.predict_proba(X_test)
    n_classes = y_prob.shape[1]
    
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(path)
    plt.close()

def _collect_worst_errors(model: Any, X_test: List[str], y_test: np.ndarray,
                         y_pred: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
    """Collect confident misclassifications for error analysis.
    
    Args:
        model: Trained classifier with predict_proba method
        X_test: Test documents
        y_test: Test labels
        y_pred: Predicted labels
        top_k: Number of worst errors to collect
        
    Returns:
        List of error dictionaries containing text, true label,
        predicted label, and confidence
    """
    if not hasattr(model, 'predict_proba'):
        return []
        
    y_prob = model.predict_proba(X_test)
    errors = []
    
    for i in range(len(X_test)):
        if y_test[i] != y_pred[i]:
            confidence = y_prob[i, y_pred[i]]
            errors.append({
                'text': X_test[i],
                'true_label': y_test[i],
                'predicted_label': y_pred[i],
                'confidence': confidence
            })
    
    # Sort by confidence (descending)
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    return errors[:top_k]

def collect_evaluation_results(models_results: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Dict[str, float]]]]:
    """Organize evaluation results from multiple models into overall and per-class metrics.
    
    This function processes evaluation results from multiple models and organizes them into:
    1. Overall metrics (accuracy, F1, precision, recall) per model
    2. Per-class metrics (precision, recall, F1) for each model
    
    Args:
        models_results: Dictionary mapping model names to their evaluation results
            
    Returns:
        Tuple containing:
            - Dictionary mapping model names to overall metrics
            - Dictionary mapping class names to model-specific metrics
            
    Example:
        >>> overall, per_class = collect_evaluation_results(all_results)
        >>> print(f"Best model by accuracy: {max(overall.items(), key=lambda x: x[1]['accuracy'])[0]}")
    """
    # Initialize dictionaries to store overall and per-class results
    overall_results = {}
    class_results = {}

    # Process each model's results
    for name, result in models_results.items():
        if 'test' in result:
            test_result = result['test']
            report = test_result['report']
            
            # Store overall metrics
            overall_results[name] = {
                'accuracy': report['accuracy'],
                'macro_f1': test_result['macro_f1'],
                'weighted_f1': report['weighted avg']['f1-score'],
                'macro_precision': report['macro avg']['precision'],
                'macro_recall': report['macro avg']['recall'],
                'weighted_precision': report['weighted avg']['precision'],
                'weighted_recall': report['weighted avg']['recall']
            }
            
            # Store per-class metrics
            for class_name in [c for c in report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]:
                if class_name not in class_results:
                    class_results[class_name] = {}
                
                class_results[class_name][name] = {
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1-score': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                }
        else:
            print(f"Warning: 'test' key not found in results for {name}")
    
    return overall_results, class_results

def compare_models(overall_results: Dict[str, Dict[str, float]], 
                   metrics: List[str] = ['accuracy', 'macro_f1', 'weighted_f1']) -> pd.DataFrame:
    """Create a DataFrame comparing models based on selected metrics.
    
    Args:
        overall_results: Dictionary mapping model names to overall metrics
        metrics: List of metrics to include in the comparison
        
    Returns:
        DataFrame with models as rows and metrics as columns
        
    Example:
        >>> df = compare_models(overall_results)
        >>> print(df)
    """
    df_results = pd.DataFrame(overall_results).T[metrics]
    column_mapping = {
        'accuracy': 'Accuracy',
        'macro_f1': 'Macro F1', 
        'weighted_f1': 'Weighted F1',
        'macro_precision': 'Macro Precision',
        'macro_recall': 'Macro Recall',
        'weighted_precision': 'Weighted Precision', 
        'weighted_recall': 'Weighted Recall'
    }
    
    df_results.columns = [column_mapping.get(col, col) for col in metrics]
    return df_results.round(3)

def print_evaluation_results(overall_results: Dict[str, Dict[str, float]], 
                             class_results: Dict[str, Dict[str, Dict[str, float]]],
                             num_sample_columns: int = 10) -> None:
    """Format and print the evaluation results in a readable way.
    
    This function takes the results from collect_evaluation_results and 
    prints them in a formatted, human-readable way, including:
    1. Overall metrics for all models
    2. Per-class metrics for each model
    3. A multi-level indexed view for detailed comparison
    
    Args:
        overall_results: Dictionary mapping model names to overall metrics
        class_results: Dictionary mapping class names to model-specific metrics
        num_sample_columns: Number of columns to display in the detailed metrics sample
        
    Example:
        >>> overall, per_class = collect_evaluation_results(all_results)
        >>> print_evaluation_results(overall, per_class)
    """
    import pandas as pd
    
    # Create overall metrics dataframe
    df_overall = pd.DataFrame(overall_results).T
    df_overall = df_overall.round(3)

    print("Overall Model Comparison:")
    print(df_overall)

    # Create per-class dataframes
    print("\nPer-Class Model Comparison:")
    for class_name, class_data in class_results.items():
        df_class = pd.DataFrame(class_data).T
        df_class = df_class.round(3)
        print(f"\nClass: {class_name}")
        print(df_class)

    # Alternative: Create one multi-level dataframe for per-class metrics
    # This creates a hierarchical dataframe with classes and metrics
    multiindex_data = {}
    for class_name, class_data in class_results.items():
        for model_name, metrics in class_data.items():
            if model_name not in multiindex_data:
                multiindex_data[model_name] = {}
            for metric_name, value in metrics.items():
                multiindex_data[model_name][f"{class_name}_{metric_name}"] = value

    df_multiindex = pd.DataFrame(multiindex_data).T
    df_multiindex = df_multiindex.round(3)

    print("\nDetailed per-class metrics (sample):")
    # Display just a subset of columns for readability
    print(df_multiindex.iloc[:, :num_sample_columns])