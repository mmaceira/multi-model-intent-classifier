"""
Utility Functions for Model Evaluation

This module provides essential utility functions for the evaluation package,
supporting various aspects of model assessment and analysis.

Key Features:
- Logging configuration and management
- Prediction file loading and processing
- Error pattern analysis
- Consistent misclassification detection
- Results export and visualization

Functions:
    setup_logging: Configures logging for the evaluation process
    load_all_prediction_files: Loads prediction files from model directories
    analyse_error_patterns: Analyzes common error patterns across models
    consistently_misclassified: Identifies consistently misclassified examples
    export_analysis_results: Exports analysis results to files

Dependencies:
    logging: For progress tracking and error reporting
    pandas: For data manipulation
    numpy: For numerical operations
    matplotlib: For visualization
    seaborn: For enhanced visualization
    pathlib: For file system operations

Example Usage:
    >>> from pathlib import Path
    >>> from evaluation.utils import setup_logging, load_all_prediction_files
    >>> 
    >>> # Set up logging
    >>> logger = setup_logging(verbose=True)
    >>> 
    >>> # Load prediction files
    >>> predictions = load_all_prediction_files(Path("experiments/model1"))
    >>> 
    >>> # Analyze error patterns
    >>> error_patterns = analyse_error_patterns(predictions)
    >>> 
    >>> # Export results
    >>> export_analysis_results(
    ...     results={
    ...         "error_patterns": error_patterns,
    ...         "misclassified_examples": misclassified_examples,
    ...         "text_features": text_features
    ...     },
    ...     output_dir=Path("results/analysis")
    ... )
"""

import logging
from pathlib import Path
from typing import Union, Dict, List, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.file_ops import ensure_dir

def setup_logging(verbose: bool = True) -> logging.Logger:
    """
    Set up logging configuration for the evaluation process.
    
    This function configures the logging system with appropriate formatting
    and verbosity levels for model evaluation tasks.
    
    Parameters
    ----------
    verbose : bool, optional
        Whether to enable verbose logging, by default True.
        When True, sets logging level to INFO, otherwise WARNING.
        
    Returns
    -------
    logging.Logger
        Configured logger instance for use in evaluation functions.
        
    Example
    -------
    >>> logger = setup_logging(verbose=True)
    >>> logger.info("Starting model evaluation")
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(levelname)s | %(message)s',
        force=True  # Force reconfiguration of the root logger
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    return logger

def load_all_prediction_files(experiment_dir: str | Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load prediction files from all model directories in an experiment.
    
    This function recursively searches for prediction files in model directories
    and loads them into a structured dictionary. It handles both train and test
    predictions, adding necessary metadata and ensuring consistent structure.
    
    Parameters
    ----------
    experiment_dir : str or Path
        Directory containing model directories with prediction files.
        Each model directory should contain 'train_predictions.csv' and/or
        'test_predictions.csv' files.
        
    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames.
        Each DataFrame contains:
        - id: Unique identifier for each example
        - text: Input text (or placeholder if not available)
        - y_true: True labels
        - y_pred: Predicted labels
        - model: Model name
        - Additional columns from the prediction files
        
    Raises
    ------
    FileNotFoundError
        If no prediction files are found in the experiment directory
        
    Example
    -------
    >>> predictions = load_all_prediction_files(Path("experiments/model1"))
    >>> print(predictions["model1"]["test"].head())
    """
    exp = Path(experiment_dir)
    dfs: Dict[str, Dict[str, pd.DataFrame]] = {}
    
    # Find all model directories
    model_dirs = [d for d in exp.glob('*') if d.is_dir()]
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        dfs[model_name] = {}
        
        # Process both train and test predictions
        for split in ['train', 'test']:
            pred_file = model_dir / f"{split}_predictions.csv"
            if pred_file.exists():
                logger = setup_logging(True)
                logger.info(f"Loading predictions from {pred_file}")
                df = pd.read_csv(pred_file)
                
                # Add model name column
                df['model'] = model_name
                
                # Add ID column if it doesn't exist
                if 'id' not in df.columns:
                    df['id'] = range(len(df))
                
                # Add text column if it doesn't exist
                if 'text' not in df.columns:
                    df['text'] = "Placeholder text"
                
                dfs[model_name][split] = df
                logger.info(f"Successfully loaded {split} predictions for {model_name}")
    
    if not dfs:
        raise FileNotFoundError(f'No prediction files found in {exp}')
        
    return dfs

def analyse_error_patterns(pred_dfs: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Analyze and summarize error patterns across models and splits.
    
    This function identifies common error patterns by analyzing misclassifications
    across different models and data splits. It helps understand systematic errors
    and model behavior.
    
    Parameters
    ----------
    pred_dfs : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames.
        Each DataFrame should contain:
        - y_true: True labels
        - y_pred: Predicted labels
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing error pattern analysis with columns:
        - error_type: String describing the error (true_label -> predicted_label)
        - total_count: Number of times the error occurred
        - model: Model that made the error
        - split: Data split where the error occurred
        
    Example
    -------
    >>> error_patterns = analyse_error_patterns(predictions)
    >>> print(error_patterns.head())
    """
    frames = []
    for name, splits in pred_dfs.items():
        for split_name, df in splits.items():
            errs = df[df['y_true'] != df['y_pred']].copy()
            errs['error_type'] = errs['y_true'] + ' -> ' + errs['y_pred']
            errs['model'] = name
            errs['split'] = split_name
            frames.append(errs)
    if not frames:
        return pd.DataFrame(columns=['error_type', 'total_count'])
    merged = pd.concat(frames, ignore_index=True)
    return (merged.groupby('error_type', as_index=False)
                  .size()
                  .rename(columns={'size': 'total_count'})
                  .sort_values('total_count', ascending=False))

def consistently_misclassified(
    pred_dfs: Dict[str, Dict[str, pd.DataFrame]],
    min_models: int = 2
) -> pd.DataFrame:
    """
    Identify examples that are consistently misclassified by multiple models.
    
    This function finds documents that are misclassified in the same way by
    at least a specified number of models, helping to identify particularly
    challenging or ambiguous examples.
    
    Parameters
    ----------
    pred_dfs : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames.
        Each DataFrame should contain:
        - id: Unique identifier for each example
        - text: Input text
        - y_true: True labels
        - y_pred: Predicted labels
    min_models : int, optional
        Minimum number of models that must misclassify a document in the same way,
        by default 2.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing consistently misclassified examples with columns:
        - id: Example identifier
        - text: Input text
        - y_true: True label
        - y_pred: Predicted label
        - model_columns: Boolean columns indicating which models misclassified the example
        
    Example
    -------
    >>> misclassified = consistently_misclassified(predictions, min_models=3)
    >>> print(misclassified.head())
    """
    combined = None
    for name, splits in pred_dfs.items():
        # We'll only look at test set predictions for consistency
        if 'test' in splits:
            df = splits['test']
            wrong = df[df['y_true'] != df['y_pred']][['id', 'text', 'y_true', 'y_pred']].copy()
            wrong[name] = True
            combined = wrong if combined is None else combined.merge(wrong, how='outer')
    
    if combined is None:
        return pd.DataFrame()
        
    combined = combined.fillna(False)
    mask = combined.drop(columns=['id', 'text', 'y_true', 'y_pred']).sum(1) >= min_models
    return combined[mask]

def export_analysis_results(
    results: Dict[str, Any],
    output_dir: Union[str, Path]
) -> None:
    """
    Export analysis results to various file formats.
    
    This function saves different types of analysis results to appropriate
    file formats in the specified output directory.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing analysis results with keys:
        - error_patterns: DataFrame with error pattern analysis
        - misclassified_examples: DataFrame with misclassified examples
        - text_features: DataFrame with text feature analysis
    output_dir : Union[str, Path]
        Directory where results will be saved
        
    Returns
    -------
    None
        This function saves files but does not return any values.
        
    Example
    -------
    >>> export_analysis_results(
    ...     results={
    ...         "error_patterns": error_patterns,
    ...         "misclassified_examples": misclassified_examples,
    ...         "text_features": text_features
    ...     },
    ...     output_dir=Path("results/analysis")
    ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save error patterns
    if not results['error_patterns'].empty:
        results['error_patterns'].to_csv(output_dir / 'error_patterns.csv', index=False)
    
    # Save misclassified examples
    if not results['misclassified_examples'].empty:
        results['misclassified_examples'].to_csv(output_dir / 'misclassified_examples.csv', index=False)
    
    # Save text features
    if not results['text_features'].empty:
        results['text_features'].to_csv(output_dir / 'text_features.csv', index=False)
    
    print(f"Analysis results exported to {output_dir}")
