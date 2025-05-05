"""
Utility functions for the evaluation package.

This module provides helper functions used across the evaluation package.
"""

import logging
from pathlib import Path
from typing import Union, Dict, List, Any

import pandas as pd

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to the directory to ensure exists.
        
    Returns
    -------
    Path
        The path to the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def setup_logging(verbose: bool = True) -> logging.Logger:
    """Set up logging configuration.
    
    Parameters
    ----------
    verbose : bool, optional
        Whether to enable verbose logging, by default True.
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
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

def load_all_prediction_files(experiment_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Load every CSV prediction file from model directories into a dict.
    
    Parameters
    ----------
    experiment_dir : str or Path
        Directory containing model directories with prediction files.
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping model names to their prediction dataframes.
    """
    exp = Path(experiment_dir)
    dfs: Dict[str, pd.DataFrame] = {}
    
    # Find all model directories
    model_dirs = [d for d in exp.glob('*') if d.is_dir()]
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        test_pred_file = model_dir / "test_predictions.csv"
        
        if test_pred_file.exists():
            df = pd.read_csv(test_pred_file)
            
            # Rename columns if necessary to match expected names
            if 'y_true' in df.columns and 'y_pred' in df.columns:
                df = df.rename(columns={
                    'y_true': 'true_label',
                    'y_pred': 'pred_label'
                })
            
            # Add model name column
            df['model'] = model_name
            
            # Add ID column if it doesn't exist
            if 'id' not in df.columns:
                df['id'] = range(len(df))
            
            # Add text column if it doesn't exist
            if 'text' not in df.columns:
                df['text'] = "Placeholder text"  # In a real scenario, you would join with a dataset containing the text
            
            dfs[model_name] = df
    
    if not dfs:
        raise FileNotFoundError(f'No prediction files found in {exp}')
        
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
    """Docs misclassified by >= min_models models in exactly the same way.
    
    Parameters
    ----------
    pred_dfs : Dict[str, pd.DataFrame]
        Dictionary mapping model names to their prediction dataframes.
    min_models : int, optional
        Minimum number of models that must misclassify a document in the same way,
        by default 2.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing consistently misclassified examples with their true and
        predicted labels, and which models misclassified them.
    """
    combined = None
    for name, df in pred_dfs.items():
        wrong = df[df.true_label != df.pred_label][['id', 'text', 'true_label', 'pred_label']].copy()
        wrong[name] = True
        combined = wrong if combined is None else combined.merge(wrong, how='outer')
    combined = combined.fillna(False)
    mask = combined.drop(columns=['id', 'text', 'true_label', 'pred_label']).sum(1) >= min_models
    return combined[mask]

def export_analysis_results(results: Dict[str, Any], output_dir: Union[str, Path]) -> None:
    """Export analysis results to files.
    
    Parameters
    ----------
    results : dict
        Dictionary with analysis results
    output_dir : str or Path
        Directory to save results
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