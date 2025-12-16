"""
Utility functions for the evaluation package.

This module provides helper functions used across the evaluation package.
"""

import logging
from pathlib import Path

import pandas as pd


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
    # Get logger and set level (don't configure root logger - that's for entry points)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    return logger


def load_all_prediction_files(experiment_dir: str | Path) -> dict[str, dict[str, pd.DataFrame]]:
    """Load every CSV prediction file from model directories into a dict.

    Parameters
    ----------
    experiment_dir : str or Path
        Directory containing model directories with prediction files.

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames.
    """
    exp = Path(experiment_dir)
    dfs: dict[str, dict[str, pd.DataFrame]] = {}

    # Find all model directories (now all at same depth since names are sanitized)
    model_dirs = [d for d in exp.glob("*") if d.is_dir()]

    for model_dir in model_dirs:
        model_name = model_dir.name
        dfs[model_name] = {}

        # Process both train and test predictions
        for split in ["train", "test"]:
            pred_file = model_dir / f"{split}_predictions.csv"
            if pred_file.exists():
                logger = setup_logging(True)
                logger.info(f"Loading predictions from {pred_file}")
                df = pd.read_csv(pred_file)

                # Add model name column
                df["model"] = model_name

                # Add ID column if it doesn't exist
                if "id" not in df.columns:
                    df["id"] = range(len(df))

                # Add text column if it doesn't exist
                if "text" not in df.columns:
                    df["text"] = "Placeholder text"

                dfs[model_name][split] = df
                logger.info(f"Successfully loaded {split} predictions for {model_name}")

    if not dfs:
        # Return empty dict instead of raising - let caller handle gracefully
        return {}

    return dfs


def analyse_error_patterns(pred_dfs: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Return dataframe with a row per distinct (true -> pred) error."""
    frames = []
    for name, splits in pred_dfs.items():
        for split_name, df in splits.items():
            errs = df[df["y_true"] != df["y_pred"]].copy()
            # Convert to string to handle both string and numeric labels
            errs["error_type"] = errs["y_true"].astype(str) + " -> " + errs["y_pred"].astype(str)
            errs["model"] = name
            errs["split"] = split_name
            frames.append(errs)
    if not frames:
        return pd.DataFrame(columns=["error_type", "total_count"])
    merged = pd.concat(frames, ignore_index=True)
    return (
        merged.groupby("error_type", as_index=False)
        .size()
        .rename(columns={"size": "total_count"})
        .sort_values("total_count", ascending=False)
    )


def consistently_misclassified(pred_dfs: dict[str, dict[str, pd.DataFrame]], min_models: int = 2):
    """Docs misclassified by >= min_models models in exactly the same way.

    Parameters
    ----------
    pred_dfs : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary mapping model names to another dictionary with 'train' and 'test' DataFrames.
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
    for name, splits in pred_dfs.items():
        # We'll only look at test set predictions for consistency
        if "test" in splits:
            df = splits["test"]
            wrong = df[df["y_true"] != df["y_pred"]][["id", "text", "y_true", "y_pred"]].copy()
            wrong[name] = True
            if combined is None:
                combined = wrong
            else:
                combined = combined.merge(wrong, how="outer")

    if combined is None:
        return pd.DataFrame()

    # Fill NaN values and convert to bool, avoiding pandas deprecation warning
    # Use infer_objects to avoid FutureWarning about downcasting
    combined = combined.infer_objects(copy=False).fillna(False)
    # Convert object columns to bool explicitly to avoid FutureWarning
    for col in combined.columns:
        if col not in ["id", "text", "y_true", "y_pred"]:
            combined[col] = combined[col].astype(bool)
    mask = combined.drop(columns=["id", "text", "y_true", "y_pred"]).sum(1) >= min_models
    return combined[mask]
