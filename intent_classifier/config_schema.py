"""Configuration Schema Validation

This module provides Pydantic models for validating configuration files.
It ensures type safety and catches configuration errors at startup.

Usage:
    >>> from intent_classifier.config_schema import load_and_validate_config
    >>> config = load_and_validate_config("config/dataset/{dataset_name}/tiny.yaml")
    >>> print(config.general.run_name)
    "experiment_tiny_dataset"
"""

from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class GeneralCfg(BaseModel):
    """General configuration settings."""

    run_name: str = Field(..., description="Experiment identifier")
    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")


class DatasetCfg(BaseModel):
    """Dataset configuration settings."""

    name: str = Field(..., description="Dataset name (must match a dataset in config/dataset/)")
    use_oos: bool = Field(default=False, description="Include out-of-scope examples")
    multilabel: bool = Field(default=False, description="Enable multi-label mode")
    max_classes: Optional[int] = Field(None, ge=1, description="Limit number of classes")
    max_train_samples: Optional[int] = Field(None, ge=1, description="Limit training samples")
    max_test_samples: Optional[int] = Field(None, ge=1, description="Limit test samples")
    min_samples_per_label: Optional[int] = Field(
        None, ge=1, description="Minimum samples per label (filters rare labels)"
    )


class PathsCfg(BaseModel):
    """Path configuration settings."""

    data_exploration_dir: str = Field(..., description="Data exploration output directory")
    embeddings_dir: str = Field(..., description="Embeddings output directory")
    models_dir: str = Field(..., description="Models output directory")
    predictions_dir: str = Field(..., description="Predictions output directory")
    results_dir: str = Field(..., description="Results output directory")


class ModelCfg(BaseModel):
    """Model configuration settings."""

    embedding_backend: Literal["sbert", "openai"] = Field(
        default="sbert", description="Embedding backend to use"
    )
    sbert_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="SBERT model name",
    )
    openai_model_name: str = Field(
        default="text-embedding-3-small", description="OpenAI model name"
    )
    classifier: Literal["linear_svm", "naive_bayes", "transformer_logreg"] = Field(
        default="linear_svm", description="Default classifier type"
    )
    rag_top_k: int = Field(default=25, ge=1, description="Number of neighbors for RAG models")
    llm_model: str = Field(
        default="ollama/llama3.1:8b", description="LLM model for RAG-LLM classification"
    )
    ollama_endpoint: Optional[str] = Field(
        default=None,
        description="Ollama API endpoint URL (e.g., http://localhost:11434). "
        "If not set, uses OLLAMA_API_BASE env var or default http://localhost:11434",
    )


class TrainingCfg(BaseModel):
    """Training configuration settings."""

    # Allow flexible training config (can be extended)
    model_config: Optional[str] = Field(None, description="Path to models config file")

    # Add other training-specific fields as needed
    # For now, we'll allow arbitrary fields
    class Config:
        extra = "allow"


class EvaluationCfg(BaseModel):
    """Evaluation configuration settings."""

    # Allow flexible evaluation config (can be extended)
    # For now, we'll allow arbitrary fields
    class Config:
        extra = "allow"


class Config(BaseModel):
    """Main configuration model."""

    general: GeneralCfg
    dataset: DatasetCfg
    paths: PathsCfg
    model: ModelCfg
    training: Optional[Dict[str, Any]] = Field(default_factory=dict)
    evaluation: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("training", "evaluation", mode="before")
    @classmethod
    def validate_optional_sections(cls, v):
        """Allow training and evaluation to be optional."""
        return v if v is not None else {}

    class Config:
        extra = "forbid"  # Reject unknown top-level keys


def load_and_validate_config(config_path: str | Path) -> Config:
    """Load and validate a YAML configuration file.

    Args:
        config_path: Path to the configuration YAML file

    Returns:
        Validated Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config doesn't match schema
        yaml.YAMLError: If YAML is malformed
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    # Validate using Pydantic
    try:
        config = Config.model_validate(raw_config)
    except Exception as e:
        raise ValueError(f"Configuration validation failed for {config_path}: {e}") from e

    return config


def load_config_dict(config_path: str | Path) -> Dict[str, Any]:
    """Load config as dict (for backward compatibility).

    This function loads and validates the config, then returns it as a dict.
    Use this when you need the raw dict format for existing code.

    Args:
        config_path: Path to the configuration YAML file

    Returns:
        Validated configuration as a dictionary
    """
    config = load_and_validate_config(config_path)
    return config.model_dump()
