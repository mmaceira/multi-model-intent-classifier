"""Helpers to generate run-level README and per-model model cards.

This module is responsible for taking the artefacts produced by a completed run
and emitting human-readable markdown summaries under the canonical layout:

    output/runs/{run_id}/README.md
    output/runs/{run_id}/models/{model_id}/model_card.md

It is intentionally lightweight and avoids additional dependencies (no Jinja2);
templates are plain ``.format`` strings stored under ``docs/templates``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from intent_classifier.utils.config_loader import load_config_with_metadata
from intent_classifier.utils.file_ops import ensure_dir
from intent_classifier.utils.paths import get_repo_root
from intent_classifier.utils.slugify import slugify_model_id


@dataclass(frozen=True)
class AlgorithmInfo:
    """Human-readable description for a model family."""

    display_name: str
    algorithm_family: str
    role: str
    description: str
    text_input: str
    features: str
    hyperparameters: str
    complexity: str
    latency: str
    failure_modes: str


def _slugify_model_name(name: str) -> str:
    """Backward-compatible wrapper around the shared slugify helper.

    This keeps the public behaviour and doc examples stable while delegating
    the actual implementation to ``intent_classifier.utils.slugify``.
    """
    return slugify_model_id(name)


# Small, hand-written registry of model families that appear in this project.
ALGORITHM_REGISTRY: dict[str, AlgorithmInfo] = {
    "naive_bayes": AlgorithmInfo(
        display_name="Naive Bayes",
        algorithm_family="Bag-of-words + Naive Bayes",
        role="Fast lexical baseline for intent classification.",
        description=(
            "A classic multinomial Naive Bayes classifier on sparse TF‑IDF "
            "features. Strong baseline for short texts and cheap to train."
        ),
        text_input="Tokenised utterances.",
        features="TF‑IDF bag-of-words features over the training corpus.",
        hyperparameters="- ngram_range\n- smoothing (alpha)",
        complexity="Linear in number of non‑zero features.",
        latency="Sub‑millisecond per query on CPU.",
        failure_modes=(
            "- Struggles with paraphrases not seen in training.\n"
            "- Sensitive to stopword handling and vocabulary coverage."
        ),
    ),
    "linear_svm": AlgorithmInfo(
        display_name="Linear SVM",
        algorithm_family="Linear SVM on sparse TF‑IDF",
        role="Stronger lexical baseline than Naive Bayes.",
        description=(
            "Linear SVM trained on TF‑IDF features. Captures sharper decision "
            "boundaries than Naive Bayes while remaining efficient."
        ),
        text_input="Tokenised utterances.",
        features="TF‑IDF bag-of-words or n‑gram features.",
        hyperparameters="- C (regularisation)\n- class_weight",
        complexity="Linear in number of non‑zero features at inference.",
        latency="Low single‑digit milliseconds per query on CPU.",
        failure_modes=(
            "- Can overfit on rare intents without proper regularisation.\n"
            "- Still purely lexical; no semantic generalisation."
        ),
    ),
    "tfidf_bigrams_svm": AlgorithmInfo(
        display_name="TF-IDF bigrams + SVM",
        algorithm_family="Linear SVM with uni/bi‑gram TF‑IDF",
        role="Strong sparse baseline that captures short patterns and phrases.",
        description=(
            "Linear SVM on TF‑IDF features including both unigrams and bigrams. "
            "Good trade‑off between accuracy and cost for production baselines."
        ),
        text_input="Tokenised utterances.",
        features="TF‑IDF features over unigrams and bigrams.",
        hyperparameters="- C (regularisation)\n- ngram_range=(1, 2)",
        complexity="Linear in number of non‑zero features at inference.",
        latency="Low, suitable for real‑time APIs.",
        failure_modes=(
            "- Long‑tail intents with few examples can be under‑represented.\n"
            "- Still fails on paraphrases that share few surface n‑grams."
        ),
    ),
    "minilm_logreg": AlgorithmInfo(
        display_name="MiniLM + LogReg",
        algorithm_family="Sentence-transformer embeddings + Logistic Regression",
        role="Semantic baseline using a compact transformer encoder.",
        description=(
            "Encodes utterances with a MiniLM sentence transformer and trains "
            "a logistic regression classifier on top. Captures semantic "
            "similarity beyond exact wording."
        ),
        text_input="Raw utterances (tokenised by the transformer).",
        features="Dense sentence embeddings from MiniLM.",
        hyperparameters="- embedding_model\n- C (regularisation)\n- class_weight",
        complexity="Embedding cost dominates; classification head is cheap.",
        latency="Few to tens of milliseconds per query depending on hardware.",
        failure_modes=(
            "- Can confuse intents with very similar semantics.\n"
            "- Quality depends heavily on the chosen embedding model."
        ),
    ),
    "embedding_logreg_sbert": AlgorithmInfo(
        display_name="Embedding + LogReg (SBERT)",
        algorithm_family="SBERT embeddings + Logistic Regression",
        role="High‑quality semantic baseline using SBERT.",
        description=(
            "Uses SBERT embeddings as input to a logistic regression classifier. "
            "Good balance between accuracy and latency for many production use‑cases."
        ),
        text_input="Raw utterances.",
        features="SBERT sentence embeddings.",
        hyperparameters="- sbert_model\n- C (regularisation)\n- class_weight",
        complexity="Embedding encoding dominates; linear head is negligible.",
        latency="Tens of milliseconds per query on CPU; much faster on GPU.",
        failure_modes=(
            "- May underperform on domain‑specific jargon without fine‑tuning.\n"
            "- Still constrained by training label coverage."
        ),
    ),
    "embedding_logreg_qwen_ollama": AlgorithmInfo(
        display_name="Embedding + LogReg (Qwen_Ollama)",
        algorithm_family="Ollama/Qwen embeddings + Logistic Regression",
        role="Semantic baseline built on Ollama‑hosted Qwen embeddings.",
        description=(
            "Uses embeddings served by an Ollama‑hosted Qwen model with a "
            "logistic regression head. Useful when you already run Ollama in infra."
        ),
        text_input="Raw utterances sent to the Ollama embedding endpoint.",
        features="Dense embeddings returned by the Qwen model.",
        hyperparameters="- embedding model name\n- C (regularisation)",
        complexity="Network + embedding inference dominate cost.",
        latency="Depends on Ollama deployment; typically tens of milliseconds.",
        failure_modes=(
            "- Sensitive to network latency and server load.\n"
            "- Requires careful endpoint configuration and resource allocation."
        ),
    ),
    "rag_centroidnn": AlgorithmInfo(
        display_name="RAG-CentroidNN",
        algorithm_family="RAG over label centroids + nearest-neighbour",
        role="Hybrid baseline combining retrieval with sparse or dense features.",
        description=(
            "Builds centroid representations per intent and classifies new "
            "utterances via nearest‑neighbour search in embedding space."
        ),
        text_input="Raw utterances embedded by the chosen backend.",
        features="Intent centroids plus nearest‑neighbour distances.",
        hyperparameters="- embedding backend\n- top_k\n- distance metric",
        complexity="Index lookup in vector store plus a small aggregation step.",
        latency="Moderate; dominated by vector search.",
        failure_modes=(
            "- Requires enough examples per intent for stable centroids.\n"
            "- Can be brittle when intents are very similar in embedding space."
        ),
    ),
    "rag_kmajority_sbert": AlgorithmInfo(
        display_name="RAG-kMajority (SBERT)",
        algorithm_family="k‑NN over SBERT embeddings",
        role="Non‑parametric baseline using SBERT with majority vote.",
        description=(
            "Retrieves the top‑k nearest training utterances in SBERT space and "
            "predicts labels via majority vote. Simple but often strong."
        ),
        text_input="Raw utterances.",
        features="SBERT embeddings and nearest neighbours.",
        hyperparameters="- k (number of neighbours)\n- distance metric",
        complexity="Vector search with k‑NN.",
        latency="Moderate; depends on index size and hardware.",
        failure_modes=(
            "- Memory footprint grows with dataset size.\n"
            "- May inherit label noise from neighbours."
        ),
    ),
    "rag_kmajority_qwen_ollama": AlgorithmInfo(
        display_name="RAG-kMajority (Qwen_Ollama)",
        algorithm_family="k‑NN over Ollama/Qwen embeddings",
        role="RAG baseline using Ollama‑served embeddings.",
        description=(
            "Same k‑NN majority‑vote strategy as the SBERT variant but powered "
            "by embeddings served from an Ollama Qwen model."
        ),
        text_input="Raw utterances via Ollama embedding endpoint.",
        features="Dense embeddings and nearest neighbours.",
        hyperparameters="- embedding model\n- k\n- distance metric",
        complexity="Network + vector search.",
        latency="Higher than local embeddings; dominated by network and server load.",
        failure_modes=(
            "- Subject to network flakiness and server saturation.\n"
            "- Requires robust timeout and retry strategies in production."
        ),
    ),
    "rag_llm_tfidf_default": AlgorithmInfo(
        display_name="RAG-LLM (TF-IDF, default prompt)",
        algorithm_family="TF‑IDF retrieval + LLM classification",
        role="LLM‑based classifier using sparse TF‑IDF retrieval.",
        description=(
            "Uses TF‑IDF to retrieve relevant training examples and feeds them "
            "to an LLM with a default prompt to decide the intent."
        ),
        text_input="Raw utterances wrapped in an instruction prompt.",
        features="Retrieved support examples and LLM reasoning.",
        hyperparameters="- retrieval top_k\n- prompt template\n- LLM model",
        complexity="LLM call dominates cost; retrieval is cheap.",
        latency="Typically hundreds of milliseconds per query.",
        failure_modes=(
            "- Sensitive to prompt wording and context window limits.\n"
            "- Can hallucinate labels not seen in training if not constrained."
        ),
    ),
    "rag_llm_sbert_default": AlgorithmInfo(
        display_name="RAG-LLM (SBERT embeddings, default prompt)",
        algorithm_family="Dense SBERT retrieval + LLM classification",
        role="Higher‑recall RAG‑LLM variant using SBERT retrieval.",
        description=(
            "Retrieves similar utterances using SBERT embeddings and passes "
            "them with a default prompt to an LLM classifier."
        ),
        text_input="Raw utterances and retrieved neighbours.",
        features="SBERT embeddings for retrieval; LLM for final decision.",
        hyperparameters="- embedding backend\n- top_k\n- prompt template\n- LLM model",
        complexity="Vector search + LLM inference.",
        latency="Higher than pure embedding methods; dominated by LLM.",
        failure_modes=(
            "- Can be brittle if retrieval returns off‑topic neighbours.\n"
            "- Requires careful cost/latency budgeting."
        ),
    ),
    "rag_llm_qwen_default": AlgorithmInfo(
        display_name="RAG-LLM (Qwen embeddings, default prompt)",
        algorithm_family="Dense Qwen embeddings + LLM classification",
        role="End‑to‑end RAG‑LLM built on Qwen embeddings.",
        description=(
            "Uses Qwen embeddings for retrieval, then delegates intent "
            "classification to an LLM using a default prompt template."
        ),
        text_input="Raw utterances.",
        features="Qwen embeddings and retrieved support examples.",
        hyperparameters="- embedding backend\n- top_k\n- prompt\n- LLM model",
        complexity="Vector search + LLM inference.",
        latency="Similar to other RAG‑LLM variants; dominated by LLM calls.",
        failure_modes=(
            "- Sensitive to both embedding quality and LLM behaviour.\n"
            "- Requires monitoring for cost, latency, and failure rates."
        ),
    ),
    "rag_llm_tfidf_short": AlgorithmInfo(
        display_name="RAG-LLM (TF-IDF, short prompt)",
        algorithm_family="TF‑IDF retrieval + short LLM prompt",
        role="Cheaper / faster RAG‑LLM variant using a compact prompt.",
        description=(
            "Same TF‑IDF retrieval as the default variant but with a shorter "
            "prompt to reduce tokens and latency."
        ),
        text_input="Raw utterances with a concise instruction prompt.",
        features="Retrieved examples and compressed instructions.",
        hyperparameters="- top_k\n- short prompt template\n- LLM model",
        complexity="LLM inference with fewer tokens.",
        latency="Lower than verbose prompts but still LLM‑bound.",
        failure_modes=(
            "- Short prompts can omit important constraints.\n"
            "- May reduce classification robustness on ambiguous intents."
        ),
    ),
    "rag_llm_tfidf_n8n": AlgorithmInfo(
        display_name="RAG-LLM (TF-IDF, n8n prompt)",
        algorithm_family="TF‑IDF retrieval + workflow‑oriented prompt",
        role="RAG‑LLM tuned for n8n / workflow integration examples.",
        description=(
            "Uses TF‑IDF retrieval with a specialised n8n‑style prompt, "
            "optimised for workflow / automation scenarios."
        ),
        text_input="Utterances describing workflow actions.",
        features="Retrieved task examples and workflow‑oriented instructions.",
        hyperparameters="- top_k\n- n8n prompt template\n- LLM model",
        complexity="LLM inference plus light retrieval.",
        latency="Comparable to other RAG‑LLM variants.",
        failure_modes=(
            "- May overfit to workflow phrasing seen in prompt examples.\n"
            "- Needs careful evaluation on non‑workflow datasets."
        ),
    ),
}


def _load_template(template_name: str) -> str:
    """Load a markdown template from ``docs/templates``."""
    repo_root = get_repo_root()
    template_path = repo_root / "docs" / "templates" / f"{template_name}.md.j2"
    return template_path.read_text(encoding="utf-8")


class _SafeDict(dict[str, str]):
    """Dictionary that returns an empty string for missing keys when formatting."""

    def __missing__(self, key: str) -> str:
        return ""


def _build_models_section(results: Mapping[str, Mapping[str, Any]]) -> str:
    """Return a markdown bullet list describing all models in the run."""
    lines: list[str] = []
    for display_name in sorted(results.keys()):
        model_id = _slugify_model_name(display_name)
        lines.append(f"- **{display_name}** → `models/{model_id}/model_card.md`")
    return "\n".join(lines) if lines else "_No models were evaluated in this run._"


def _build_metrics_table(summary_df: pd.DataFrame, max_models: int = 12) -> str:
    """Build a compact markdown table from ``summary_metrics.csv``."""
    if summary_df.empty:
        return "_No summary metrics available yet._"

    # Heuristic: prefer a small set of well-known metrics if present.
    preferred_cols = [
        "test_accuracy",
        "test_macro_f1",
        "test_micro_f1",
        "test_weighted_f1",
    ]
    cols = [c for c in preferred_cols if c in summary_df.columns]
    if not cols:
        # Fallback to the first few numeric columns.
        numeric_cols = [c for c in summary_df.columns if summary_df[c].dtype != "O"]
        cols = numeric_cols[:4]

    df = summary_df[cols].copy()
    df = df.iloc[:max_models]

    header = "| Model | " + " | ".join(cols) + " |"
    separator = "| --- " + " | ".join(["| ---"] * len(cols)) + " |"

    rows: list[str] = [header, separator]
    for model_name, row in df.iterrows():
        values = []
        for c in cols:
            val = row[c]
            if isinstance(val, (int, float)):
                values.append(f"{val:.3f}")
            else:
                values.append(str(val))
        rows.append("| " + str(model_name) + " | " + " | ".join(values) + " |")

    return "\n".join(rows)


def generate_run_readme_and_model_cards(
    results: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Generate README and per-model model cards for the current run.

    Parameters
    ----------
    results:
        Optional mapping from display model name to its scalar metric
        dictionary as returned by
        :func:`intent_classifier.evaluation.run_evaluations`. When omitted
        (or empty), this function will derive the available models from the
        on-disk ``compare/summary_metrics.*`` artefacts.
    """
    config_meta = load_config_with_metadata()
    cfg = config_meta["config"]
    dataset_name = config_meta["dataset_name"]
    config_name = config_meta["config_name"]
    label_type = config_meta["label_type"]
    config_path = config_meta["config_path"]

    repo_root = get_repo_root()
    paths_cfg = cfg.get("paths", {})
    run_id = paths_cfg.get("run_dir", "").replace("output/runs/", "") or cfg.get(
        "resolved", {}
    ).get("run_id", "")

    # Canonical run directories
    run_dir = repo_root / paths_cfg.get("run_dir", f"output/runs/{run_id}")
    dataset_dir = repo_root / paths_cfg.get("dataset_dir", run_dir / "dataset")
    features_dir = repo_root / paths_cfg.get("features_dir", run_dir / "features")
    models_dir = repo_root / paths_cfg.get("models_dir", run_dir / "models")
    eval_dir = repo_root / paths_cfg.get("eval_dir", run_dir / "eval")
    compare_dir = repo_root / paths_cfg.get("compare_dir", run_dir / "compare")

    ensure_dir(run_dir)
    ensure_dir(models_dir)
    ensure_dir(compare_dir)
    ensure_dir(eval_dir)
    ensure_dir(features_dir)
    ensure_dir(dataset_dir)

    # -------------------------- Dataset summary -------------------------------
    summary_path = Path(dataset_dir) / "dataset_summary.csv"
    dataset_stats: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary_df = pd.read_csv(summary_path)
            if not summary_df.empty:
                row = summary_df.iloc[0].to_dict()
                dataset_stats = row
        except Exception:
            dataset_stats = {}

    # Fallbacks if we couldn't read summary.
    train_samples = int(dataset_stats.get("train_samples", 0))
    test_samples = int(dataset_stats.get("test_samples", 0))
    total_classes = int(dataset_stats.get("total_classes", 0))
    train_classes = int(dataset_stats.get("train_classes", 0))
    test_classes = int(dataset_stats.get("test_classes", 0))
    classes_in_both = int(dataset_stats.get("classes_in_both", 0))
    multilabel = bool(dataset_stats.get("multilabel", label_type == "multilabel"))

    # -------------------------- Metrics summary -------------------------------
    summary_df = pd.DataFrame()
    # Prefer CSV but fall back to Parquet if needed.
    summary_csv = Path(compare_dir) / "summary_metrics.csv"
    summary_parquet = Path(compare_dir) / "summary_metrics.parquet"
    if summary_csv.exists():
        try:
            summary_df = pd.read_csv(summary_csv, index_col=0)
        except Exception:
            summary_df = pd.DataFrame()
    elif summary_parquet.exists():
        try:
            summary_df = pd.read_parquet(summary_parquet)
        except Exception:
            summary_df = pd.DataFrame()

    metrics_table = _build_metrics_table(summary_df)

    # -------------------------- Config snapshot -------------------------------
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    resolved = cfg.get("resolved", {}) if isinstance(cfg.get("resolved"), dict) else {}

    embedding_backend = resolved.get("embedding_backend", model_cfg.get("embedding_backend", ""))
    llm_backend = resolved.get("llm_backend", model_cfg.get("llm_backend", ""))
    rag_top_k = model_cfg.get("rag_top_k", "")

    # Prompt style lives under model or prompts depending on config layout.
    prompt_style = ""
    if "prompt_style" in model_cfg:
        prompt_style = str(model_cfg.get("prompt_style") or "")
    elif "prompts" in cfg and isinstance(cfg["prompts"], dict):
        prompt_style = str(cfg["prompts"].get("style", ""))

    # -------------------------- Run README ------------------------------------
    run_template = _load_template("run_readme")

    # If results were not provided, infer a minimal mapping from the summary
    # metrics index. This keeps the function usable from ``finalize_run``
    # without re-running evaluation in memory.
    if not results:
        inferred: dict[str, dict[str, Any]] = {}
        if not summary_df.empty:
            for model_name in summary_df.index:
                inferred[str(model_name)] = {}
        results = inferred

    models_section = _build_models_section(results)

    run_context = _SafeDict(
        run_id=run_id,
        dataset_name=dataset_name,
        variant_name=config_name,
        label_type=label_type,
        multilabel=str(multilabel),
        train_samples=str(train_samples),
        test_samples=str(test_samples),
        total_classes=str(total_classes),
        train_classes=str(train_classes),
        test_classes=str(test_classes),
        classes_in_both=str(classes_in_both),
        models_section=models_section,
        metrics_table=metrics_table,
        embedding_backend=str(embedding_backend),
        llm_backend=str(llm_backend),
        rag_top_k=str(rag_top_k),
        prompt_style=prompt_style,
    )

    readme_text = run_template.format_map(run_context)
    (run_dir / "README.md").write_text(readme_text, encoding="utf-8")

    # -------------------------- Model cards -----------------------------------
    model_template = _load_template("model_card")

    for display_name, model_metrics in results.items():
        model_id = _slugify_model_name(display_name)

        # If metrics are missing from results, try to load from
        # eval/<model_id>/test/test_metrics.json or from compare/summary_metrics.csv
        # Convert to dict to allow modifications (Mapping is read-only)
        metrics_dict: dict[str, Any] = dict(model_metrics) if model_metrics else {}

        if not metrics_dict:
            # Try loading from per-model eval directory
            model_eval_dir = eval_dir / model_id / "test"
            test_metrics_file = model_eval_dir / "test_metrics.json"
            if test_metrics_file.exists():
                try:
                    with open(test_metrics_file, encoding="utf-8") as f:
                        loaded_metrics = json.load(f)
                        # Convert to the expected format (flat dict with test_ prefix)
                        for key, value in loaded_metrics.items():
                            if not key.startswith("test_"):
                                metrics_dict[f"test_{key}"] = value
                            else:
                                metrics_dict[key] = value
                except Exception:
                    pass

            # Fallback: try to get from summary_metrics.csv
            if not metrics_dict and not summary_df.empty:
                if display_name in summary_df.index:
                    row_series = summary_df.loc[display_name]
                    for col in summary_df.columns:
                        val = row_series[col]
                        if pd.notna(val):
                            metrics_dict[col] = float(val)
                # Also try slugified name
                elif model_id in summary_df.index:
                    row_series = summary_df.loc[model_id]
                    for col in summary_df.columns:
                        val = row_series[col]
                        if pd.notna(val):
                            metrics_dict[col] = float(val)

        model_metrics = metrics_dict

        # Try fuzzy matching for RagLLM variants
        registry_key = model_id
        if model_id not in ALGORITHM_REGISTRY:
            # Try fuzzy matching for RagLLM variants
            if model_id.startswith("rag_llm_"):
                # Try to match patterns like rag_llm_tfidf_* -> rag_llm_tfidf_default
                parts = model_id.split("_")
                if len(parts) >= 3 and parts[0] == "rag" and parts[1] == "llm":
                    # Try common variants - match first 3 parts (rag_llm_<retrieval_method>)
                    retrieval_method = parts[2] if len(parts) > 2 else None
                    if retrieval_method:
                        # Try to find a matching variant based on retrieval method
                        base_variants = [
                            "rag_llm_tfidf_default",
                            "rag_llm_sbert_default",
                            "rag_llm_qwen_default",
                            "rag_llm_tfidf_short",
                            "rag_llm_tfidf_n8n",
                        ]
                        for variant in base_variants:
                            variant_parts = variant.split("_")
                            if len(variant_parts) >= 3 and variant_parts[2] == retrieval_method:
                                registry_key = variant
                                break

        algo_info = ALGORITHM_REGISTRY.get(
            registry_key,
            AlgorithmInfo(
                display_name=display_name,
                algorithm_family="Unknown / custom",
                role="Custom model in this run.",
                description="No detailed description available yet.",
                text_input="Raw utterances.",
                features="Project-specific features.",
                hyperparameters="Not documented.",
                complexity="Not documented.",
                latency="Not documented.",
                failure_modes="Failure modes not documented yet.",
            ),
        )

        # Metrics snippet: focus on a handful of standard scores if present.
        def _fmt_metric(key: str, metrics: Mapping[str, Any] = model_metrics) -> str:
            val = metrics.get(key)
            if isinstance(val, (int, float)):
                return f"{val:.3f}"
            return "" if val is None else str(val)

        metric_lines = [
            f"- **Accuracy (test)**: {_fmt_metric('test_accuracy')}",
            f"- **Macro F1 (test)**: {_fmt_metric('test_macro_f1')}",
            f"- **Micro F1 (test)**: {_fmt_metric('test_micro_f1')}",
            f"- **Weighted F1 (test)**: {_fmt_metric('test_weighted_f1')}",
        ]
        metrics_snippet = "\n".join(metric_lines)

        card_context = _SafeDict(
            model_display_name=algo_info.display_name,
            model_id=model_id,
            algorithm_family=algo_info.algorithm_family,
            role=algo_info.role,
            description=algo_info.description,
            text_input=algo_info.text_input,
            features=algo_info.features,
            hyperparameters=algo_info.hyperparameters,
            complexity=algo_info.complexity,
            latency=algo_info.latency,
            metrics_snippet=metrics_snippet,
            failure_modes=algo_info.failure_modes,
            run_id=run_id,
            config_path=str(config_path),
            dataset_name=dataset_name,
        )

        card_text = model_template.format_map(card_context)
        model_dir = ensure_dir(models_dir / model_id)
        (model_dir / "model_card.md").write_text(card_text, encoding="utf-8")


__all__ = [
    "generate_run_readme_and_model_cards",
    "_slugify_model_name",
    "ALGORITHM_REGISTRY",
]
