# Document Classification Pipeline

This repository contains a modular, multi-stage document classification pipeline. Each stage is separated into its own Jupyter notebook, enabling clear documentation, easy debugging, and independent execution. Intermediate artifacts are saved to disk, allowing each step to pick up where the previous one left off.

---

## Repository Structure

```
.
├── data/
│   ├── raw/                     # Raw datasets
│   ├── processed/               # Preprocessed train/test splits
│   └── embeddings/              # Serialized embedding vectors (MiniLM)
├── models/
│   └── num_classes_<N>/         # Saved trained models and vectorizers for N classes
├── results/
│   ├── roc_curves/              # Per-model ROC curve PNGs
│   ├── confusion_matrices/      # Per-model confusion matrix PNGs
│   ├── worst_docs/              # CSVs listing worst-classified documents per class
│   └── test_metrics.json        # Aggregated test metrics (accuracy, F1, AUC)
├── notebooks/
│   ├── 01_DATA_SPLIT.ipynb
│   ├── 02_EXPLORATORY_DATA_ANALYSIS.ipynb
│   ├── 03_MODEL_TRAINING.ipynb
│   ├── 04_MODEL_EVALUATION.ipynb
│   ├── 05_RESULTS_COMPARISON.ipynb
│   └── 06_SEMANTIC_SEARCH_DEMO.ipynb
├── src/
│   ├── utils/
│   │   ├── data_io.py           # Loading/saving datasets
│   │   ├── preprocessing.py     # Tokenization & vectorization
│   │   ├── model_storage.py     # save_model / load_model helpers
│   │   └── evaluation.py        # Metric computation & plotting
│   └── search/
│       ├── build_index.py       # FAISS index builder
│       └── query.py             # Semantic search wrapper
├── environment.yml              # Conda environment specification
└── README.md                    # This file
```

---

## Getting Started

1. **Clone the repo**  
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install dependencies**  
   ```bash
   conda env create -f environment.yml
   conda activate doc-class-env
   ```

3. **Prepare your data**  
   - Place your raw CSV/TXT data under `data/raw/`.  
   - Verify the file naming follows the notebook expectations (e.g., `train.csv`, `test.csv`).

---

## Pipeline Stages

### 1. Data Splitting (`01_DATA_SPLIT.ipynb`)
- Splits raw data into stratified train/test sets.
- Outputs processed datasets under `data/processed/`.

### 2. Exploratory Data Analysis (`02_EXPLORATORY_DATA_ANALYSIS.ipynb`)
- Performs univariate and class-distribution analysis.
- Generates summary plots and tables for stakeholder review.

### 3. Baseline Model Training (`03_MODEL_TRAINING.ipynb`)
- Trains baseline models (TF-IDF + LogReg, TF-IDF + SVM, MiniLM + LogReg, etc.).
- Saves each model and its associated vectorizer to `models/num_classes_<N>/`.

### 4. Model Evaluation (`04_MODEL_EVALUATION.ipynb`)
- Loads all saved models from the previous stage.
- Evaluates on the test set, computing:
  - Accuracy, macro/weighted F1, per-class AUC
  - Confusion matrices
  - ROC curves (one-vs-rest)
  - Listings of worst-classified documents
- Saves artifacts under `results/`.

### 5. Results Comparison (`05_RESULTS_COMPARISON.ipynb`)
- Reads the aggregated metrics (`results/test_metrics.json`).
- Produces a comparison table and bar charts for:
  - Accuracy
  - Macro F1
  - Weighted F1
  - Average AUC
- Outputs visualizations to `results/`.

### 6. Semantic Search Demo (`06_SEMANTIC_SEARCH_DEMO.ipynb`)
- Demonstrates document retrieval using:
  - **SentenceTransformer (MiniLM)** embeddings + FAISS
  - **Optional** Retrieval-Augmented Generation (RAG) with OpenAI API
- Builds or loads an embeddings index at `data/embeddings/`.
- Provides a `search(query, k)` helper for interactive demo.

---

## Usage

1. **End-to-End Run**  
   ```bash
   jupyter nbconvert --to notebook --execute notebooks/01_DATA_SPLIT.ipynb
   jupyter nbconvert --to notebook --execute notebooks/02_EXPLORATORY_DATA_ANALYSIS.ipynb
   jupyter nbconvert --to notebook --execute notebooks/03_MODEL_TRAINING.ipynb
   jupyter nbconvert --to notebook --execute notebooks/04_MODEL_EVALUATION.ipynb
   jupyter nbconvert --to notebook --execute notebooks/05_RESULTS_COMPARISON.ipynb
   jupyter nbconvert --to notebook --execute notebooks/06_SEMANTIC_SEARCH_DEMO.ipynb
   ```

2. **Interactive Jupyter**  
   Launch JupyterLab and run each notebook in order:
   ```bash
   jupyter lab
   ```

---

## Adding New Models

- Update `03_MODEL_TRAINING.ipynb` to include your new model class/pipeline.
- Run the training notebook to save the model.
- Re-run `04_MODEL_EVALUATION.ipynb` to generate evaluation artifacts.

---

## Environment

- Python 3.8+
- Key libraries: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `sentence-transformers`, `faiss-cpu`, `openai` (optional)

---

## Contact

For questions or issues, please open a GitHub issue or contact the maintainer at `you@example.com`.

---

*Generated on* `2025-05-03`
