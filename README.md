
# Reuters RAG Classifier 📚🔍

A modular pipeline that marries traditional text‑classification algorithms with Retrieval‑Augmented Generation (RAG) techniques on the classic **Reuters‑21578** corpus.

![Pipeline Overview](docs/pipeline_overview.svg)

---

##  Quick start

```bash
git clone https://github.com/<you>/reuters-rag-classifier.git
cd reuters-rag-classifier

# 🔧 create environment
conda env create -f environment.yml
conda activate reuters-rag

# 📦 install package in editable mode
pip install -e .

# ⚙️ configure the run
cp config/config_example.yaml config/config.yaml     # edit as needed

# 🚀 run end‑to‑end (CLI)
python -m src.pipeline.train --config config/config.yaml
```

| Stage | Script / Notebook | Output |
|-------|-------------------|--------|
| **00 Embeddings** | `00_Build_Embeddings.ipynb` | FAISS index & `.npy` vectors |
| **01 EDA** | `02_Exploratory_Analysis.ipynb` | Charts & label stats |
| **02 Training** | `03_Model_Training.ipynb` / `src/pipeline/train.py` | Scikit‑learn models |
| **03 Evaluation** | `04_Model_Evaluation_ROC.ipynb` | ROC, confusion matrices |
| **04 Comparison** | `05_Results_Comparison.ipynb` | Markdown summary table |
| **05 Semantic Search** | `06_Semantic_Search_Demo.ipynb` | Interactive QA |

> **Tip 💡** All notebooks read the same `config/config.yaml` so you can switch corpus, embeddings, or number of classes with *one* edit.

### Project structure
```
reuters-rag-classifier
├── config/                  # YAML configs
├── docs/                    # diagrams & rationale
├── notebooks/               # ordered 00‑..‑06
├── src/                     # importable python package
└── experiments/<run_name>/  # auto‑generated artefacts
```

### License
Apache‑2.0
