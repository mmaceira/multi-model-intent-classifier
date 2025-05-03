"""Build FAISS index & metadata with sensible defaults.

Run **without arguments**:

    python -m src.rag.build_index

Defaults:
* emb_model  : sentence-transformers/all-MiniLM-L6-v2
* source     : reuters (temporal split)
* cutoff     : 1996
* n_classes  : 10
* artifacts  : artifacts/index.faiss, artifacts/meta.jsonl
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm

from .vector_store import VectorStore

DEFAULT_EMB_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SOURCE     = "reuters"
DEFAULT_CUTOFF     = 1996
DEFAULT_N_CLASSES  = 10
DEFAULT_FAISS_PATH = Path("artifacts/index.faiss")
DEFAULT_META_PATH  = Path("artifacts/meta.jsonl")


def _extract_year(text: str) -> int | None:
    m = re.search(r"<DATE>[^<]*?(\d{2})-(\w{3})-(\d{2,4})", text)
    if not m:
        return None
    _, _, yr = m.groups()
    yr = int(yr)
    return yr + 1900 if yr < 100 else yr

def _load_csv(csv_path: str):
    rows = list(csv.DictReader(open(csv_path)))
    return [r["text"] for r in rows], [r["label"] for r in rows], [int(r.get("year", -1)) for r in rows]

def _load_reuters(cutoff: int, n_cls: int | None):
    from src.datasets.dataset import load_data_temporal
    X_train, y_train, *_ = load_data_temporal(cutoff_year=cutoff, n_classes=n_cls)
    years = [_extract_year(txt) or -1 for txt in X_train]
    return X_train, y_train, years


def main():
    p = argparse.ArgumentParser(description="Build FAISS index for RAG")
    p.add_argument("--emb_model", default=DEFAULT_EMB_MODEL)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--train_csv")
    src.add_argument("--source", default=DEFAULT_SOURCE, choices=["reuters"])
    p.add_argument("--cutoff_year", type=int, default=DEFAULT_CUTOFF)
    p.add_argument("--n_classes", type=int, default=DEFAULT_N_CLASSES)
    p.add_argument("--faiss_path", default=str(DEFAULT_FAISS_PATH))
    p.add_argument("--meta_path",  default=str(DEFAULT_META_PATH))
    args = p.parse_args()

    if args.train_csv:
        texts, labels, years = _load_csv(args.train_csv)
    else:
        texts, labels, years = _load_reuters(args.cutoff_year, args.n_classes)

    print(f"Embedding {len(texts)} documents …")
    emb = VectorStore.embed(args.emb_model, texts)

    meta: List[dict] = []
    for i, (t,l,y,v) in enumerate(zip(texts, labels, years, emb)):
        meta.append({"id": i, "label": l, "year": y, "text": t, "vector": v.tolist()})

    VectorStore.build(emb, meta, emb.shape[1], Path(args.faiss_path), Path(args.meta_path))
    print(f"✅ Index saved → {args.faiss_path}\n✅ Meta saved  → {args.meta_path}")

if __name__ == "__main__":
    main()
