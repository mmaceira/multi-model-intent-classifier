## Algorithms

This document gives a high‑level overview of the main models in this project and how they behave. It focuses on **what each model does**, **how it represents text**, and **when you would typically choose it**, without claiming specific performance numbers.

## Overview

The project includes:

- **Multinomial Naive Bayes (`nb_tfidf`)**: classic bag‑of‑words baseline.
- **Linear SVM (`svm_linear`, `svm_bigram`)**: stronger linear classifier on TF‑IDF features.
- **MiniLM + Logistic Regression (`transformer_logreg`)**: transformer embeddings + simple classifier.
- **Embedding + Logistic Regression (`embedding_logreg`)**: same idea as above, but with pluggable embedding backends.
- **RAG‑based models (`rag_faiss`, `rag_llm`, centroid / k‑NN variants)**: retrieval‑augmented, context‑aware classification.

## 1. Multinomial Naive Bayes (`nb_tfidf`)

**Core idea**: represent each document as a sparse TF‑IDF vector and assume that word occurrences are conditionally independent given the class. The model learns, for each label, which words are comparatively more or less likely.

**How it models text**
- Uses a **bag‑of‑words** representation: only word counts and frequencies matter, not order.
- TF‑IDF down‑weights very common words and up‑weights words that are distinctive for particular documents or classes.
- Each class is associated with a **probability distribution over words**, estimated with smoothing to avoid zero probabilities.

**Behavior and trade‑offs**
- Tends to work well when classes are characterized by a few **strong, label‑specific keywords**.
- Can struggle when distinguishing relies on **word order, subtle phrasing, or long‑range context**.
- Very simple to train and interpret; often used as a **baseline model** or when resources are limited.

## 2. Linear SVM (`svm_linear`, `svm_bigram`)

**Core idea**: still uses high‑dimensional TF‑IDF features, but instead of modeling word probabilities, it learns a **linear decision boundary** that separates classes in feature space.

**How it models text**
- Uses TF‑IDF on **uni‑grams** (single words) and, in the `svm_bigram` variant, **bi‑grams** (two‑word sequences).
- Bi‑grams help capture short patterns like "stock market", "not good", or "interest rates" that Naive Bayes and uni‑grams alone may blur.
- The SVM assigns **weights to each feature** (word or n‑gram). The sign and magnitude of those weights reflect how strongly that feature pushes predictions toward or away from a class.

**Behavior and trade‑offs**
- Usually more robust than Naive Bayes when the data is high‑dimensional and not strictly separable by independent word counts.
- The linear decision boundary makes it relatively **interpretable**: you can inspect top‑weighted features per class.
- Still relies on sparse, surface‑level features; it does not capture deeper semantics or synonymy on its own.

## 3. MiniLM + Logistic Regression (`transformer_logreg`)

**Core idea**: use a transformer encoder (MiniLM) to produce **dense semantic embeddings** for each document, then train a logistic regression classifier on top of these embeddings.

**How it models text**
- The transformer reads the full text with attention, allowing it to encode **word order, context, and long‑range dependencies**.
- Each document is mapped to a **low‑dimensional dense vector** where semantically similar documents have similar embeddings, even if they share few exact words.
- Logistic regression then learns **linear decision surfaces in embedding space**, effectively separating clusters of documents by label.

**Behavior and trade‑offs**
- Typically stronger when labels depend on **semantic intent**, paraphrases, or subtle phrasing rather than exact keywords.
- More robust to vocabulary shifts (e.g., synonyms, rephrasings) because the transformer maps similar meanings close together.
- Requires more compute than purely TF‑IDF models, but the classifier itself remains simple and easy to analyze via feature weights on embedding dimensions.

## 4. Embedding + Logistic Regression (`embedding_logreg`)

**Core idea**: keep the same simple classifier (logistic regression) but allow **different embedding backends** so you can swap how text is encoded without changing the downstream model.

**How it models text**
- The pipeline is: text → embedding model → dense vector → logistic regression.
- Embeddings can come from:
  - A **local SBERT model** (e.g., `sentence-transformers/all-MiniLM-L6-v2`), running on your own hardware.
  - An **API‑based embedding model** (e.g., OpenAI), called remotely.
- The classifier only sees the final embedding, so you can experiment with different encoders while keeping the training and prediction interface stable.

**Behavior and trade‑offs**
- Lets you trade off **latency, cost, and privacy** by choosing local vs. remote embeddings.
- As with MiniLM + LogReg, the model benefits from **semantic similarity**: documents with similar meaning cluster together regardless of exact words.
- A good choice when you want to keep a **simple, well‑understood classifier**, but improve or change the underlying language representation.

## 5. RAG‑based Classification (`rag_faiss`, `rag_llm`, centroid / k‑NN)

**Core idea**: instead of classifying each document in isolation, RAG models **retrieve similar examples** and optionally use an LLM to make a decision **conditioned on those neighbors**. This combines semantic search with classification.

**Common components**
- **Embedding model**: converts each document into a dense vector representation.
- **FAISS index or similar retrieval backend**: supports fast nearest‑neighbor search over all embedded documents.
- **Retrieved context**: for a query document, the system fetches the most similar labeled examples and uses them as context for classification.

**Main variants**
- **k‑NN / k‑majority**: classify a document by looking at the labels of its nearest neighbors and aggregating them (e.g., majority vote or distance‑weighted vote).
- **Centroid‑based**: compute a centroid (average embedding) per class and assign the label whose centroid is closest to the query embedding.
- **RAG‑LLM (`rag_llm`)**: feed the query text plus retrieved examples to an LLM, which outputs a label (and optionally an explanation) based on both the document and its neighbors.

**Behavior and trade‑offs**
- Naturally supports **few‑shot and evolving label behavior**: adding new labeled examples to the index can immediately influence predictions without fully retraining a parametric model.
- Particularly useful when you want **explanations** grounded in real examples (you can show which neighbors were used) or when context from similar documents is crucial.
- RAG‑LLM variants add the ability to reason over retrieved context in natural language but depend on LLM availability and configuration.

## Where to find the implementations

- Algorithms live in `intent_classifier/algorithms/`:
  - `naive_bayes.py`: Multinomial Naive Bayes on TF‑IDF.
  - `linear_svm.py`: Linear SVM on uni‑gram / bi‑gram TF‑IDF.
  - `transformer_logreg.py`: MiniLM embeddings + logistic regression.
  - `embedding_logreg.py`: Flexible embedding backends + logistic regression.

- RAG components live in `intent_classifier/rag/`:
  - `rag_kmajority.py`: k‑NN / k‑majority style classification on retrieved neighbors.
  - `centroid_nn.py`: centroid‑based nearest‑neighbor classifier.
  - `rag_llm/`: LLM‑based RAG classifier.
  - `adapter_sklearn.py`: sklearn‑compatible adapter for RAG models.
