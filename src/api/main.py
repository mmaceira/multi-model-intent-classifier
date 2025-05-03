"""
FastAPI microservice exposing the RAG classifier as an HTTP endpoint.
"""
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from src.rag import load_kmajority, load_centroid, load_llm

app = FastAPI(title="Reuters RAG Classifier API")

class Query(BaseModel):
    """
    Request schema for classification.
    """
    doc: str = Field(..., description="Raw text of the document to classify")
    model: Literal["km", "centroid", "llm"] = Field("km", description="Model: 'km', 'centroid', or 'llm'")
    top_k: int = Field(5, ge=1, description="Number of context documents to retrieve")

class Response(BaseModel):
    """
    Response schema for classification.
    """
    label: str

@lru_cache(maxsize=1)
def get_classifiers():
    """
    Lazy-load classifier instances at startup.
    """
    return {
        "km": load_kmajority(top_k=5),
        "centroid": load_centroid(),
        "llm": load_llm(top_k=5),
    }

@app.post("/classify", response_model=Response)
def classify(query: Query):
    """
    Classify a document using the specified RAG model.
    """
    models = get_classifiers()
    if query.model not in models:
        raise HTTPException(status_code=400, detail=f"Unsupported model '{query.model}'")
    clf = models[query.model]
    label = clf.predict([query.doc], top_k=query.top_k)[0]
    return Response(label=label)
