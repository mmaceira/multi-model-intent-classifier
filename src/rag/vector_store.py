"""
Vector Store Module

This module provides a vector store implementation for the RAG (Retrieval-Augmented
Generation) system. It handles the storage and retrieval of document embeddings
using FAISS (Facebook AI Similarity Search) and manages associated metadata.

Key Features:
- FAISS-based vector storage and retrieval
- Efficient nearest neighbor search
- Document embedding generation
- Metadata management
- Performance logging
- Model caching

Classes:
- VectorStore: Main class for vector storage and retrieval

Functions:
- None (Class methods only)

Dependencies:
- faiss
- numpy
- sentence_transformers
- json
- logging
- pathlib
- typing

Example Usage:
    >>> # Initialize vector store
    >>> store = VectorStore("path/to/index.faiss", "path/to/meta.jsonl")
    
    >>> # Search for similar documents
    >>> results = store.search(query_embedding, k=5)
    
    >>> # Generate embeddings
    >>> embeddings = VectorStore.embed("sentence-transformers/all-MiniLM-L6-v2", documents)
"""

import json
import time
import logging
import os
from pathlib import Path
from typing import List, Sequence, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.embeddings import EmbeddingGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Module-level cache for SentenceTransformer models
_CACHED_MODELS = {}

class VectorStore:
    def __init__(self, *args, **kwargs):
        """Initialize the vector store.
        
        Parameters
        ----------
        Either:
            api_key : str
                OpenAI API key
            model : str, optional
                Model to use for embeddings, by default "text-embedding-3-small"
        Or:
            index_path : str | Path
                Path to the FAISS index file
            meta_path : str | Path
                Path to the metadata file
        """
        if len(args) == 2 and isinstance(args[0], (str, Path)) and isinstance(args[1], (str, Path)):
            # Initialize with index and metadata paths
            index_path = Path(args[0])
            meta_path = Path(args[1])
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            self._meta = []
            with open(meta_path, 'r') as f:
                for line in f:
                    self._meta.append(json.loads(line))
            
            # Initialize embedder for search functionality
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY not set in environment")
                
            self.embedder = EmbeddingGenerator(
                api_key=api_key,  # Use the actual API key from environment
                model="text-embedding-3-small",
                batch_size=100,
                max_retries=3
            )
            self.vectors = {}
            self.metadata = {}
        else:
            # Initialize with API key and model
            api_key = args[0] if args else kwargs.get('api_key')
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise EnvironmentError("OPENAI_API_KEY not set in environment")
            
            self.embedder = EmbeddingGenerator(
                api_key=api_key,
                model=kwargs.get('model', "text-embedding-3-small"),
                batch_size=100,
                max_retries=3
            )
            self.vectors = {}
            self.metadata = {}
            self._meta = []
    
    @property
    def meta(self) -> List[Dict[str, Any]]:
        """Return metadata in the old format for backward compatibility."""
        if hasattr(self, '_meta') and self._meta:
            return self._meta
        return [
            {
                'label': meta.get('label', ''),
                'text': meta.get('text', ''),
                'vector': self.vectors.get(doc, []).tolist() if doc in self.vectors else []
            }
            for doc, meta in self.metadata.items()
        ]
    
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents to the vector store.
        
        Parameters
        ----------
        documents : List[str]
            List of documents to add
        metadata : List[Dict[str, Any]], optional
            List of metadata dictionaries for each document, by default None
        """
        embeddings = self.embedder.generate_embeddings(documents)
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            self.vectors[doc] = embedding
            if metadata:
                self.metadata[doc] = metadata[i]
    
    def search(self, query: str | np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Parameters
        ----------
        query : str | np.ndarray
            Query text or embedding vector
        k : int, optional
            Number of results to return, by default 5
            
        Returns
        -------
        List[Dict[str, Any]]
            List of results with scores and metadata
        """
        # Handle both string queries and embedding vectors
        if isinstance(query, str):
            query_embedding = self.embedder.generate_embeddings([query])[0]
        elif isinstance(query, np.ndarray):
            query_embedding = query
        else:
            raise ValueError(f"Query must be a string or numpy array, got {type(query)}")
        
        # Normalize the query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search using FAISS
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k
        )
        
        # Convert to list of results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:  # FAISS returns -1 for invalid indices
                result = {
                    "text": self._meta[idx].get('text', ''),
                    "label": self._meta[idx].get('label', ''),
                    "score": float(distances[0][i])
                }
                results.append(result)
        
        return results

    @staticmethod
    def build(emb: np.ndarray, meta: List[dict], dim: int, faiss_path, meta_path):
        """Build and save a FAISS index with metadata.
        
        Args:
            emb: Embedding vectors
            meta: Metadata for each embedding
            dim: Dimension of embeddings
            faiss_path: Path to save the FAISS index
            meta_path: Path to save the metadata
        """
        # Ensure paths are Path objects
        faiss_path = Path(faiss_path) if not isinstance(faiss_path, Path) else faiss_path
        meta_path = Path(meta_path) if not isinstance(meta_path, Path) else meta_path
        
        start_time = time.time()
        logger.info(f"Building FAISS index with {len(meta)} documents of dimension {dim}")
        
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Creating IndexFlatIP...")
        index = faiss.IndexFlatIP(dim)
        
        logger.info("Normalizing embeddings...")
        faiss.normalize_L2(emb)
        
        logger.info("Adding embeddings to index...")
        index_add_start = time.time()
        index.add(emb.astype('float32'))
        logger.info(f"Added embeddings to index in {time.time() - index_add_start:.2f} seconds")
        
        logger.info(f"Writing index to {faiss_path}...")
        write_start = time.time()
        faiss.write_index(index, str(faiss_path))
        logger.info(f"Wrote index in {time.time() - write_start:.2f} seconds")
        
        logger.info(f"Writing metadata to {meta_path}...")
        meta_start = time.time()
        meta_path.write_text('\n'.join(json.dumps(m) for m in meta))
        logger.info(f"Wrote metadata in {time.time() - meta_start:.2f} seconds")
        
        logger.info(f"Index built in {time.time() - start_time:.2f} seconds")

    @staticmethod
    def embed(model_name: str, docs: Sequence[str], embedder=None) -> np.ndarray:
        """Generate embeddings for documents using either a provided embedder or SentenceTransformer.
        
        Args:
            model_name: Name of the SentenceTransformer model (only used if embedder is None)
            docs: Documents to embed
            embedder: Optional custom embedder with encode() method
            
        Returns:
            Numpy array of embeddings
        """
        start_time = time.time()
        logger.info(f"Generating embeddings for {len(docs)} documents")
        
        # Import here to avoid circular imports
        try:
            from . import _EMBEDDINGS_DIR
            logger.info(f"Embeddings will be stored in: {_EMBEDDINGS_DIR}")
        except ImportError:
            logger.warning("Could not import _EMBEDDINGS_DIR from rag module")
        
        # If embedder is provided, use it directly
        if embedder is not None:
            if hasattr(embedder, "encode"):
                logger.info(f"Using provided embedder: {embedder.__class__.__name__}")
                embeddings = embedder.encode(list(docs))
                
                total_time = time.time() - start_time
                logger.info(f"Generated {len(docs)} embeddings with provided embedder in {total_time:.2f} seconds")
                
                return np.array(embeddings, dtype="float32")
        
        # Otherwise, use cached SentenceTransformer or create a new one
        logger.info(f"Using SentenceTransformer: {model_name}")
        
        # Check if model is in cache
        if model_name in _CACHED_MODELS:
            logger.info(f"Using cached SentenceTransformer model: {model_name}")
            model = _CACHED_MODELS[model_name]
            model_load_time = 0
        else:
            # Load model
            model_load_start = time.time()
            logger.info(f"Use pytorch device_name: cpu")
            logger.info(f"Load pretrained SentenceTransformer: {model_name}")
            model = SentenceTransformer(model_name)
            model_load_time = time.time() - model_load_start
            logger.info(f"Loaded new SentenceTransformer model in {model_load_time:.2f} seconds")
            
            # Cache the model
            _CACHED_MODELS[model_name] = model
        
        # Generate embeddings
        encode_start = time.time()
        embeddings = model.encode(list(docs), batch_size=32, show_progress_bar=False).astype('float32')
        encode_time = time.time() - encode_start
        
        total_time = time.time() - start_time
        logger.info(f"Generated {len(docs)} embeddings (shape: {embeddings.shape}) in {total_time:.2f} seconds")
        logger.info(f"Encoding rate: {len(docs)/encode_time:.1f} docs/second")
        
        return embeddings
