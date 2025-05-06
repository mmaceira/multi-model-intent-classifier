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
- Performance logging and monitoring
- Model caching
- Support for both OpenAI and SentenceTransformer embeddings
- Batch processing capabilities

Classes:
    VectorStore: Main class for vector storage and retrieval
        - Handles document storage and retrieval
        - Manages embeddings and metadata
        - Provides efficient similarity search
        - Supports multiple embedding types

Functions:
    None (Class methods only)

Dependencies:
    faiss: For efficient similarity search
    numpy: For numerical operations
    sentence_transformers: For embedding generation
    json: For metadata serialization
    logging: For performance monitoring
    pathlib: For path handling
    typing: For type hints

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
from typing import List, Sequence, Dict, Any, Optional, Union
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.embeddings import EmbeddingGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Module-level cache for SentenceTransformer models
_CACHED_MODELS = {}


class VectorStore:
    """
    A class for storing and retrieving document embeddings using FAISS.
    
    This class provides functionality for efficient document storage and retrieval
    using vector embeddings. It supports both OpenAI and SentenceTransformer embeddings,
    and includes features for metadata management and performance monitoring.
    
    Attributes:
        index (faiss.Index): FAISS index for vector storage
        embedder (EmbeddingGenerator): Embedding generator instance
        vectors (Dict[str, np.ndarray]): Document embeddings cache
        metadata (Dict[str, Dict]): Document metadata cache
        _meta (List[Dict]): Legacy metadata format for backward compatibility
    """
    
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the vector store.
        
        The constructor supports two initialization modes:
        1. With index and metadata paths:
            - index_path: Path to FAISS index file
            - meta_path: Path to metadata file
        
        2. With API key and model:
            - api_key: OpenAI API key
            - model: Model name for embeddings (default: "text-embedding-3-small")
        
        Args:
            *args: Positional arguments for initialization
            **kwargs: Keyword arguments for initialization
            
        Raises:
            EnvironmentError: If OpenAI API key is not set
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
                api_key=api_key,
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
        """
        Return metadata in the old format for backward compatibility.
        
        Returns:
            List[Dict[str, Any]]: List of metadata dictionaries
        """
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
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add documents to the vector store.
        
        This method generates embeddings for the documents and stores them along
        with their metadata in the vector store.
        
        Args:
            documents (List[str]): List of documents to add
            metadata (Optional[List[Dict[str, Any]]]): List of metadata dictionaries
                for each document
            
        Example:
            >>> store = VectorStore(api_key="your-api-key")
            >>> documents = ["Document 1", "Document 2"]
            >>> metadata = [{"label": "A"}, {"label": "B"}]
            >>> store.add_documents(documents, metadata)
        """
        embeddings = self.embedder.generate_embeddings(documents)
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            self.vectors[doc] = embedding
            if metadata:
                self.metadata[doc] = metadata[i]
    
    def search(self, query: Union[str, np.ndarray], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        This method performs a similarity search using either a text query or
        an embedding vector. It returns the k most similar documents along with
        their scores and metadata.
        
        Args:
            query (Union[str, np.ndarray]): Query text or embedding vector
            k (int): Number of results to return (default: 5)
            
        Returns:
            List[Dict[str, Any]]: List of results with scores and metadata
            
        Raises:
            ValueError: If query is neither a string nor a numpy array
            
        Example:
            >>> store = VectorStore("path/to/index.faiss", "path/to/meta.jsonl")
            >>> results = store.search("query text", k=3)
            >>> for result in results:
            ...     print(f"Text: {result['text']}, Score: {result['score']}")
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
    def build(emb: np.ndarray, meta: List[dict], dim: int, faiss_path: Union[str, Path], meta_path: Union[str, Path]) -> None:
        """
        Build and save a FAISS index with metadata.
        
        This static method creates a new FAISS index from embeddings and saves it
        along with the associated metadata to disk.
        
        Args:
            emb (np.ndarray): Embedding vectors
            meta (List[dict]): Metadata for each embedding
            dim (int): Dimension of embeddings
            faiss_path (Union[str, Path]): Path to save the FAISS index
            meta_path (Union[str, Path]): Path to save the metadata
            
        Example:
            >>> embeddings = np.random.rand(100, 768)
            >>> metadata = [{"text": f"Doc {i}"} for i in range(100)]
            >>> VectorStore.build(embeddings, metadata, 768, "index.faiss", "meta.jsonl")
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
    def embed(model_name: str, docs: Sequence[str], embedder: Optional[Any] = None) -> np.ndarray:
        """
        Generate embeddings for documents using either a provided embedder or SentenceTransformer.
        
        This static method provides a convenient way to generate embeddings for
        documents using either a custom embedder or a SentenceTransformer model.
        
        Args:
            model_name (str): Name of the SentenceTransformer model (only used if embedder is None)
            docs (Sequence[str]): Documents to embed
            embedder (Optional[Any]): Optional custom embedder with encode() method
            
        Returns:
            np.ndarray: Generated embeddings
            
        Example:
            >>> docs = ["Document 1", "Document 2"]
            >>> embeddings = VectorStore.embed("sentence-transformers/all-MiniLM-L6-v2", docs)
        """
        if embedder is not None:
            return embedder.encode(docs)
        
        if model_name not in _CACHED_MODELS:
            _CACHED_MODELS[model_name] = SentenceTransformer(model_name)
        
        return _CACHED_MODELS[model_name].encode(docs)
