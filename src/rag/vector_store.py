"""\
Vector Store module.

Classes:
- VectorStore

Functions:
- None

Created: 2025-05-03
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Sequence
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Module-level cache for SentenceTransformer models
_CACHED_MODELS = {}

class VectorStore:
    def __init__(self, faiss_path, meta_path):
        """Initialize VectorStore by loading FAISS index and metadata.
        
        Args:
            faiss_path: Path to FAISS index file
            meta_path: Path to metadata JSONL file
        """
        start_time = time.time()
        
        # Ensure paths are Path objects
        self.index_path = Path(faiss_path) if not isinstance(faiss_path, Path) else faiss_path
        self.meta_path = Path(meta_path) if not isinstance(meta_path, Path) else meta_path
        
        logger.info(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        
        logger.info(f"Loading metadata from {self.meta_path}")
        self.meta: List[dict] = [json.loads(l) for l in Path(self.meta_path).read_text().splitlines()]
        
        logger.info(f"VectorStore initialized with {len(self.meta)} documents in {time.time() - start_time:.2f} seconds")

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

    def search(self, q: np.ndarray, k: int) -> List[dict]:
        """Search the index for nearest neighbors.
        
        Args:
            q: Query vector
            k: Number of nearest neighbors to return
            
        Returns:
            List of metadata entries for nearest neighbors
        """
        start_time = time.time()
        logger.debug(f"Starting vector search for top-{k} results")
        
        normalize_start = time.time()
        faiss.normalize_L2(q)
        normalize_time = time.time() - normalize_start
        
        search_start = time.time()
        _, idx = self.index.search(q.astype('float32'), k)
        search_time = time.time() - search_start
        
        gather_start = time.time()
        results = [self.meta[i] for i in idx[0]]
        gather_time = time.time() - gather_start
        
        total_time = time.time() - start_time
        logger.debug(f"Vector search completed in {total_time:.4f}s (normalize: {normalize_time:.4f}s, search: {search_time:.4f}s, gather: {gather_time:.4f}s)")
        
        return results

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
