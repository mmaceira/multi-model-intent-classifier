"""Embedding generation utilities."""

import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np
from openai import OpenAI
from src.utils.retry import with_retry

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        max_retries: int = 3
    ):
        """Initialize the embedding generator.
        
        Parameters
        ----------
        api_key : str
            OpenAI API key
        model : str, optional
            Model to use for embeddings, by default "text-embedding-3-small"
        batch_size : int, optional
            Number of texts to process in each batch, by default 100
        max_retries : int, optional
            Maximum number of retry attempts for API calls, by default 3
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
    
    @with_retry(
        max_retries=3,
        initial_delay=1.0,
        max_delay=10.0,
        backoff_factor=2.0,
        logger=logger
    )
    def _get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a batch of texts.
        
        Parameters
        ----------
        texts : List[str]
            List of texts to embed
            
        Returns
        -------
        np.ndarray
            Array of embeddings
        """
        # Ensure texts is a list of strings and clean them
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        elif not isinstance(texts, list):
            texts = [str(texts)]
            
        # Clean and validate texts
        cleaned_texts = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            # Remove any problematic characters and ensure text is not empty
            text = text.strip()
            if not text:
                text = " "  # Use space for empty texts
            cleaned_texts.append(text)
            
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=cleaned_texts
            )
            return np.array([data.embedding for data in response.data])
        except Exception as e:
            logger.error(f"Error in _get_embeddings_batch: {str(e)}")
            logger.error(f"Input texts: {cleaned_texts}")
            raise
    
    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for a list of texts.
        
        Parameters
        ----------
        texts : List[str]
            List of texts to embed
        show_progress : bool, optional
            Whether to show progress information, by default True
            
        Returns
        -------
        np.ndarray
            Array of embeddings
        """
        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_start_time = time.time()
            
            try:
                batch_embeddings = self._get_embeddings_batch(batch)
                embeddings.extend(batch_embeddings)
                
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                
                if show_progress:
                    current_batch = (i // self.batch_size) + 1
                    logger.info(
                        f"Processed batch {current_batch}/{total_batches} "
                        f"({len(batch)} texts) in {batch_duration:.2f}s"
                    )
            
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                raise
        
        return np.array(embeddings) 