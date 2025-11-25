"""
Retrieval-Augmented Generation (RAG) LLM Intent Classifier
==========================================================

A production-ready intent classifier using RAG with LLMs. Combines vector similarity
search with LLM reasoning.

Key Features:
- Rate limiting and batched processing
- Robust error handling with exponential backoff
- Concurrency control for optimal performance

Implementation:
1. Two-Stage Process:
   a) Retrieval: Finds k similar examples from database
   b) Generation: Uses LLM to classify based on examples

2. Database Usage:
   - Uses database to find context examples
   - LLM uses examples to make informed decisions
   - More flexible than centroid or k-NN approaches

3. Production Features:
   - Batched processing for efficiency
   - Rate limiting to prevent throttling
   - Error handling with retries

Advantages:
- Most flexible approach
- Handles complex cases well
- Adapts to new patterns

Disadvantages:
- Expensive (requires LLM API calls)
- Slower than other approaches
- More complex implementation

Example usage:
    ```python
    classifier = RagLLM.load_default()
    results = classifier.predict([
        "Apple's stock rose 2% after strong quarterly earnings.",
        "Manchester United signed a new striker for £80 million."
    ])
    # Returns: ["business", "sports"]
    ```
"""

from .classifier import RagLLM

__all__ = ["RagLLM"]
