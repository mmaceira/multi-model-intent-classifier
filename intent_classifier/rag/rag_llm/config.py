"""Configuration constants and environment setup for RAG LLM classifier."""

import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Disable litellm's proxy-related logging to avoid import errors
# These errors occur because litellm tries to import proxy modules
# even when not using the proxy, and those modules have optional dependencies
os.environ.setdefault(
    "LITELLM_LOG", "ERROR"
)  # Only show errors, not warnings about missing modules
os.environ.setdefault("LITELLM_SUPPRESS_LOGGING", "true")  # Suppress litellm's verbose logging

# System prompt for the LLM classification task
PROMPT_SYSTEM = """You are an intent classifier. Your task is to classify user utterances
into intent categories.

CRITICAL RULES:
1. Count the number of utterances in the user's message
2. Return EXACTLY that many labels in your 'labels' array
3. Use ONLY the exact label names provided - copy them character-by-character
4. Do NOT return utterance text - return ONLY label names
5. The 'labels' array must match the utterance count exactly
6. Return ONLY valid JSON with a 'labels' array - no other text

Example: If there are 3 utterances, return 3 labels: {"labels": ["label1", "label2", "label3"]}
Remember: Count utterances, return that many labels, use exact label names only."""
