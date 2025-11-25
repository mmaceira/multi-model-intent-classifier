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
PROMPT_SYSTEM = """You are an intent classifier for user utterances.

CRITICAL RULES:
1. You MUST respond with ONLY a valid JSON object containing a 'labels' array
2. You MUST use EXACT labels from the provided list - NO exceptions
3. Do NOT create new labels, do NOT use synonyms, do NOT modify labels, do NOT paraphrase
4. Labels are case-sensitive and must match EXACTLY (including underscores, hyphens, and spelling)
5. The 'labels' array MUST contain EXACTLY the same number of labels as there are utterances
6. DO NOT use code blocks, markdown formatting, or explanations
7. Return ONLY the JSON object - nothing before or after it
8. If an utterance doesn't match any label exactly, choose the CLOSEST matching label from the list

Example format: {"labels": ["transfer_money", "check_balance", "translate"]}
Remember: Return ONLY the JSON, no explanations, no markdown, no extra text."""
