"""Result processing utilities for analyzing AI responses."""

import re
from typing import List


def check_brand_mentioned(text: str, brand: str) -> bool:
    """Check if a brand is mentioned in the text (case-insensitive).

    Args:
        text: The response text to analyze.
        brand: The brand name to search for.

    Returns:
        True if the brand is mentioned, False otherwise.
    """
    if not text or not brand:
        return False

    # Case-insensitive search using word boundaries for better accuracy
    # This will match "Nike" but not "nike's" becoming a partial match issue
    pattern = re.compile(re.escape(brand), re.IGNORECASE)
    return bool(pattern.search(text))


def extract_competitors_mentioned(text: str, competitors: List[str]) -> List[str]:
    """Extract which competitors are mentioned in the text.

    Args:
        text: The response text to analyze.
        competitors: List of competitor names to search for.

    Returns:
        List of competitor names that were found in the text.
    """
    if not text or not competitors:
        return []

    mentioned = []
    for competitor in competitors:
        if competitor:  # Skip empty strings
            pattern = re.compile(re.escape(competitor), re.IGNORECASE)
            if pattern.search(text):
                mentioned.append(competitor)

    return mentioned


def classify_response_type(text: str) -> str:
    """Classify whether the response is a list or prose.

    Args:
        text: The response text to analyze.

    Returns:
        'list' if the response appears to be a list, 'prose' otherwise.
    """
    if not text:
        return "unknown"

    lines = text.strip().split("\n")

    # Patterns that indicate list items
    list_patterns = [
        r"^\s*\d+[\.\)]\s",       # Numbered: 1. or 1)
        r"^\s*[-*•]\s",           # Bullets: - * •
        r"^\s*[a-zA-Z][\.\)]\s",  # Lettered: a. or a)
        r"^\s*\*\*\d+\.",         # Markdown bold numbered: **1.
    ]

    list_line_count = 0
    for line in lines:
        for pattern in list_patterns:
            if re.match(pattern, line):
                list_line_count += 1
                break

    # If at least 2 lines look like list items, classify as list
    if list_line_count >= 2:
        return "list"

    return "prose"


def estimate_cost(provider: str, tokens_input: int, tokens_output: int) -> float:
    """Estimate the cost of an API call.

    Args:
        provider: The AI provider ('openai' or 'gemini').
        tokens_input: Number of input tokens.
        tokens_output: Number of output tokens.

    Returns:
        Estimated cost in dollars.
    """
    if provider == "openai":
        # GPT-4o pricing: $0.005/1K input, $0.015/1K output
        return (tokens_input * 0.005 / 1000) + (tokens_output * 0.015 / 1000)
    elif provider == "gemini":
        # Gemini Flash pricing: $0.000075/1K input, $0.0003/1K output
        return (tokens_input * 0.000075 / 1000) + (tokens_output * 0.0003 / 1000)
    else:
        return 0.0


def estimate_run_cost(
    num_prompts: int,
    num_providers: int,
    num_temperatures: int,
    num_repeats: int,
    providers: List[str],
) -> float:
    """Estimate total cost for a visibility run.

    Assumes average of ~100 input tokens and ~500 output tokens per call.

    Args:
        num_prompts: Number of prompts.
        num_providers: Number of providers.
        num_temperatures: Number of temperature settings.
        num_repeats: Number of repeats.
        providers: List of provider names.

    Returns:
        Estimated total cost in dollars.
    """
    total_calls = num_prompts * num_providers * num_temperatures * num_repeats
    avg_input_tokens = 100
    avg_output_tokens = 500

    total_cost = 0.0

    # Calculate cost per provider
    calls_per_provider = total_calls // num_providers if num_providers > 0 else 0

    for provider in providers:
        if provider == "openai":
            cost_per_call = estimate_cost("openai", avg_input_tokens, avg_output_tokens)
        elif provider == "gemini":
            cost_per_call = estimate_cost("gemini", avg_input_tokens, avg_output_tokens)
        else:
            cost_per_call = 0.01  # Default estimate

        total_cost += cost_per_call * calls_per_provider

    return round(total_cost, 4)


def estimate_duration_seconds(total_calls: int) -> int:
    """Estimate duration for a run in seconds.

    Assumes ~3 seconds per API call with 10 parallel workers.

    Args:
        total_calls: Total number of API calls.

    Returns:
        Estimated duration in seconds.
    """
    seconds_per_call = 3
    parallel_workers = 10

    # Calculate with parallelization
    batches = (total_calls + parallel_workers - 1) // parallel_workers
    return batches * seconds_per_call
