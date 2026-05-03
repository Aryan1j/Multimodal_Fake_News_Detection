"""
API Check - Optional news headline evidence checking via external APIs.

This module is completely isolated from ML training code.
Failures here never crash training or inference.
"""

import os
from typing import Optional

import requests
from dotenv import load_dotenv

from .config import PROJECT_ROOT


# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")


# Configuration from environment
ENABLE_NEWS_EVIDENCE = os.getenv("ENABLE_NEWS_EVIDENCE", "false").lower() == "true"
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")


def clean_query(text: str, max_words: int = 6) -> str:
    """
    Clean and truncate query for API search.
    
    Args:
        text: Raw input text
        max_words: Maximum words to include (default 6)
        
    Returns:
        Cleaned query string
    """
    # Take first max_words words, lowercase
    words = text.lower().split()[:max_words]
    return ' '.join(words)


def search_gnews(query: str) -> Optional[str]:
    """
    Search GNews API for headlines matching query.
    
    Args:
        query: Search query
        
    Returns:
        Status string or None if no results/error
    """
    if not GNEWS_API_KEY:
        return None
    
    try:
        url = "https://gnews.io/api/v4/search"
        params = {
            "q": query[:100],  # GNews query limit
            "lang": "en",
            "max": 5,
            "apikey": GNEWS_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            
            if articles:
                return f"GNews: Found {len(articles)} related articles"
        else:
            print(f"GNews API returned status {response.status_code}")
        
        return None
        
    except Exception as e:
        print(f"GNews API error: {e}")
        return None


def search_newsapi(query: str) -> Optional[str]:
    """
    Search NewsAPI for headlines matching query.
    
    Args:
        query: Search query
        
    Returns:
        Status string or None if no results/error
    """
    if not NEWS_API_KEY:
        return None
    
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query[:100],  # API query limit
            "language": "en",
            "pageSize": 5,
            "apiKey": NEWS_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            
            if articles:
                return f"NewsAPI: Found {len(articles)} related articles"
        else:
            print(f"NewsAPI returned status {response.status_code}")
        
        return None
        
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return None


def check_news(text: str) -> str:
    """
    Check external news APIs for evidence of the story.
    
    This function is designed to never crash - all errors are caught
    and returned as status strings.
    
    Args:
        text: Input text to search for
        
    Returns:
        Status string describing the evidence check result
    """
    # Check if feature is enabled
    if not ENABLE_NEWS_EVIDENCE:
        return "Evidence checking disabled (set ENABLE_NEWS_EVIDENCE=true)"
    
    # Check if any API keys are configured
    if not GNEWS_API_KEY and not NEWS_API_KEY:
        return "No API keys configured (set GNEWS_API_KEY or NEWS_API_KEY)"
    
    # Clean the query
    query = clean_query(text)
    
    if not query:
        return "Query too short for evidence check"
    
    # Try GNews first
    result = search_gnews(query)
    if result:
        return result
    
    # Fall back to NewsAPI
    result = search_newsapi(query)
    if result:
        return result
    
    # No results from either API
    if GNEWS_API_KEY or NEWS_API_KEY:
        return "No matching news articles found"
    
    return "API unavailable"


# CLI interface for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.api_check 'Your news text here'")
        sys.exit(1)
    
    text = sys.argv[1]
    result = check_news(text)
    print(f"Evidence check: {result}")
