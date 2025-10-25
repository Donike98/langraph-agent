"""
Online retrieval using Tavily Search API
Free tier: 1000 requests/month
"""
import logging
import os
from typing import List, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


def retrieve_online(query: str, top_k: int = 5, api_key: str = None) -> List[Dict[str, str]]:
    """
    Retrieve information using Tavily API (free tier: 1000 requests/month)
    
    Args:
        query: Search query
        top_k: Number of results to return
        api_key: Tavily API key (or from TAVILY_API_KEY env var)
    
    Returns:
        List of dicts with 'text', 'url', 'title'
    """
    try:
        from tavily import TavilyClient
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not api_key:
            logger.error("No Tavily API key provided. Set TAVILY_API_KEY environment variable.")
            return []
        
        logger.info(f"Searching Tavily for: {query}")
        
        client = TavilyClient(api_key=api_key)
        
        # Add LangGraph/LangChain context
        enhanced_query = f"LangGraph LangChain {query}"
        
        # Search with Tavily
        response = client.search(
            query=enhanced_query,
            max_results=top_k,
            search_depth="basic",  # "basic" or "advanced"
            include_answer=True,
            include_raw_content=False
        )
        
        docs = []
        
        # Add the AI-generated answer first if available
        if response.get('answer'):
            docs.append({
                'text': response['answer'],
                'url': 'tavily-summary',
                'title': 'AI Summary',
                'source': 'Tavily AI Summary'
            })
        
        # Add search results
        for i, result in enumerate(response.get('results', []), start=1):
            docs.append({
                'text': f"{result.get('title', '')}\n\n{result.get('content', '')}",
                'url': result.get('url', ''),
                'title': result.get('title', ''),
                'source': result.get('url', '')
            })
        
        logger.info(f"Retrieved {len(docs)} results from Tavily")
        return docs
        
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return []

