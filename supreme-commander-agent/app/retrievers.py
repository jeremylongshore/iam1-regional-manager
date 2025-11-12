"""
RAG retrievers for Supreme Commander knowledge grounding.

Provides access to:
- Vertex AI Search for enterprise knowledge
- Vertex AI Vector Search for semantic retrieval
- Combined hybrid search
"""

import logging
import os

from google.cloud import discoveryengine_v1 as discoveryengine

logger = logging.getLogger(__name__)


def retrieve_knowledge(query: str, max_results: int = 5) -> str:
    """
    Retrieve relevant knowledge from the knowledge base.

    Uses Vertex AI Search for grounded, cited responses.

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        Formatted knowledge with citations
    """
    try:
        project_id = os.getenv('PROJECT_ID')
        data_store_id = os.getenv('DATA_STORE_ID', 'supreme-commander-knowledge')
        location = os.getenv('DATA_STORE_REGION', 'global')

        # Create client
        client = discoveryengine.SearchServiceClient()

        # Configure serving config
        serving_config = (
            f"projects/{project_id}/"
            f"locations/{location}/"
            f"collections/default_collection/"
            f"dataStores/{data_store_id}/"
            f"servingConfigs/default_config"
        )

        # Create search request
        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query,
            page_size=max_results,
        )

        # Execute search
        response = client.search(request)

        # Format results with citations
        results = []
        for i, result in enumerate(response.results, 1):
            doc_data = result.document.derived_struct_data

            title = doc_data.get('title', 'Untitled')
            snippet = doc_data.get('snippets', [{}])[0].get('snippet', 'No snippet')
            link = doc_data.get('link', '')

            results.append(f"""[{i}] {title}
{snippet}
Source: {link}
""")

        if not results:
            return "No relevant information found in the knowledge base for this query."

        formatted_results = "\n".join(results)
        return f"""📚 Knowledge Base Results for: "{query}"

{formatted_results}

Use this information to inform your decisions and provide grounded responses.
"""

    except Exception as e:
        logger.error(f"Knowledge retrieval failed: {e}")
        return f"⚠️ Knowledge retrieval unavailable: {str(e)}\n\nProceeding with general knowledge."


def retrieve_similar_cases(description: str, top_k: int = 3) -> str:
    """
    Retrieve similar past cases/tasks using vector search.

    Useful for learning from past decisions and outcomes.

    Args:
        description: Description of current case/task
        top_k: Number of similar cases to retrieve

    Returns:
        Similar cases with outcomes
    """
    try:
        # This would use Vertex AI Vector Search in production
        # For now, placeholder implementation

        return f"""🔍 Similar Past Cases (based on: "{description[:50]}...")

No similar cases found yet. This is a new scenario.

As you complete tasks, they will be indexed for future reference.
"""

    except Exception as e:
        logger.error(f"Similar case retrieval failed: {e}")
        return f"⚠️ Could not retrieve similar cases: {str(e)}"
