from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, EMBEDDING_MODEL

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None

def get_embeddings(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Get embeddings for a list of texts using the new google-genai SDK."""
    if not client:
        raise ValueError("GEMINI_API_KEY is missing. Cannot get embeddings.")
        
    embeddings = []
    # Google GenAI embed_content supports a list of strings
    # But let's process them in batches to be safe and handle API rate limits
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type
            )
        )
        # response.embeddings is a list of EmbedContentResponse objects, which have `values`
        for emb in response.embeddings:
            embeddings.append(emb.values)
            
    return embeddings

def get_query_embedding(query: str) -> list[float]:
    """Get embedding for a single search query."""
    result = get_embeddings([query], task_type="RETRIEVAL_QUERY")
    return result[0]
