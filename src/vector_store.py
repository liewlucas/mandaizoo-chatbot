import chromadb
from src.config import CHROMA_DB_PATH
from src.embeddings import get_embeddings, get_query_embedding
import logging

logger = logging.getLogger(__name__)

class MandaiVectorStore:
    def __init__(self):
        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name="mandai_zoo",
            metadata={"hnsw:space": "cosine"}
        )

    def upsert_chunks(self, chunks: list[dict]):
        """Upsert a list of chunks into ChromaDB."""
        if not chunks:
            return
            
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Generate stable IDs based on URL and text hash
        import hashlib
        ids = []
        for chunk in chunks:
            text_hash = hashlib.md5(chunk["text"].encode('utf-8')).hexdigest()[:8]
            url_hash = hashlib.md5(chunk["metadata"]["source_url"].encode('utf-8')).hexdigest()[:8]
            ids.append(f"{url_hash}_{text_hash}")
            
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = get_embeddings(texts, task_type="RETRIEVAL_DOCUMENT")
        
        logger.info(f"Upserting {len(texts)} chunks to ChromaDB...")
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        logger.info("Upsert complete.")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the vector store for the most relevant chunks."""
        query_embedding = get_query_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results and results['distances'] else 0.0
                })
                
        return formatted_results
