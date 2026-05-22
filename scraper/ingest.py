import argparse
import logging
import os
import sys

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.crawl import fetch_page
from scraper.chunker import chunk_content

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def get_seed_urls():
    seed_urls_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'seed_urls.txt')
    with open(seed_urls_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def dry_run():
    urls = get_seed_urls()
    all_chunks = []
    for url in urls:
        page_data = fetch_page(url)
        chunks = chunk_content(page_data)
        all_chunks.extend(chunks)
        logger.info(f"Generated {len(chunks)} chunks for {url}")
        
    sizes = [len(c["text"]) for c in all_chunks]
    if not sizes:
        logger.error("No chunks generated!")
        return
        
    sizes.sort()
    logger.info("\n--- DRY RUN ANALYSIS ---")
    logger.info(f"Total Chunks: {len(all_chunks)}")
    logger.info(f"Min Size: {sizes[0]}")
    logger.info(f"Max Size: {sizes[-1]}")
    logger.info(f"Median Size: {sizes[len(sizes)//2]}")
    
    logger.info("\n--- SAMPLE CHUNKS ---")
    import random
    samples = random.sample(all_chunks, min(5, len(all_chunks)))
    for s in samples:
        print(f"\n[{s['metadata']['content_type'].upper()}] - {s['metadata']['page_title']}")
        print(s['text'][:300] + ("..." if len(s['text']) > 300 else ""))
        print("-" * 60)

def ingest():
    urls = get_seed_urls()
    all_chunks = []
    for url in urls:
        page_data = fetch_page(url)
        chunks = chunk_content(page_data)
        all_chunks.extend(chunks)
        logger.info(f"Generated {len(chunks)} chunks for {url}")
        
    if not all_chunks:
        logger.error("No chunks to ingest!")
        return
        
    logger.info(f"Total chunks to ingest: {len(all_chunks)}")
    
    from src.vector_store import MandaiVectorStore
    store = MandaiVectorStore()
    store.upsert_chunks(all_chunks)
    logger.info("Ingestion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    if args.dry_run:
        dry_run()
    else:
        ingest()
