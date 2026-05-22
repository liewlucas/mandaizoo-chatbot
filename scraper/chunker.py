import re
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

def chunk_content(page_data: dict) -> list[dict]:
    url = page_data["url"]
    title = page_data["title"]
    soup = page_data["soup"]
    
    if not soup:
        return []

    # FAQ Parsing
    if '/faq' in url:
        return _chunk_faq(soup, url, title)
    # Animal Parsing
    elif '/animals-and-zones' in url:
        return _chunk_animals(soup, url, title)
    # Default Parsing
    else:
        return _chunk_general(soup, url, title)

def _chunk_faq(soup, url, title):
    # Hacky way to inject the javascript:void(0) hrefs into the text
    for a in soup.find_all('a', href=True):
        if 'javascript:void(0)' in a['href']:
            a.string = f"[{a.get_text(strip=True)}](javascript:void(0))"
            
    text = soup.get_text(separator='\n', strip=True)
    parts = re.split(r'\[([^\]]+)\]\(javascript:void\(0\)\)', text)
    
    qa_pairs = []
    for i in range(1, len(parts), 2):
        question = parts[i].strip()
        answer = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if question and answer:
            qa_pairs.append({
                "text": f"Q: {question}\nA: {answer}",
                "metadata": {
                    "source_url": url,
                    "page_title": title,
                    "content_type": "faq",
                    "scraped_at": datetime.now(timezone.utc).isoformat()
                }
            })
            
    if not qa_pairs:
        # Fallback to general chunking if no toggles were found
        return _chunk_general(soup, url, title, content_type="faq")
        
    return qa_pairs

def _chunk_animals(soup, url, title):
    chunks = []
    headings = soup.find_all(['h4', 'h3'])
    for heading in headings:
        animal_name = heading.get_text(strip=True)
        content = []
        for sibling in heading.find_next_siblings():
            if sibling.name in ['h2', 'h3', 'h4']:
                break
            content.append(sibling.get_text(separator=' ', strip=True))
            
        description = " ".join(content).strip()
        if animal_name and description and len(description) > 30:
            chunks.append({
                "text": f"{animal_name}\n{description}",
                "metadata": {
                    "source_url": url,
                    "page_title": title,
                    "content_type": "animal",
                    "scraped_at": datetime.now(timezone.utc).isoformat()
                }
            })
            
    if not chunks:
        return _chunk_general(soup, url, title, content_type="animal")
        
    return chunks

def _chunk_general(soup, url, title, content_type="general"):
    chunks = []
    if "ticket" in url:
        content_type = "ticket"
        
    text = soup.get_text(separator='\n\n', strip=True)
    
    def split_text(text, max_len=1000):
        if len(text) <= max_len:
            return [text]
        paragraphs = text.split('\n\n')
        current_chunk = ""
        result = []
        for p in paragraphs:
            if len(current_chunk) + len(p) < max_len:
                current_chunk += p + "\n\n"
            else:
                if current_chunk:
                    result.append(current_chunk.strip())
                current_chunk = p + "\n\n"
        if current_chunk:
            result.append(current_chunk.strip())
        return result

    raw_chunks = split_text(text)
    
    for c in raw_chunks:
        if len(c) > 80:
            chunks.append({
                "text": c,
                "metadata": {
                    "source_url": url,
                    "page_title": title,
                    "content_type": content_type,
                    "scraped_at": datetime.now(timezone.utc).isoformat()
                }
            })
            
    return chunks
