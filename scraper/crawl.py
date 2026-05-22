import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

def fetch_page(url: str) -> dict:
    """Fetches a URL and returns the cleaned HTML soup, title, and metadata."""
    logger.info(f"Fetching {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    title = soup.title.string.strip() if soup.title else "Unknown Title"
    
    # Remove noise elements
    for tag in soup.select('nav, footer, script, style, .cookie-banner, header, aside, .mega-menu'):
        tag.decompose()
        
    main = soup.select_one('main') or soup.select_one('[role="main"]') or soup.body
    
    return {
        "url": url,
        "title": title,
        "soup": main
    }
