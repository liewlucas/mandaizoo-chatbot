from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, GEMINI_MODEL, TOP_K
from src.vector_store import MandaiVectorStore
import logging

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """
You are the Mandai Zoo Assistant, a helpful and friendly AI chatbot that answers
questions about Singapore Zoo and the Mandai Wildlife Reserve.

HOW TO ANSWER:

TIER 1 — FULLY COVERED (context contains the answer):
Answer directly using the context. Cite the source. Be confident.

TIER 2 — PARTIALLY COVERED (context has related info but not the exact answer):
Share what IS in the context, then clearly flag the gap. Example:
"Based on the official website, Singapore Zoo is open daily from 8:30 AM
to 6:00 PM. However, I don't have specific info about public holiday hours
— I'd recommend checking https://www.mandai.com or calling +65 6269 3411
for the latest updates."

TIER 3 — ADJACENT / GENERAL KNOWLEDGE (context doesn't cover it, but the
answer is stable public knowledge like transport routes):
You MAY provide helpful general knowledge, but you MUST:
- Clearly label it: "Based on general knowledge (not from the official website)..."
- Keep it brief
- Still point to the official source for confirmation

TIER 4 — NOT COVERED / HIGH-RISK:
For anything you're unsure about, or for data that changes frequently
(prices, promotions, event dates), DO NOT guess. Say:
"I don't have that specific information. Please check the official website
at https://www.mandai.com for the latest details, or contact the Mandai
helpline at +65 6269 3411."

HARD RULES:
- NEVER invent animal names, exhibit names, or attractions that aren't in context.
- NEVER state specific prices without them being in the context. If prices ARE
  in context, always add: "Prices are subject to change — please verify at
  https://www.mandai.com."
- NEVER make up opening hours or event schedules.
- If the user asks about something completely outside Mandai/zoo scope,
  politely redirect them.
- Always be conversational and friendly — imagine you're a helpful guide at
  the zoo entrance.
- Keep responses concise but complete. Use bullet points for lists.
- When citing context, mention which source it comes from.
"""

class RAGPipeline:
    def __init__(self):
        self.vector_store = MandaiVectorStore()
        if GEMINI_API_KEY:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        else:
            self.client = None
            
    def answer_query(self, user_query: str) -> str:
        if not self.client:
            return "Error: Gemini API key is not configured."
            
        # 1. Retrieve context
        logger.info(f"Retrieving context for: {user_query}")
        results = self.vector_store.search(user_query, top_k=TOP_K)
        
        # 2. Format context
        context_parts = []
        source_urls = set()
        
        for res in results:
            title = res['metadata'].get('page_title', 'Unknown')
            url = res['metadata'].get('source_url', 'Unknown')
            text = res['text']
            
            context_parts.append(f"[Source: {title} | URL: {url}]\n{text}\n---")
            if url != 'Unknown':
                source_urls.add(url)
                
        context_str = "\n\n".join(context_parts)
        sources_str = "\n".join(f"- {url}" for url in source_urls)
        
        # 3. Construct prompt
        prompt = f"""
Context (retrieved from mandai.com):
---
{context_str}
---

Sources:
{sources_str}

User Question: {user_query}

Instructions:
- Answer using the context above when possible (Tier 1-2).
- If the context only partially covers the question, share what you have
  and clearly flag what's missing.
- For general/stable knowledge not in context (e.g., MRT routes), you may
  help briefly but label it as general knowledge (Tier 3).
- Never guess at prices, hours, or event details not in context (Tier 4).
- Cite sources when referencing context.
"""
        
        # 4. Call Gemini
        logger.info("Calling Gemini API...")
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2, # Keep it grounded
            )
        )
        
        return response.text
