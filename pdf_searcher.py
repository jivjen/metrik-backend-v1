from googleapiclient.discovery import build
import logging
from typing import Callable
from models import ResearchStatus, ResearchProgress
import os

logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

google_search = build("customsearch", "v1", developerKey=GOOGLE_API_KEY).cse()

async def search_for_pdf_files(keywords: list[str], update_status: Callable, max_results: int = 30, max_attempts: int = 10):
    logger.info(f"Starting PDF search with {len(keywords)} keywords, max_results={max_results}, max_attempts={max_attempts}")
    update_status(ResearchProgress(
        total_steps=5,
        current_step=3,
        status=ResearchStatus.SEARCHING_DOCUMENTS,
        details=f"Starting PDF search with {len(keywords)} keywords"
    ))
    logger.info(f"Keywords: {keywords}")
    pdf_links = []
    attempts = 0
    for idx, keyword in enumerate(keywords, 1):
        if attempts >= max_attempts or len(pdf_links) >= max_results:
            break
        try:
            logger.info(f"Searching for keyword: '{keyword}'")
            update_status(ResearchProgress(
                total_steps=5,
                current_step=3,
                status=ResearchStatus.DOCUMENT_SEARCH_KEYWORD,
                details=f"Searching PDFs for keyword {keyword}"
            ))
            results = google_search.list(q=keyword, cx=GOOGLE_CSE_ID).execute()
            logger.info(f"Received {len(results.get('items', []))} items for keyword: '{keyword}'")
            for item in results.get('items', []):
                if attempts >= max_attempts or len(pdf_links) >= max_results:
                    break
                file_url = item['link']
                if file_url.lower().endswith('.pdf'):
                    logger.info(f"Found PDF: {file_url}")
                    pdf_links.append(file_url)
                attempts += 1
            update_status(ResearchProgress(
                total_steps=5,
                current_step=3,
                status=ResearchStatus.DOCUMENT_SEARCH_KEYWORD_COMPLETED,
                details=f"Completed search for keyword {keyword}" 
            ))
        except Exception as e:
            logger.error(f"Error occurred while searching for keyword '{keyword}': {e}", exc_info=True)
            attempts += 1
    
    logger.info(f"PDF search completed. Found {len(pdf_links)} PDF links")
    update_status(ResearchProgress(
        total_steps=5,
        current_step=3,
        status=ResearchStatus.DOCUMENT_SEARCH_KEYWORD_COMPLETED,
        details=f"PDF search completed. Found {len(pdf_links)} PDF links"
    ))
    logger.info(f"PDF links found: {pdf_links}")
    return pdf_links
