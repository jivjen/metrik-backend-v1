from googleapiclient.discovery import build
import logging

logger = logging.getLogger(__name__)

GOOGLE_API_KEY = "AIzaSyBiTmP3mKXTUb13BtpDivIDZ5X5KccFaqU"
GOOGLE_CSE_ID = "82236a47a9b6e47e6"

google_search = build("customsearch", "v1", developerKey=GOOGLE_API_KEY).cse()

def search_for_pdf_files(keywords: list[str], max_results: int = 30, max_attempts: int = 10):
    logger.info(f"Starting PDF search with {len(keywords)} keywords, max_results={max_results}, max_attempts={max_attempts}")
    logger.info(f"Keywords: {keywords}")
    pdf_links = []
    attempts = 0
    for keyword in keywords:
        if attempts >= max_attempts or len(pdf_links) >= max_results:
            break
        try:
            logger.info(f"Searching for keyword: '{keyword}'")
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
        except Exception as e:
            logger.error(f"Error occurred while searching for keyword '{keyword}': {e}", exc_info=True)
            attempts += 1
    
    logger.info(f"PDF search completed. Found {len(pdf_links)} PDF links")
    logger.info(f"PDF links found: {pdf_links}")
    return pdf_links
