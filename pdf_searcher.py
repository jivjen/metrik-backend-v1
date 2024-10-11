import asyncio
import logging
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

GOOGLE_API_KEY= "AIzaSyBiTmP3mKXTUb13BtpDivIDZ5X5KccFaqU"
GOOGLE_CSE_ID= "82236a47a9b6e47e6"

google_search = build("customsearch", "v1", developerKey=GOOGLE_API_KEY).cse()

async def search_for_pdf_files(keywords: list[str], max_results: int = 30, max_attempts: int = 10):
    pdf_links = []
    attempts = 0
    
    async def search_keyword(keyword):
        nonlocal attempts
        if attempts >= max_attempts or len(pdf_links) >= max_results:
            return
        try:
            results = await asyncio.to_thread(google_search.list(q=keyword, cx=GOOGLE_CSE_ID).execute)
            for item in results.get('items', []):
                if attempts >= max_attempts or len(pdf_links) >= max_results:
                    break
                file_url = item['link']
                if file_url.lower().endswith('.pdf'):
                    pdf_links.append(file_url)
                attempts += 1
        except Exception as e:
            logger.error(f"Error occurred while searching for keyword '{keyword}': {e}")
            attempts += 1

    await asyncio.gather(*[search_keyword(keyword) for keyword in keywords])
    logger.info(f"Found {len(pdf_links)} PDF links")
    return pdf_links
