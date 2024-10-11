import asyncio
import logging
import aiohttp
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

GOOGLE_API_KEY = "AIzaSyBiTmP3mKXTUb13BtpDivIDZ5X5KccFaqU"
GOOGLE_CSE_ID = "82236a47a9b6e47e6"

async def search_for_pdf_files(keywords: list[str], max_results: int = 30, max_attempts: int = 10):
    pdf_links = []
    attempts = 0
    
    async def search_keyword(session, keyword):
        nonlocal attempts
        if attempts >= max_attempts or len(pdf_links) >= max_results:
            return
        try:
            encoded_keyword = quote_plus(keyword)
            url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}&q={encoded_keyword}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    for item in data.get('items', []):
                        if attempts >= max_attempts or len(pdf_links) >= max_results:
                            break
                        file_url = item['link']
                        if file_url.lower().endswith('.pdf'):
                            pdf_links.append(file_url)
                        attempts += 1
                else:
                    logger.error(f"Error occurred while searching for keyword '{keyword}': HTTP {response.status}")
        except Exception as e:
            logger.error(f"Error occurred while searching for keyword '{keyword}': {e}")
        attempts += 1

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[search_keyword(session, keyword) for keyword in keywords])
    
    logger.info(f"Found {len(pdf_links)} PDF links")
    return pdf_links
