import asyncio
import logging
from urllib.parse import quote_plus
import aiohttp
from aiohttp import ClientSession

logger = logging.getLogger(__name__)

GOOGLE_API_KEY = "AIzaSyBiTmP3mKXTUb13BtpDivIDZ5X5KccFaqU"
GOOGLE_CSE_ID = "82236a47a9b6e47e6"

async def search_for_pdf_files(keywords: list[str], max_results: int = 30, max_attempts: int = 10):
    logger.info(f"Starting PDF search with {len(keywords)} keywords, max_results={max_results}, max_attempts={max_attempts}")
    logger.info(f"Keywords: {keywords}")
    pdf_links = []
    attempts = 0
    
    async def search_keyword(session, keyword):
        nonlocal attempts
        if attempts >= max_attempts or len(pdf_links) >= max_results:
            logger.info(f"Skipping search for keyword '{keyword}' due to reaching limits")
            return
        try:
            encoded_keyword = quote_plus(keyword)
            url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}&q={encoded_keyword}"
            logger.info(f"Sending request for keyword: '{keyword}'")
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Received {len(data.get('items', []))} items for keyword: '{keyword}'")
                    for item in data.get('items', []):
                        if attempts >= max_attempts or len(pdf_links) >= max_results:
                            logger.info("Reached max attempts or results limit")
                            return
                        file_url = item['link']
                        if file_url.lower().endswith('.pdf'):
                            logger.info(f"Found PDF: {file_url}")
                            pdf_links.append(file_url)
                        attempts += 1
                else:
                    logger.error(f"Error occurred while searching for keyword '{keyword}': HTTP {response.status}")
        except Exception as e:
            logger.error(f"Error occurred while searching for keyword '{keyword}': {e}", exc_info=True)
        logger.info(f"Completed search for keyword '{keyword}'. Total attempts: {attempts}")

    async with ClientSession() as session:
        logger.info("Starting concurrent keyword searches")
        tasks = [search_keyword(session, keyword) for keyword in keywords]
        await asyncio.gather(*tasks)
    
    logger.info(f"PDF search completed. Found {len(pdf_links)} PDF links")
    logger.info(f"PDF links found: {pdf_links}")
    return pdf_links
