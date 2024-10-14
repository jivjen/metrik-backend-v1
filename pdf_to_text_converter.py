import requests
import os
import fitz
import logging
from time import time
import asyncio

logger = logging.getLogger(__name__)

async def convert_to_text(file_url: str) -> str:
    logger.info(f"Starting conversion for file: {file_url}")
    start_time = time()

    async def jina_ai_conversion():
        logger.info(f"Attempting Jina AI conversion for {file_url}")
        try:
            url = f'https://r.jina.ai/{file_url}'
            headers = {
                "Authorization": "Bearer jina_cdfde91597854ce89ef3daed22947239autBdM5UrHeOgwRczhd1JYzs51OH"
            }
            logger.info(f"Sending GET request to Jina AI for {file_url}")
            response = await asyncio.to_thread(requests.get, url, headers=headers)
            if response.status_code == 200:
                text = response.text
                logger.info(f"Successfully converted {file_url} to text using Jina AI")
                logger.info(f"Jina AI conversion result (first 100 chars): {text[:100]}")
                return text
            else:
                logger.error(f"Error converting {file_url} to text using Jina AI. Status code: {response.status_code}")
                return None
        except Exception as e:
            logger.exception(f"Exception during Jina AI conversion for {file_url}: {str(e)}")
            return None

    async def mupdf_conversion():
        logger.info(f"Attempting PyMuPDF conversion for {file_url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            logger.info(f"Downloading file from {file_url}")
            response = await asyncio.to_thread(requests.get, file_url, headers=headers)
            if response.status_code != 200:
                logger.error(f"Error downloading file from {file_url}. Status code: {response.status_code}")
                return ""
            
            file_name = os.path.basename(file_url)
            logger.info(f"Saving downloaded file as {file_name}")
            with open(file_name, 'wb') as f:
                f.write(response.content)

            logger.info(f"Converting {file_name} to text using PyMuPDF")
            doc = fitz.open(file_name)
            text = ""
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text()
                text += page_text
                logger.info(f"Extracted text from page {page_num} (first 50 chars): {page_text[:50]}")
            doc.close()

            logger.info(f"Removing temporary file: {file_name}")
            os.remove(file_name)

            logger.info(f"Successfully converted {file_url} to text using PyMuPDF")
            logger.info(f"PyMuPDF conversion result (first 100 chars): {text[:100]}")
            return text
        except Exception as e:
            logger.exception(f"Exception during PyMuPDF conversion for {file_url}: {str(e)}")
            return ""

    # Try Jina AI first, then fall back to PyMuPDF
    text = await jina_ai_conversion()
    logger.info(f"Jina AI conversion result length: {len(text) if text else 0}")
    if not text or len(text) < 3000:
        logger.warning(f"Jina AI conversion failed or returned insufficient text for {file_url}. Falling back to PyMuPDF.")
        logger.info(f"Jina AI conversion result: {text}")
        text = await mupdf_conversion()

    end_time = time()
    logger.info(f"Conversion completed for {file_url}. Total time: {end_time - start_time:.2f} seconds")
    logger.info(f"Final text length: {len(text)}")
    return text
