import requests
import os
import fitz
import logging
from time import time
import asyncio
from typing import Callable
from models import ResearchStatus, ResearchProgress

logger = logging.getLogger(__name__)

async def convert_to_text(file_url: str, update_status: Callable) -> str:
    logger.info(f"Starting conversion for file: {file_url}")
    update_status(ResearchProgress(
        total_steps=5,
        current_step=4,
        status=ResearchStatus.PROCESSING_DOCUMENTS,
        details=f"Converting PDF to text: {os.path.basename(file_url)}"
    ))
    start_time = time()

    async def primary_conversion():
        logger.info(f"Attempting primary conversion for {file_url}")
        update_status(ResearchStatus.PRIMARY_CONVERSION, f"Attempting PDF conversion for {os.path.basename(file_url)}")
        try:
            url = f'https://r.jina.ai/{file_url}'
            headers = {
                "Authorization": "Bearer jina_cdfde91597854ce89ef3daed22947239autBdM5UrHeOgwRczhd1JYzs51OH"
            }
            logger.info(f"Sending GET request for primary conversion of {file_url}")
            response = await asyncio.to_thread(requests.get, url, headers=headers)
            if response.status_code == 200:
                text = response.text
                logger.info(f"Successfully converted {file_url} to text using primary method")
                update_status(ResearchStatus.PRIMARY_CONVERSION_COMPLETED, f"Successfully converted {os.path.basename(file_url)} using primary method")
                logger.info(f"Primary conversion result (first 100 chars): {text[:100]}")
                return text
            else:
                logger.error(f"Error converting {file_url} to text using primary method. Status code: {response.status_code}")
                return None
        except Exception as e:
            logger.exception(f"Exception during primary conversion for {file_url}: {str(e)}")
            return None

    async def secondary_conversion():
        logger.info(f"Attempting secondary conversion for {file_url}")
        update_status(ResearchStatus.SECONDARY_CONVERSION, f"Attempting PDF conversion for {os.path.basename(file_url)}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            logger.info(f"Downloading file from {file_url}")
            update_status(ResearchStatus.DOCUMENT_DOWNLOAD, f"Downloading {os.path.basename(file_url)}")
            response = await asyncio.to_thread(requests.get, file_url, headers=headers)
            if response.status_code != 200:
                update_status(ResearchStatus.DOCUMENT_DOWNLOAD_FAILED, f"Failed to download {os.path.basename(file_url)}")
                return ""
            
            file_name = os.path.basename(file_url)
            logger.info(f"Saving downloaded file as {file_name}")
            with open(file_name, 'wb') as f:
                f.write(response.content)

            try:
                logger.info(f"Converting {file_name} to text using secondary method")
                update_status(ResearchStatus.DOCUMENT_TEXT_EXTRACTION, f"Extracting text from {file_name}")
                doc = fitz.open(file_name)
                text = ""
                for page_num, page in enumerate(doc, 1):
                    page_text = page.get_text()
                    text += page_text
                    logger.info(f"Extracted text from page {page_num} (first 50 chars): {page_text[:50]}")
                    update_status(ResearchStatus.DOCUMENT_PAGE_EXTRACTED, f"Extracted text from page {page_num}/{doc.page_count} of {file_name}")
                doc.close()
            finally:
                logger.info(f"Removing temporary file: {file_name}")
                os.remove(file_name)

            logger.info(f"Successfully converted {file_url} to text using secondary method")
            update_status(ResearchStatus.SECONDARY_CONVERSION_COMPLETED, f"Successfully converted {file_name}")
            logger.info(f"Secondary conversion result (first 100 chars): {text[:100]}")
            return text
        except Exception as e:
            logger.exception(f"Exception during secondary conversion for {file_url}: {str(e)}")
            return ""

    # Try primary conversion first, then fall back to secondary
    text = await primary_conversion()
    logger.info(f"Primary conversion result length: {len(text) if text else 0}")
    if not text or len(text) < 3000:
        logger.warning(f"Primary conversion failed or returned insufficient text for {file_url}. Falling back to secondary method.")
        logger.info(f"Primary conversion result: {text}")
        text = await secondary_conversion()

    end_time = time()
    logger.info(f"Conversion completed for {file_url}. Total time: {end_time - start_time:.2f} seconds")
    logger.info(f"Final text length: {len(text)}")
    update_status(ResearchStatus.PDF_CONVERSION_COMPLETED, f"Completed conversion for {os.path.basename(file_url)}. Text length: {len(text)}")
    return text
