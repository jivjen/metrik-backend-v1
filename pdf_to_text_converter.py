import aiohttp
import aiofiles
import os
import fitz
import logging

logger = logging.getLogger(__name__)

async def convert_to_text(file_url: str) -> str:
    async def jina_ai_conversion():
        try:
            url = f'https://r.jina.ai/{file_url}'
            headers = {
                "Authorization": "Bearer jina_cdfde91597854ce89ef3daed22947239autBdM5UrHeOgwRczhd1JYzs51OH"
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        text = await response.text()
                        logger.info(f"Successfully converted {file_url} to text using Jina AI")
                        logger.debug(f"READ PDF + {text[:100]}")
                        return text
                    else:
                        logger.error(f"Error converting {file_url} to text using Jina AI. Status code: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error converting {file_url} to text using Jina AI: {e}")
            return None

    async def mupdf_conversion():
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(file_url, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Error downloading file from {file_url}. Status code: {response.status}")
                        return ""
                    
                    file_name = os.path.basename(file_url)
                    async with aiofiles.open(file_name, mode='wb') as f:
                        await f.write(await response.read())

            # Convert PDF to text using PyMuPDF
            doc = fitz.open(file_name)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            # Remove the temporary file
            os.remove(file_name)

            logger.info(f"Successfully converted {file_url} to text using PyMuPDF")
            logger.debug(f"READ PDF MuPdf + {text[:100]}")
            return text
        except Exception as e:
            logger.error(f"Error converting {file_url} to text using PyMuPDF: {e}")
            return ""

    # Try Jina AI first, then fall back to PyMuPDF
    text = await jina_ai_conversion()
    logger.info(f"length of text = {len(text) if text else 0}")
    if not text or len(text) < 3000:
        logger.info(f"Jina AI text = {text}")
        logger.info("Jina AI conversion failed. Trying PyMuPDF...")
        text = await mupdf_conversion()

    return text
