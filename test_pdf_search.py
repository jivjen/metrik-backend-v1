import asyncio
import logging
import sys
import asyncio
import logging
import sys
from pdf_searcher import search_for_pdf_files

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    keywords = ["artificial intelligence", "machine learning"]
    logger.info(f"Starting PDF search with keywords: {keywords}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Asyncio event loop: {asyncio.get_event_loop().__class__.__name__}")
    
    try:
        logger.info("Calling search_for_pdf_files function")
        pdf_links = await search_for_pdf_files(keywords, max_results=5)
        
        logger.info(f"Found {len(pdf_links)} PDF links:")
        for link in pdf_links:
            logger.info(link)
    except Exception as e:
        logger.exception(f"An error occurred during PDF search: {e}")

if __name__ == "__main__":
    logger.info("Starting the main function")
    asyncio.run(main())
    logger.info("Main function completed")
