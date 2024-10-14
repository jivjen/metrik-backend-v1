import asyncio
import logging
from pdf_searcher import search_for_pdf_files

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    keywords = ["artificial intelligence", "machine learning"]
    logger.info(f"Starting PDF search with keywords: {keywords}")
    
    pdf_links = await search_for_pdf_files(keywords, max_results=5)
    
    logger.info(f"Found {len(pdf_links)} PDF links:")
    for link in pdf_links:
        logger.info(link)

if __name__ == "__main__":
    asyncio.run(main())
