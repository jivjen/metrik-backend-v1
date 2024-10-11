import logging
from tavily import TavilyClient
from openai import AsyncOpenAI
from search_result_analyser import search_result_analyzer
from pymongo import MongoClient
import os

logger = logging.getLogger(__name__)

def setup_logger():
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler('keyword_processor.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

setup_logger()

# MongoDB connection
client = MongoClient("mongodb+srv://hireloom_admin:QuickVogue%40123@quickvogue.x163n.mongodb.net/?retryWrites=true&w=majority&appName=quickvogue")
db = client.research_jobs
logger.info("MongoDB connection established")

async def get_tavily_api_key():
    logger.info("Retrieving Tavily API key")
    last_used_key = db.tavily_keys.find_one({"_id": "last_used_key"})
    current_key_num = int(last_used_key["key_num"]) if last_used_key else 1
    logger.debug(f"Current key number: {current_key_num}")
    
    api_key = os.environ.get(f"TAVILY_API_KEY_{current_key_num}")
    if not api_key:
        logger.error(f"TAVILY_API_KEY_{current_key_num} not found in environment variables")
        raise ValueError(f"TAVILY_API_KEY_{current_key_num} not found in environment variables")
    
    logger.info(f"Retrieved Tavily API key (number {current_key_num})")
    return api_key, current_key_num

async def update_tavily_api_key(current_key_num):
    logger.info(f"Updating Tavily API key from number {current_key_num}")
    next_key_num = (current_key_num % 4) + 1  # Assuming we have 4 API keys
    logger.debug(f"Next key number: {next_key_num}")
    db.tavily_keys.update_one({"_id": "last_used_key"}, {"$set": {"key_num": str(next_key_num)}}, upsert=True)
    logger.info(f"Updated Tavily API key to number {next_key_num}")

async def process_keyword(keyword: str, user_input: str, sub_question: str, openai: AsyncOpenAI):
    max_retries = 4  # Maximum number of API keys to try
    for attempt in range(max_retries):
        try:
            api_key, current_key_num = await get_tavily_api_key()
            tavily_client = TavilyClient(api_key=api_key)
            logger.info(f"Processing keyword: {keyword}")
            tavily_result = await tavily_client.get_search_context(keyword)
            break
        except Exception as e:
            logger.error(f"Error getting search context for keyword {keyword} with API key {current_key_num}: {str(e)}")
            await update_tavily_api_key(current_key_num)
            if attempt == max_retries - 1:
                logger.error(f"All API keys exhausted. Unable to process keyword: {keyword}")
                tavily_result = ""

    tavily_analyzed = await search_result_analyzer(tavily_result, user_input, sub_question, openai)

    return tavily_analyzed
