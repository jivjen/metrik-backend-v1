import logging
from tavily import TavilyClient
from openai import AsyncOpenAI
from search_result_analyser import search_result_analyzer
from pymongo import MongoClient
import os
import asyncio
from dotenv import load_dotenv
from typing import Callable
from models import ResearchStatus, ResearchProgress

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# MongoDB connection
client = MongoClient(os.getenv("MONGODB_URI"))
db = client.research_jobs
logger.info("MongoDB connection established")

async def get_tavily_api_key():
    logger.info("Retrieving Tavily API key")
    last_used_key = db.tavily_keys.find_one({"_id": "last_used_key"})
    current_key_num = int(last_used_key["key_num"]) if last_used_key else 1
    logger.info(f"Current key number: {current_key_num}")
    
    api_key = os.getenv(f"TAVILY_API_KEY_{current_key_num}")
    if not api_key:
        logger.error(f"TAVILY_API_KEY_{current_key_num} not found in environment variables")
        raise ValueError(f"TAVILY_API_KEY_{current_key_num} not found in environment variables")
    
    logger.info(f"Retrieved Tavily API key (number {current_key_num})")
    return api_key, current_key_num

async def update_tavily_api_key(current_key_num):
    logger.info(f"Updating Tavily API key from number {current_key_num}")
    next_key_num = (current_key_num % 4) + 1  # Assuming we have 4 API keys
    logger.info(f"Next key number: {next_key_num}")
    db.tavily_keys.update_one({"_id": "last_used_key"}, {"$set": {"key_num": str(next_key_num)}}, upsert=True)
    logger.info(f"Updated Tavily API key to number {next_key_num}")

async def process_keyword(keyword: str, user_input: str, sub_question: str, openai: AsyncOpenAI, update_status: Callable):
    logger.info(f"Processing keyword: '{keyword}' for sub-question: '{sub_question}'")
    update_status(ResearchProgress(
        total_steps=5,
        current_step=3,
        status=ResearchStatus.PROCESSING_KEYWORDS,
        details=f"Processing keyword: {keyword}"
    ))
    max_retries = 4  # Maximum number of API keys to try
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1} of {max_retries}")
            update_status(ResearchProgress(
                total_steps=5,
                current_step=3,
                status=ResearchStatus.SEARCHING_ONLINE,
                details=f"Searching Online for keyword: {keyword} (Attempt {attempt + 1})"
            ))
            api_key, current_key_num = await get_tavily_api_key()
            tavily_client = TavilyClient(api_key=api_key)
            tavily_result = await asyncio.to_thread(tavily_client.get_search_context, keyword)
            logger.info(f"Received Tavily search result (first 100 chars): {tavily_result[:100]}...")
            update_status(ResearchProgress(
                total_steps=5,
                current_step=3,
                status=ResearchStatus.SEARCH_COMPLETED,
                details=f"Completed search for keyword: {keyword}"
            ))
            break
        except Exception as e:
            logger.error(f"Error in Tavily search for keyword '{keyword}': {str(e)}")
            await update_tavily_api_key(current_key_num)
            if attempt == max_retries - 1:
                logger.warning(f"All API keys exhausted. Unable to process keyword: '{keyword}'")
                tavily_result = ""

    logger.info(f"Analyzing Tavily search result for keyword: '{keyword}'")
    update_status(ResearchProgress(
        total_steps=5,
        current_step=3,
        status=ResearchStatus.ANALYZING_SEARCH_RESULT,
        details=f"Analyzing search result for keyword: {keyword}"
    ))
    tavily_analyzed = await search_result_analyzer(tavily_result, user_input, sub_question, openai)
    logger.info(f"Analyzed result for keyword '{keyword}' (first 100 chars): {str(tavily_analyzed)[:100]}...")
    update_status(ResearchProgress(
        total_steps=5,
        current_step=3,
        status=ResearchStatus.KEYWORD_ANALYSIS_COMPLETED,
        details=f"Completed analysis for keyword: {keyword}"
    ))

    logger.info(f"Completed processing for keyword: '{keyword}'")
    return tavily_analyzed
