import logging
from tavily import TavilyClient
from openai import AsyncOpenAI
from search_result_analyser import search_result_analyzer
from pymongo import MongoClient
import os

logger = logging.getLogger(__name__)

# MongoDB connection
client = MongoClient("mongodb+srv://hireloom_admin:QuickVogue%40123@quickvogue.x163n.mongodb.net/?retryWrites=true&w=majority&appName=quickvogue")
db = client.research_jobs

async def get_next_tavily_api_key():
    last_used_key = db.tavily_keys.find_one({"_id": "last_used_key"})
    if last_used_key:
        current_key_num = int(last_used_key["key_num"])
        next_key_num = (current_key_num % 4) + 1  # Assuming we have 5 API keys
    else:
        next_key_num = 1
    
    new_key = os.environ.get(f"TAVILY_API_KEY_{next_key_num}")
    if not new_key:
        raise ValueError(f"TAVILY_API_KEY_{next_key_num} not found in environment variables")
    
    db.tavily_keys.update_one({"_id": "last_used_key"}, {"$set": {"key_num": str(next_key_num)}}, upsert=True)
    return new_key

async def process_keyword(keyword: str, user_input: str, sub_question: str, openai: AsyncOpenAI):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            api_key = await get_next_tavily_api_key()
            tavily_client = TavilyClient(api_key=api_key)
            logger.info(f"Processing keyword: {keyword}")
            tavily_result = await tavily_client.get_search_context(keyword)
            break
        except Exception as e:
            logger.error(f"Error getting search context for keyword {keyword} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"Max retries reached. Unable to process keyword: {keyword}")
                tavily_result = ""

    tavily_analyzed = await search_result_analyzer(tavily_result, user_input, sub_question, openai)

    return tavily_analyzed
