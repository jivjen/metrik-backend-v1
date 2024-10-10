import logging
from tavily import TavilyClient
from openai import AsyncOpenAI
from search_result_analyser import search_result_analyzer

logger = logging.getLogger(__name__)

async def process_keyword(keyword: str, user_input: str, sub_question: str, openai: AsyncOpenAI):
    tavily_client = TavilyClient(api_key="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")
    try:
        logger.info(f"Processing keyword: {keyword}")
        tavily_result = await tavily_client.get_search_context(keyword)
    except Exception as e:
        logger.error(f"Error getting search context for keyword {keyword}: {str(e)}")
        tavily_result = ""

    tavily_analyzed = await search_result_analyzer(tavily_result, user_input, sub_question, openai)

    return tavily_analyzed
