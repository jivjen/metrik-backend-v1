from tavily import TavilyClient
from openai import OpenAI
from search_result_analyser import search_result_analyzer

def process_keyword(user_input: str, sub_question: str, openai: OpenAI):
    tavily_client = TavilyClient(api_key="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")
    try:
        tavily_result = tavily_client.get_search_context()
    except Exception as e:
        tavily_result = ""

    tavily_analyzed = search_result_analyzer(tavily_result, user_input, sub_question, openai)

    return tavily_analyzed