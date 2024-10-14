import logging
from openai import AsyncOpenAI
from models import AnalyzedSearchPoints, SearchResultAnalysis
from typing import Callable
from models import ResearchStatus, ResearchProgress

logger = logging.getLogger(__name__)

async def search_result_analyzer(search_result: str, user_input: str, sub_question: str, client: AsyncOpenAI) -> AnalyzedSearchPoints:
    logger.info(f"Starting search result analysis for sub-question: '{sub_question}'")
    logger.info(f"User input: '{user_input}'")
    logger.info(f"Search result length: {len(search_result)} characters")

    try:
        logger.info("Preparing OpenAI API request")
        system_content = f"""
        You are an expert market researcher and consultant specializing in data analysis and insight extraction.
        Your task is to analyze search results related to the specific sub-question: {sub_question} keeping in mind the context of the main query: {user_input}.
        A previous AI agent has generated the search results and data for the analysis.
        You will receive the entire dump of this data.
        Your responsibilities:

        1. Thoroughly review the provided search result data.
        2. Extract all potentially relevant information related to the sub-question: {sub_question} and the main query: {user_input}.
        3. Extrapolate insights and connections from the data, even if not explicitly stated.
        4. Condense the information into clear, concise points.
        5. For each point, provide a reference to its source
        6. In case of deductions or extrapolations of info please explain the approach and thought process behind coming up with that deduction
        7. Make sure that the reference numbers are added inline in the same way that they are ordered in the references section
        8. The references should strictly be the actual URL of the source and not the name or anything

        Prioritize depth and breadth of analysis. Extrapolate required info that answers the sub-question or main query, drawing connections where possible. Do not invent information, but interpret and analyze deeply.
        Your output should be a comprehensive summary of all potentially relevant information found in the search results, including extrapolated insights, with each point clearly linked to its source and labeled appropriately.
        """
        user_content = f"Main query: {user_input} \nSub-question: {sub_question} \nHere is the search result chunk: {search_result}"

        logger.info(f"System content length: {len(system_content)} characters")
        logger.info(f"User content length: {len(user_content)} characters")

        logger.info("Sending request to OpenAI API")
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            response_format=SearchResultAnalysis
        )

        logger.info("Successfully received response from OpenAI API")
        logger.info(f"Raw API response: {response}")

        result = AnalyzedSearchPoints(points=response.choices[0].message.parsed.points)
        logger.info(f"Analyzed {len(result.points)} points for sub-question: '{sub_question}'")
        
        for idx, point in enumerate(result.points, 1):
            logger.info(f"Point {idx}: {point.point[:100]}... (Reference: {point.reference})")

        return result
    except Exception as e:
        logger.error(f"Error analyzing search results: {str(e)}", exc_info=True)
        logger.info(f"Error details: {type(e).__name__}, {str(e)}")
        return None
    finally:
        logger.info(f"Completed search result analysis for sub-question: '{sub_question}'")
