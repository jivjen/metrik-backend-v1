import logging
from openai.types.chat import ParsedChatCompletion
from openai import AsyncOpenAI
from models import KeywordGeneration

logger = logging.getLogger(__name__)

async def keyword_generator(user_input: str, sub_question: str, client: AsyncOpenAI) -> ParsedChatCompletion[KeywordGeneration]:
    try:
        logger.info(f"Starting keyword generation for main query: '{user_input}' and sub-question: '{sub_question}'")
        logger.debug(f"Preparing OpenAI API request for keyword generation")
        system_content = """
        You are a professional Google search researcher.
        Given a main user query for context and a specific sub-question, your primary task is to generate 5 unique Google search keywords that will help gather detailed information primarily related to the sub-question.

        Instructions:
        1. Focus primarily on the sub-question when generating keywords. The main query serves as context but should not dominate the keyword selection.
        2. Ensure that at least 4 out of 5 keywords are directly relevant to the sub-question.
        3. You may use 1 keyword to bridge the sub-question with the broader context of the main query if relevant.
        4. Generate keywords that cover various aspects of the sub-question, including but not limited to: specific details, expert opinions, case studies, recent developments, and historical context (if applicable).
        5. Aim for a mix of broad and specific keywords related to the sub-question to ensure comprehensive coverage.
        6. Ensure all keywords are unique and shed light on different aspects of the sub-question.

        The goal is to provide a focused and in-depth understanding of the sub-question while maintaining awareness of its place within the broader topic.
        """
        user_content = f"Main query (for context): {user_input}\nSub-question (primary focus): {sub_question}\nPlease generate keywords primarily addressing the sub-question, while considering the main query as context."
        
        logger.debug(f"System content for OpenAI API: {system_content}")
        logger.debug(f"User content for OpenAI API: {user_content}")
        
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            response_format=KeywordGeneration
        )
        
        logger.info(f"Successfully received response from OpenAI API")
        logger.debug(f"Raw API response: {response}")
        
        generated_keywords = response.choices[0].message.parsed.keywords
        logger.info(f"Generated {len(generated_keywords)} keywords")
        for idx, keyword in enumerate(generated_keywords, 1):
            logger.debug(f"Generated keyword {idx}: {keyword}")
        
        return response
    except Exception as e:
        logger.error(f"Error generating keywords: {str(e)}", exc_info=True)
        logger.debug(f"Error details: {type(e).__name__}, {str(e)}")
        return None
    finally:
        logger.info("Completed keyword generation process")
