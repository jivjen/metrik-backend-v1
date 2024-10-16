import logging
from openai.types.chat import ParsedChatCompletion
from openai import AsyncOpenAI
from models import SubQuestionGeneration 
from typing import Callable
from models import ResearchStatus, ResearchProgress

logger = logging.getLogger(__name__)

async def generate_sub_questions(user_input: str, client: AsyncOpenAI) -> ParsedChatCompletion[SubQuestionGeneration]:
    logger.info(f"Starting sub-question generation for user input: '{user_input}'")
    try:
        logger.info("Preparing OpenAI API request for sub-question generation")
        system_content = """
        You are an expert researcher and critical thinker. Your task is to analyze the user's input and generate a comprehensive set of sub-questions. Follow these guidelines:

        1. If the user has provided specific sub-questions or topics, incorporate them into your output by making it super specific and detailed.
        2. If the user hasn't provided enough sub-questions (less than 5), add more to ensure comprehensive coverage.
        3. Ensure all aspects of the topic are covered such that the user can understand the topic in depth. Ex. historical, current, future, technological, societal, economic, etc.
        4. Consider both broad, overarching questions and specific, detailed inquiries.
        5. Frame questions to encourage deep, insightful responses rather than simple facts.
        6. If the user input specifies a particular output format or structure directly or indirectly, make sure the questions are framed accordingly and also that they are noted in your response.
        7. Aim for a maximum of 5 sub-questions including the extrapolated ones from the user input to keep the research focused.
        8. If and only if the user input scope is not covered to its entirety then you can generate 2 more questions such that all of the user input is covered.
        """
        user_content = f"Analyze and generate sub-questions for the following user input: {user_input}"
        
        logger.info(f"System content for OpenAI API: {system_content}")
        logger.info(f"User content for OpenAI API: {user_content}")
        
        logger.info("Sending request to OpenAI API for sub-question generation")
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            response_format=SubQuestionGeneration
        )
        
        logger.info("Successfully received response from OpenAI API")
        logger.info(f"Raw API response: {response}")
        
        generated_questions = response.choices[0].message.parsed.questions
        format_notes = response.choices[0].message.parsed.format_notes
        
        logger.info(f"Generated {len(generated_questions)} sub-questions")
        for idx, question in enumerate(generated_questions, 1):
            logger.info(f"Generated sub-question {idx}: {question}")
        
        logger.info(f"Format notes: {format_notes}")
        
        return response
    except Exception as e:
        logger.error(f"Error generating sub-questions: {str(e)}", exc_info=True)
        logger.info(f"Error details: {type(e).__name__}, {str(e)}")
        return None
    finally:
        logger.info("Completed sub-question generation process")
