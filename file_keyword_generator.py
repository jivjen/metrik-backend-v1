import logging
from openai.types.chat import ParsedChatCompletion
from openai import AsyncOpenAI
from models import FileSearchKeywords

logger = logging.getLogger(__name__)

async def file_keyword_generator(user_input: str, client: AsyncOpenAI) -> ParsedChatCompletion[FileSearchKeywords]:
    logger.info(f"Starting file keyword generation for input: {user_input}")
    try:
        logger.debug(f"Preparing to send request to OpenAI API for file keyword generation")
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
                You are a professional Google search researcher specializing in finding PDF documents.
                Given an input query, your task is to generate 5 unique Google search keywords that will help find relevant PDFs related to the topic.
                Follow these strict guidelines:
                1. Every keyword MUST include 'filetype:pdf' to ensure only PDF results.
                2. Focus on market reports, industry analyses, company presentations, and research papers.
                3. Target high-quality sources like consulting firms, industry associations, and academic institutions.
                4. Include specific years or date ranges to find the most recent and relevant documents.
                5. Use advanced search operators like intitle:, inurl:, or site: to refine results when appropriate.
                6. Ensure each keyword is unique and covers a different aspect of the topic.
                7. Do not use quotation marks or other special characters unless part of a specific search operator.
                8. Tailor keywords to find in-depth, data-rich documents that a top market consultant would value.

                Example format:
                - market size electrical switches India filetype:pdf
                - electrical switch industry report 2022 filetype:pdf
                - site:.org Indian electrical components market analysis filetype:pdf
                """},
                {"role": "user", "content": f"Please generate the PDF-specific search keywords for the following input query: {user_input}"}
            ],
            response_format=FileSearchKeywords
        )
        logger.info(f"Successfully received response from OpenAI API")
        logger.debug(f"Raw API response: {response}")
        
        generated_keywords = response.choices[0].message.parsed.keywords
        logger.info(f"Generated {len(generated_keywords)} file search keywords")
        for idx, keyword in enumerate(generated_keywords, 1):
            logger.debug(f"Generated keyword {idx}: {keyword}")
        
        return response
    except Exception as e:
        logger.error(f"Error generating file keywords: {str(e)}", exc_info=True)
        logger.debug(f"Error details: {type(e).__name__}, {str(e)}")
        return None
    finally:
        logger.info("Completed file keyword generation process")
