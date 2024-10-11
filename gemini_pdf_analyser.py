import logging
from tenacity import retry, stop_after_attempt, wait_fixed
import google.generativeai as genai
from models import GeminiPDFAnalysisResponse, ReformatResponseAnalysis
from gemini_safety_config import safety_config
import json
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

def setup_logger():
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler('gemini_pdf_analyser.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

setup_logger()

async def analyze_with_gemini(text: str, user_input: str, sub_question: str, openai: AsyncOpenAI) -> str:
    logger.info(f"Starting analysis with Gemini for sub-question: {sub_question}")
    logger.debug(f"User input: {user_input}")
    logger.debug(f"Text length: {len(text)} characters")

    gemini_api_key = "AIzaSyAViB80an5gX6nJFZY2zQnna57a80OLKwk"
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY not found in .env file")
        raise ValueError("GEMINI_API_KEY not found in .env file")
    
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = f"""
      Conduct an in-depth analysis of the following PDF text to address:
      Main Query: {user_input}
      Sub-Question: {sub_question}

      Instructions for Comprehensive Research:
      1. Provide an exhaustive and meticulously detailed answer addressing the sub-question mainly but by keeping the context of the main query in mind.
      2. The input PDF text is mainly relevant to the sub-question but in case of there being direct data related to the main query please include that.
      3. If the text lacks directly relevant information, employ advanced reasoning to extrapolate or make well-founded assumptions. Clearly label and extensively justify each extrapolation or assumption.
      4. Rigorously analyze and include all relevant specific data points, statistics, or quotes from the text. Provide context and explain their significance to both the main query and sub-question.
      5. Organize your response in a well defined points to ensure clarity and depth.
      6. Critically examine any conflicting information in the text. Present a thorough analysis of different interpretations, weighing their merits and implications.
      7. Conduct a deep exploration of broader implications, potential trends, and future scenarios suggested by the text, even if not explicitly stated. Use logical reasoning to connect these to both the main query and sub-question.
      8. If the text appears unrelated, perform a comprehensive analysis to uncover any indirect connections or parallels to either the main query or sub-question. Explain these connections in detail.
      9. Aim for an all-encompassing answer that covers all direct aspects of both queries and explores related fields, interdisciplinary connections.

      Text to analyze:
      {text}

      Remember: Provide a thorough and insightful answer that goes beyond simply summarizing the text. Draw connections, highlight implications, and offer a nuanced understanding of the topic, ensuring you address both the main query and the specific sub-question.
      """

    logger.info(f"Sending request to Gemini API for sub-question: {sub_question}")
    try:
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=GeminiPDFAnalysisResponse
            ),
            safety_settings=safety_config
        )
        logger.debug(f"Raw Gemini API response: {response}")
        
        try:
            parsed_response = json.loads(response.candidates[0].content.parts[0].text)
            analysis = parsed_response.get("analysis")
            if analysis:
                logger.info(f"Successfully parsed Gemini API response for sub-question: {sub_question}")
                logger.debug(f"Analysis length: {len(analysis)} characters")
                return analysis
            else:
                logger.warning(f"No analysis found in parsed response for sub-question: {sub_question}")
                return ""
        except json.JSONDecodeError:
            logger.error(f"JSON parsing failed for sub-question: {sub_question}. Attempting to reformat the response.")
            return await reformat_with_openai_analysis(response.candidates[0].content.parts[0].text, openai)
    except Exception as e:
        logger.error(f"Error in Gemini API request for sub-question: {sub_question}. Error: {str(e)}")
        return ""

async def reformat_with_openai_analysis(raw_response: str, client: AsyncOpenAI) -> str:
    logger.info("Starting reformatting with OpenAI")
    try:
        prompt = f"""
        The following text is a response from an AI model that should be in JSON format with an 'analysis' key, but it may be malformed with extra newline characters or something which is resulting in a JSON parsing error. 
        Please reformat this text properly without modifying any of the existing information and give the output according to the defined schema with all the content under the analysis key.

        Raw text:
        {raw_response}
        """

        logger.debug(f"Sending prompt to OpenAI (first 500 chars): {prompt[:500]}...")
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in formatting JSON responses."},
                {"role": "user", "content": prompt}
            ],
            response_format=ReformatResponseAnalysis,
        )

        reformatted_analysis = response.choices[0].message.parsed.analysis
        logger.info("Successfully reformatted response with OpenAI")
        logger.debug(f"Reformatted analysis length: {len(reformatted_analysis)} characters")
        return reformatted_analysis
    except Exception as e:
        logger.error(f"Failed to reformat response: {str(e)}. Returning empty string.")
        return ""
