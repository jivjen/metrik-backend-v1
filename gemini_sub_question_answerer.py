import logging
from tenacity import retry, stop_after_attempt, wait_fixed
import google.generativeai as genai
from models import RefinedAnalysis, ReformatAnswerResponse, GeminiAnswerResponse
from typing import Dict
from gemini_safety_config import safety_config
import json
from openai import AsyncOpenAI
from typing import Callable
from models import ResearchStatus, ResearchProgress

logger = logging.getLogger(__name__)

async def synthesize_combined_analysis(normal_search_analysis: RefinedAnalysis, pdf_search_analysis: Dict, sub_question: str, openai: AsyncOpenAI, update_status: Callable) -> Dict[str, str]:
    logger.info(f"Starting synthesis of combined analysis for sub-question: {sub_question}")
    update_status(ResearchProgress(
        total_steps=5,
        current_step=5,
        status=ResearchStatus.SYNTHESIZING_RESULTS,
        details=f"Synthesizing results for sub-question: {sub_question[:50]}..."
    ))
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY not found in .env file")
        raise ValueError("GEMINI_API_KEY not found in .env file")
    
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    logger.info(f"Normal Analysis References = {normal_search_analysis.references}")

    normal_analysis = f"""Analysis:
      {normal_search_analysis.refined_analysis}

      References:
      {', '.join(normal_search_analysis.references)}"""

    if pdf_search_analysis.get("summary"):
        logger.info(f"PDF analysis found: {pdf_search_analysis['summary']}")
        # Format PDF search analysis
        pdf_analysis = f"""Analysis:
        {pdf_search_analysis['summary']}

        References:
        {', '.join(pdf_search_analysis['references'])}"""

        logger.info(f"PDF Analysis References = {pdf_search_analysis['references']}")
    else:
        logger.warning("PDF analysis not found")

    logger.info(f"Normal analysis length: {len(normal_analysis)}")
    logger.info(f"PDF analysis length: {len(pdf_analysis)}")

    prompt = f"""
    Synthesize the following analyses to provide a comprehensive answer to the question:

    Question: {sub_question}

    Normal Search Analysis:
    {normal_analysis}

    PDF Search Analysis:
    {pdf_analysis}

    Instructions:
    1. Provide a detailed and comprehensive answer to the question, drawing insights from both analyses.
    2. Organize the information logically in well curated unique points with great depth and breath.
    4. Include all relevant data points, statistics, and quotes from both analyses.
    5. Highlight any conflicting information or uncertainties, providing a balanced view.
    6. Make sure to shape the final answer according to the format notes provided if any. If not, use the most relevant format.
    7. Use inline citations for each piece of information, formatted as [^1^], [^2^], etc.
    8. Make sure to include the reference from the corresponding input that's given and put it under the references key.
    9. If the sub-question requires specific formatting or data presentation, prioritize that in your response.

    Remember: Your response should be a well-structured, in-depth analysis that directly addresses the sub-question while maintaining relevance to the main query context.
    Please note to keep your answer as a whole under the answer key and the references listed under the references key
    """

    logger.info(f"Prompt length: {len(prompt)}")

    try:
        logger.info("Sending request to Gemini API")
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=GeminiAnswerResponse
            ),
            safety_settings=safety_config
        )
        logger.info(f"Raw Gemini API response: {response}")
        
        try:
            result = json.loads(response.candidates[0].content.parts[0].text)
            logger.info("Successfully parsed JSON response from Gemini API")
            logger.info(f"Parsed result: {result}")
            return result
        except json.JSONDecodeError:
            logger.error("JSON parsing failed. Attempting to reformat the response.")
            return await reformat_with_openai_answer_final(response.candidates[0].content.parts[0].text, openai)
    except Exception as e:
        logger.error(f"Error in synthesizing combined analysis: {str(e)}")
        return {"answer": f"Error in synthesizing combined analysis: {str(e)}", "references": []}

async def reformat_with_openai_answer_final(raw_response: str, client: AsyncOpenAI) -> Dict[str, str]:
    logger.info("Starting reformatting with OpenAI")
    try:
        prompt = f"""
        The following text is a response from an AI model that should be in JSON format with 'answer' and 'references' keys, but it may be malformed with extra newline characters or something which is resulting in a JSON parsing error.
        Please reformat this text properly without modifying any of the existing information and give the output according to the defined schema with the entire answer content under the answer key and the references under the references key.

        Raw text:
        {raw_response}
        """

        logger.info(f"Sending prompt to OpenAI (first 500 chars): {prompt[:500]}...")
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in formatting JSON responses. You are supposed to take the input and convert it according to the specified output schema"},
                {"role": "user", "content": prompt}
            ],
            response_format=ReformatAnswerResponse,
        )

        result = {"answer": response.choices[0].message.parsed.answer, "references": response.choices[0].message.parsed.references}
        logger.info("Successfully reformatted response with OpenAI")
        logger.info(f"Reformatted result (first 500 chars): {str(result)[:500]}...")
        return result
    except Exception as e:
        logger.error(f"Failed to reformat response: {str(e)}. Returning empty result.")
        return {"answer": "", "references": []}

