import logging
from tenacity import retry, stop_after_attempt, wait_fixed
import google.generativeai as genai
from typing import List, Dict, Tuple
from models import GeminiPDFSummariseResponse, ReformatSummaryResponse
from gemini_safety_config import safety_config
import json
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

async def summarize_pdf_analyses(pdf_results: List[Tuple[str, str]], main_query: str, sub_question: str, openai: AsyncOpenAI) -> Dict[str, str]:
    logger.info("Summarizing PDF analyses")
    for url, analysis in pdf_results:
        logger.debug(f"URL: {url}\nAnalysis: {analysis[:100]}...")  # Log only the first 100 characters of each analysis
    combined_analysis = "\n\n".join([f"Analysis of {url} ({url}):\n{analysis}" for url, analysis in pdf_results])
    gemini_api_key = "AIzaSyAViB80an5gX6nJFZY2zQnna57a80OLKwk"
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = f"""
    Summarize and condense the following analyses of multiple PDF documents to conclusively address:
    Main Query: {main_query}
    Sub-Question: {sub_question}

    Instructions:
    1. Provide a comprehensive summary of all important points from the analyses, focusing on both the main query and sub-question.
    2. Organize the information into well-defined unique points providing great insight into each aspect.
    3. Include specific data points, statistics, or quotes that are particularly relevant to either query.
    4. Highlight any conflicting information or uncertainties in the data.
    5. If either query asks for specific data points, output them exactly as expected.
    6. Each piece of information is provided with the url of the file and then the analysis. So include the respective url as the reference if you are picking up data from that piece of information
    7. Ensure the summary addresses the sub-question mainly and just has any of the important details regarding the main query.

    Combined Analyses:
    {combined_analysis}

    Remember: Provide a concise yet comprehensive summary that captures key insights from all documents, addressing both the main query and sub-question, with clear references to the source materials.
    """

    response = await model.generate_content_async(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=GeminiPDFSummariseResponse
        ),
        safety_settings=safety_config
    )
    logger.debug(f"RAW PDF ANALYSIS RESPONSE = {response}")
    try:
        return json.loads(response.candidates[0].content.parts[0].text)
    except json.JSONDecodeError:
        logger.error("JSON parsing failed. Attempting to reformat the response.")
        return await reformat_with_openai_summary_1(response.candidates[0].content.parts[0].text, openai)

@retry(stop=stop_after_attempt(1), wait=wait_fixed(2))
async def reformat_with_openai_summary_1(raw_response: str, client: AsyncOpenAI) -> str:
    prompt = f"""
    The following text is a response from an AI model that should be in JSON format with 'summary' and 'references' keys, but it may be malformed with extra newline characters or something which is resulting in a JSON parsing error.
    Please reformat this text properly without modifying any of the existing information and give the output according to the defined schema with the entire summary content under the summary key and the references under the references key.

    Raw text:
    {raw_response}
    """

    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in formatting JSON responses. You are supposed to take the input and convert it according to the specified output schema"},
            {"role": "user", "content": prompt}
        ],
        response_format=ReformatSummaryResponse,
    )

    try:
        return {"summary" : response.choices[0].message.parsed.summary, "references": response.choices[0].message.parsed.references}
    except json.JSONDecodeError:
        logger.error("Failed to reformat response. Returning empty string.")
        return ""
