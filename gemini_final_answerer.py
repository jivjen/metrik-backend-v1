import logging
from tenacity import retry, stop_after_attempt, wait_fixed
import google.generativeai as genai
from models import RefinedAnalysis, ReformatAnswerResponse, GeminiAnswerResponse
from typing import List, Dict
from gemini_safety_config import safety_config
import json
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

async def final_synthesis(full_pdf: List[Dict[str, str]], full_normal: List[RefinedAnalysis], main_query: str, format_notes: str, openai: AsyncOpenAI) -> Dict[str, str]:
    gemini_api_key = "AIzaSyAViB80an5gX6nJFZY2zQnna57a80OLKwk"
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    # Format PDF analyses
    pdf_analyses = "\n\n".join([
        f"PDF Analysis {i+1}:\n{pdf['summary']}\nReferences:\n{', '.join(pdf['references'])}"
        for i, pdf in enumerate(full_pdf)
    ])

    # Format normal analyses
    normal_analyses = "\n\n".join([
        f"Normal Analysis {i+1}:\n{normal.refined_analysis}\nReferences:\n{', '.join(normal.references)}"
        for i, normal in enumerate(full_normal)
    ])

    prompt = f"""
    Synthesize all the following analyses to provide a comprehensive answer to the main query:

    Main Query: {main_query}

    Format Notes: {format_notes}

    PDF Analyses:
    {pdf_analyses}

    Normal Analyses:
    {normal_analyses}

    Instructions:
    1. Provide a detailed, comprehensive answer to the main query, synthesizing insights from all analyses.
    2. Organize the information logically, using headings and subheadings where appropriate.
    3. Include all relevant data points, statistics, and quotes from the analyses.
    4. Highlight any conflicting information or uncertainties, providing a balanced view.
    5. Use inline citations for each piece of information, formatted as [^1^], [^2^], etc.
    6. Create a consolidated, numbered References list with all unique sources used.
    7. Format each reference as a clickable link using Markdown syntax: [Reference text](URL)
    8. If the main query requires specific formatting or data presentation, prioritize that in your response.
    9. Ensure your response is well-structured, in-depth, and directly addresses the main query.

    Remember: Your task is to synthesize all the provided information into a cohesive, comprehensive answer to the main query.
    Please note to keep your answer as a whole under the answer key and the references listed under the references key
    """

    logger.info("Generating final synthesis")
    response = await model.generate_content_async(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=GeminiAnswerResponse
        ),
        safety_settings=safety_config
    )
    logger.debug(f"RAW QUESTION RESPONSE = {response}")
    try:
        return json.loads(response.candidates[0].content.parts[0].text)
    except json.JSONDecodeError as j:
        logger.error(f"JSON parsing failed. Attempting to reformat the response. {j}")
        return await reformat_with_openai_summary_fully_final(response.candidates[0].content.parts[0].text, openai)
    return {"answer": "Error in final synthesis.", "references": []}

async def reformat_with_openai_summary_fully_final(raw_response: str, client: AsyncOpenAI) -> Dict[str, str]:
    try:
        prompt = f"""
        The following text is a response from an AI model that should be in JSON format with 'answer' and 'references' keys, but it may be malformed with extra newline characters or something which is resulting in a JSON parsing error.
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
            response_format=ReformatAnswerResponse,
        )

        return {"answer": response.choices[0].message.parsed.answer, "references": response.choices[0].message.parsed.references}
    except Exception as e:
        logger.error(f"Failed to reformat response: {str(e)}. Returning empty string.")
        return {"answer": "", "references": []}
