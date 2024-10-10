import logging
from openai import AsyncOpenAI
from models import RefinedAnalysis, FinalAnalysisRefinement, CompleteAnalysis

logger = logging.getLogger(__name__)

async def final_analysis_refiner(complete_analysis: CompleteAnalysis, user_input: str, sub_question: str, client: AsyncOpenAI) -> RefinedAnalysis:
    formatted_points = "\n\n".join([f"Point: {point.point}\nReference: {point.reference}" for point in complete_analysis.analysis])

    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Refining analysis for sub-question: {sub_question} (Attempt {attempt + 1})")
            response = await client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"""
                    You are an expert market researcher and data analyst specializing in synthesizing complex information. Your task is to create a comprehensive and detailed analysis related to the main query: {{main_query}} and the specific sub-question: {{sub_question}}.
                    You will receive a set of analyzed search points with their references. Your responsibilities:

                    1. Review all points and references in the analysis.
                    2. Eliminate exact duplicates, retain similar information if it adds context.
                    3. Organize information logically, using headings and subheadings.
                    4. Present a well-written, flowing text that addresses the sub-question in depth keeping in mind the main query for context.
                    5. Include all relevant data points, statistics, and trends.
                    6. Provide detailed explanations and context.
                    7. Highlight conflicting information or uncertainties.
                    8. Use inline citations [^1^], [^2^], etc.
                    9. If the main query or sub-question asks for specific details or formatting, prioritize that in the output.
                    10. Create a consolidated, numbered References list with unique sources as clickable Markdown links.

                    Produce a detailed, comprehensive summary addressing both the main query and sub-question.
                    Focus on depth and completeness, suitable for a professional research report.
                    """},
                    {"role": "user", "content": f"Main query: {user_input}\nSub-question: {sub_question}\nComplete analysis to refine: {formatted_points}"}
                ],
                response_format=FinalAnalysisRefinement
            )
            refined_analysis = RefinedAnalysis(refined_analysis=response.choices[0].message.parsed.refined_analysis, references=response.choices[0].message.parsed.references)
            logger.info(f"Successfully refined analysis for sub-question: {sub_question}")
            return refined_analysis
        except Exception as e:
            logger.error(f"An error occurred during final analysis refinement (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise

    return None
