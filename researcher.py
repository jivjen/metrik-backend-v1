import asyncio
import logging
from typing import List, Dict
from sub_question_generator import generate_sub_questions
from keyword_generator import keyword_generator
from file_keyword_generator import file_keyword_generator
from keyword_processor import process_keyword
from final_analysis_refiner import final_analysis_refiner
from pdf_to_text_converter import convert_to_text
from pdf_searcher import search_for_pdf_files
from gemini_pdf_analyser import analyze_with_gemini
from gemini_summarize_pdf_analysis import summarize_pdf_analyses
from gemini_final_answerer import final_synthesis
from gemini_sub_question_answerer import synthesize_combined_analysis
from models import SubQuestion, CompleteAnalysis, PDFAnalysis, RefinedAnalysis
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def process_sub_question(user_input: str, question: SubQuestion, openai: AsyncOpenAI):
    logger.info(f"Processing sub-question: {question.question}")
    
    # Generate keywords concurrently
    keywords_task = asyncio.create_task(keyword_generator(user_input, question.question, openai))
    file_keywords_task = asyncio.create_task(file_keyword_generator(question.question, openai))
    
    keywords, file_keywords = await asyncio.gather(keywords_task, file_keywords_task)
    
    extracted_keywords = keywords.choices[0].message.parsed.keywords
    extracted_file_keywords = file_keywords.choices[0].message.parsed.keywords
    
    logger.info(f"Keywords: {extracted_keywords}")
    logger.info(f"File Keywords: {extracted_file_keywords}")

    # Process keywords concurrently
    processed_tasks = [process_keyword(keyword, user_input, question.question, openai) for keyword in extracted_keywords]
    list_of_processed = await asyncio.gather(*processed_tasks)
    
    complete_analysis = CompleteAnalysis(analysis=[point for result in list_of_processed for point in result.points])
    refined_analysis = await final_analysis_refiner(complete_analysis, user_input, question.question, openai)
    logger.info(f"Refined Analysis: {refined_analysis}")

    # Process PDF files concurrently
    pdf_links = await search_for_pdf_files(extracted_file_keywords)
    logger.info(f"PDF Links: {pdf_links}")
    
    pdf_tasks = [process_pdf(pdf_link, user_input, question.question, openai) for pdf_link in pdf_links]
    list_of_processed_files = await asyncio.gather(*pdf_tasks)
    list_of_processed_files = [pdf for pdf in list_of_processed_files if pdf is not None]

    full_pdf_summary = await summarize_pdf_analyses(list_of_processed_files, user_input, question.question, openai)
    logger.info(f"Full PDF Analysis: {full_pdf_summary}")

    answer = await synthesize_combined_analysis(refined_analysis, full_pdf_summary, question.question, openai)
    question.answer = answer["answer"]
    question.references = answer["references"]

    return refined_analysis, full_pdf_summary

async def process_pdf(pdf_link: str, user_input: str, sub_question: str, openai: AsyncOpenAI):
    pdf_text = await convert_to_text(pdf_link)
    analysed = await analyze_with_gemini(pdf_text, user_input, sub_question, openai)
    if len(analysed) < 2000:
        logger.info(f"Skipping PDF analysis due to short length: {analysed}")
        return None
    logger.info(f"PDF Analysis: {analysed}")
    return PDFAnalysis(url=pdf_link, analysis=analysed)

async def research(user_input: str):
    openai = AsyncOpenAI(api_key="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")

    logger.info(f"Starting research for user input: {user_input}")
    sub_questions_response = await generate_sub_questions(user_input, openai)
    sub_questions = sub_questions_response.choices[0].message.parsed.questions
    format_notes = sub_questions_response.choices[0].message.parsed.format_notes

    # Process all sub-questions concurrently
    sub_question_tasks = [process_sub_question(user_input, question, openai) for question in sub_questions]
    results = await asyncio.gather(*sub_question_tasks)

    full_normal = [result[0] for result in results]
    full_pdf = [result[1] for result in results]

    full_final_answer = await final_synthesis(full_pdf, full_normal, user_input, openai)
    logger.info("Research completed successfully")

    return full_final_answer
