import asyncio
import logging
from typing import List, Dict, Callable
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
from models import SubQuestion, CompleteAnalysis, PDFAnalysis, ResearchStatus
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def process_sub_question(user_input: str, question: SubQuestion, openai: AsyncOpenAI, update_status: Callable):
    logger.info(f"Processing sub-question: {question.question}")
    update_status(ResearchStatus.PROCESSING_SUB_QUESTIONS, f"Processing sub-question: {question.question[:50]}...")
    
    # Generate keywords concurrently
    update_status(ResearchStatus.GENERATING_KEYWORDS, f"Generating keywords for: {question.question[:50]}...")
    keywords_task = asyncio.create_task(keyword_generator(user_input, question.question, openai))
    file_keywords_task = asyncio.create_task(file_keyword_generator(question.question, openai))
    
    keywords, file_keywords = await asyncio.gather(keywords_task, file_keywords_task)
    
    extracted_keywords = keywords.choices[0].message.parsed.keywords
    extracted_file_keywords = file_keywords.choices[0].message.parsed.keywords
    
    logger.info(f"Keywords: {extracted_keywords}")
    logger.info(f"File Keywords: {extracted_file_keywords}")

    logger.info("Going to the next process")
    # Process keywords and PDF files concurrently
    update_status(ResearchStatus.PROCESSING_KEYWORDS, f"Processing keyword: {extracted_keywords[0][:30]}...")
    keyword_tasks = [process_keyword(keyword, user_input, question.question, openai) for keyword in extracted_keywords]
    update_status(ResearchStatus.SEARCHING_PDF, f"Searching PDFs with: {extracted_file_keywords[0][:30]}...")
    pdf_search_task = asyncio.create_task(search_for_pdf_files(extracted_file_keywords))
    logger.info("out to the next process")
    
    # Start PDF processing as soon as PDF search is done
    update_status(ResearchStatus.PROCESSING_PDF, f"Processing PDFs for: {question.question[:50]}...")
    pdf_processing_task = asyncio.create_task(process_pdfs(pdf_search_task, user_input, question.question, openai, update_status))
    
    # Wait for keyword processing and PDF processing to complete
    logger.info("Waiting for keyword processing and PDF processing to complete...")
    keyword_results, pdf_result = await asyncio.gather(asyncio.gather(*keyword_tasks), pdf_processing_task)
    logger.info("Keyword processing and PDF processing completed.")
    
    complete_analysis = CompleteAnalysis(analysis=[point for result in keyword_results for point in result.points])
    update_status(ResearchStatus.REFINING_ANALYSIS, f"Refining analysis for sub-question: {question.question}")
    logger.info("Starting final analysis refinement...")
    refined_analysis = await final_analysis_refiner(complete_analysis, user_input, question.question, openai)
    logger.info("Final analysis refinement completed.")

    full_pdf_summary = pdf_result
    logger.info("PDF analysis completed.")

    update_status(ResearchStatus.SYNTHESIZING_RESULTS, f"Synthesizing results for sub-question: {question.question}")
    logger.info("Starting synthesis of combined analysis...")
    answer = await synthesize_combined_analysis(refined_analysis, full_pdf_summary, question.question, openai)
    logger.info("Synthesis of combined analysis completed.")
    question.answer = answer["answer"]
    question.references = answer["references"]

    return refined_analysis, full_pdf_summary

async def process_pdfs(pdf_search_task: asyncio.Task, user_input: str, sub_question: str, openai: AsyncOpenAI, update_status: Callable):
    pdf_links = await pdf_search_task
    logger.info(f"PDF Links: {pdf_links}")
    
    pdf_tasks = [process_pdf(pdf_link, user_input, sub_question, openai, update_status) for pdf_link in pdf_links]
    list_of_processed_files = await asyncio.gather(*pdf_tasks)
    list_of_processed_files = [pdf for pdf in list_of_processed_files if pdf is not None]

    update_status(ResearchStatus.SUMMARIZING_PDF, f"Summarizing {len(list_of_processed_files)} PDFs...")
    full_pdf_summary = await summarize_pdf_analyses(list_of_processed_files, user_input, sub_question, openai)
    return full_pdf_summary

async def process_pdf(pdf_link: str, user_input: str, sub_question: str, openai: AsyncOpenAI, update_status: Callable):
    pdf_name = pdf_link.split('/')[-1]
    update_status(ResearchStatus.PROCESSING_PDF, f"Processing PDF: {pdf_name[:30]}...")
    pdf_text = await convert_to_text(pdf_link)
    analysed = await analyze_with_gemini(pdf_text, user_input, sub_question, openai)
    if len(analysed) < 2000:
        logger.info(f"Skipping PDF analysis due to short length: {analysed}")
        return None
    logger.info(f"PDF Analysis: {analysed[:100]}...")
    return PDFAnalysis(url=pdf_link, analysis=analysed)

async def research(user_input: str, update_status: Callable):
    openai = AsyncOpenAI(api_key="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")

    logger.info(f"Starting research for user input: {user_input}")
    update_status(ResearchStatus.GENERATING_SUB_QUESTIONS, "Generating sub-questions")
    sub_questions_response = await generate_sub_questions(user_input, openai)
    sub_questions = sub_questions_response.choices[0].message.parsed.questions
    format_notes = sub_questions_response.choices[0].message.parsed.format_notes
    logger.info(f"Generated {len(sub_questions)} sub-questions")

    # Process all sub-questions concurrently
    logger.info("Starting processing of sub-questions")
    sub_question_tasks = [process_sub_question(user_input, question, openai, update_status) for question in sub_questions]
    results = await asyncio.gather(*sub_question_tasks)
    logger.info("All sub-questions processed")

    full_normal = [result[0] for result in results]
    full_pdf = [result[1] for result in results]

    update_status(ResearchStatus.SYNTHESIZING_RESULTS, "Synthesizing final results")
    logger.info("Starting final synthesis")
    full_final_answer = await final_synthesis(full_pdf, full_normal, user_input, format_notes, openai)
    logger.info("Final synthesis completed")
    logger.info("Research completed successfully")

    return full_final_answer
