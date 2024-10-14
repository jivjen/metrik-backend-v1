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
from models import SubQuestion, CompleteAnalysis, PDFAnalysis, ResearchStatus, ResearchProgress
from openai import AsyncOpenAI
from typing import Callable
from models import ResearchStatus, ResearchProgress

logger = logging.getLogger(__name__)

async def process_sub_question(user_input: str, question: SubQuestion, openai: AsyncOpenAI, update_status: Callable, question_index: int, total_questions: int):
    logger.info(f"Processing sub-question {question_index}/{total_questions}: {question.question}")
    update_status(ResearchProgress(
        total_steps=5,
        current_step=1,
        status=ResearchStatus.PROCESSING_SUB_QUESTIONS,
        details=f"Analyzing research question {question_index}/{total_questions}: {question.question[:50]}..."
    ))
    
    # Generate keywords
    update_status(ResearchProgress(
        total_steps=5,
        current_step=2,
        status=ResearchStatus.GENERATING_KEYWORDS,
        details=f"Generating search terms for question {question_index}/{total_questions}"
    ))
    logger.info(f"Starting keyword generation for user input: '{user_input}' and sub-question: '{question.question}'")
    
    keywords_task = keyword_generator(user_input, question.question, openai, update_status)
    file_keywords_task = file_keyword_generator(question.question, openai, update_status)
    
    keywords, file_keywords = await asyncio.gather(keywords_task, file_keywords_task)
    
    extracted_keywords = keywords.choices[0].message.parsed.keywords
    extracted_file_keywords = file_keywords.choices[0].message.parsed.keywords
    
    logger.info(f"Generated keywords: {extracted_keywords}")
    logger.info(f"Generated file keywords: {extracted_file_keywords}")
    logger.info(f"Keyword generation completed. Regular keywords: {len(extracted_keywords)}, File keywords: {len(extracted_file_keywords)}")
    update_status(ResearchProgress(
        total_steps=5,
        current_step=2,
        status=ResearchStatus.KEYWORDS_GENERATED,
        details=f"Generated {len(extracted_keywords) + len(extracted_file_keywords)} search terms for question {question_index}/{total_questions}"
    ))

    # Process keywords and search for documents
    update_status(ResearchProgress(
        total_steps=5,
        current_step=3,
        status=ResearchStatus.SEARCHING_ONLINE,
        details=f"Searching online resources for question {question_index}/{total_questions}"
    ))
    keyword_tasks = [process_keyword(keyword, user_input, question.question, openai, update_status) for keyword in extracted_keywords]
    pdf_search_task = asyncio.create_task(search_for_pdf_files(extracted_file_keywords, update_status=update_status))
    
    keyword_results, pdf_links = await asyncio.gather(asyncio.gather(*keyword_tasks), pdf_search_task)
    
    logger.info(f"Processed {len(keyword_results)} keywords")
    logger.info(f"Found {len(pdf_links)} relevant documents")
    update_status(ResearchProgress(
        total_steps=5,
        current_step=3,
        status=ResearchStatus.SEARCH_COMPLETED,
        details=f"Completed online search for question {question_index}/{total_questions}"
    ))
    
    # Process PDFs
    update_status(ResearchProgress(
        total_steps=5,
        current_step=4,
        status=ResearchStatus.PROCESSING_DOCUMENTS,
        details=f"Analyzing {len(pdf_links)} documents for question {question_index}/{total_questions}"
    ))
    pdf_result = await process_pdfs(pdf_links, user_input, question.question, openai, update_status)
    
    # Combine and refine analyses
    update_status(ResearchProgress(
        total_steps=5,
        current_step=5,
        status=ResearchStatus.COMBINING_ANALYSES,
        details=f"Combining analyses for question {question_index}/{total_questions}"
    ))
    complete_analysis = CompleteAnalysis(analysis=[point for result in keyword_results for point in result.points])
    logger.info(f"Created complete analysis with {len(complete_analysis.analysis)} points")
    
    refined_analysis = await final_analysis_refiner(complete_analysis, user_input, question.question, openai, update_status)
    logger.info(f"Refined analysis length: {len(refined_analysis.refined_analysis)}")

    # Synthesize results
    update_status(ResearchProgress(
        total_steps=5,
        current_step=5,
        status=ResearchStatus.SYNTHESIZING_RESULTS,
        details=f"Synthesizing results for question {question_index}/{total_questions}"
    ))
    answer = await synthesize_combined_analysis(refined_analysis, pdf_result, question.question, openai, update_status)
    logger.info(f"Synthesized answer length: {len(answer['answer'])}")
    
    question.answer = answer["answer"]
    question.references = answer["references"]
    logger.info(f"Updated question with answer (length: {len(question.answer)}) and {len(question.references)} references")
    update_status(ResearchProgress(
        total_steps=5,
        current_step=5,
        status=ResearchStatus.SUB_QUESTION_COMPLETED,
        details=f"Completed research question {question_index}/{total_questions}: {question.question[:50]}..."
    ))

    return refined_analysis, pdf_result

async def process_pdfs(pdf_links: List[str], user_input: str, sub_question: str, openai: AsyncOpenAI, update_status: Callable):
    logger.info(f"Starting to process {len(pdf_links)} PDFs")
    logger.info(f"PDF Links: {pdf_links}")
    
    if not pdf_links:
        logger.warning("No PDF links to process")
        return {"summary": "No PDFs were found for analysis."}
    
    pdf_tasks = [process_pdf(pdf_link, user_input, sub_question, openai, update_status) for pdf_link in pdf_links]
    list_of_processed_files = await asyncio.gather(*pdf_tasks)
    list_of_processed_files = [pdf for pdf in list_of_processed_files if pdf is not None]
    logger.info(f"Successfully processed {len(list_of_processed_files)} out of {len(pdf_links)} PDFs")

    update_status(ResearchStatus.SUMMARIZING_PDF, f"Summarizing {len(list_of_processed_files)} PDFs...")
    logger.info(f"Starting PDF summary for {len(list_of_processed_files)} processed files")
    full_pdf_summary = await summarize_pdf_analyses(list_of_processed_files, user_input, sub_question, openai, update_status)
    logger.info(f"PDF summary completed. Summary length: {len(full_pdf_summary['summary']) if isinstance(full_pdf_summary, dict) and 'summary' in full_pdf_summary else 'N/A'}")
    return full_pdf_summary

async def process_pdf(pdf_link: str, user_input: str, sub_question: str, openai: AsyncOpenAI, update_status: Callable):
    pdf_name = pdf_link.split('/')[-1]
    logger.info(f"Starting processing of PDF: {pdf_name}")
    update_status(ResearchProgress(
        total_steps=5,
        current_step=4,
        status=ResearchStatus.PROCESSING_DOCUMENTS,
        details=f"Processing PDF: {pdf_name[:30]}..."
    ))
    
    try:
        pdf_text = await convert_to_text(pdf_link, update_status)
        logger.info(f"Converted PDF to text. Text length: {len(pdf_text)}")
        
        if len(pdf_text) < 100:  # Adjust this threshold as needed
            logger.warning(f"PDF text too short, possibly failed conversion: {len(pdf_text)} characters for {pdf_name}")
            return None
        
        analysed = await analyze_with_gemini(pdf_text, user_input, sub_question, openai, update_status)
        logger.info(f"Completed Gemini analysis for PDF: {pdf_name}. Analysis length: {len(analysed)}")
        
        if len(analysed) < 2000:
            logger.warning(f"Skipping PDF analysis due to short length: {len(analysed)} characters for {pdf_name}")
            return None
        
        logger.info(f"Completed analysis for PDF: {pdf_name}")
        logger.info(f"PDF Analysis preview: {analysed[:100]}...")
        return PDFAnalysis(url=pdf_link, analysis=analysed)
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_name}: {str(e)}")
        return None

async def research(user_input: str, update_status: Callable):
    logger.info(f"Starting research for user input: {user_input}")
    openai = AsyncOpenAI(api_key="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")
    logger.info("OpenAI client initialized")

    total_steps = 5  # Adjust this based on the main steps in your research process
    progress = ResearchProgress(total_steps=total_steps, current_step=0, status=ResearchStatus.STARTED, details="Starting research")
    update_status(progress)

    progress.current_step += 1
    progress.status = ResearchStatus.GENERATING_SUB_QUESTIONS
    progress.details = "Generating research questions"
    update_status(progress)
    logger.info("Starting sub-question generation")
    sub_questions_response = await generate_sub_questions(user_input, openai)
    sub_questions = sub_questions_response.choices[0].message.parsed.questions
    format_notes = sub_questions_response.choices[0].message.parsed.format_notes
    logger.info(f"Generated {len(sub_questions)} sub-questions")
    logger.info(f"Sub-questions: {sub_questions}")
    logger.info(f"Format notes: {format_notes}")

    progress.current_step += 1
    progress.status = ResearchStatus.PROCESSING_SUB_QUESTIONS
    progress.details = f"Analyzing {len(sub_questions)} research questions"
    update_status(progress)
    logger.info("Starting processing of sub-questions")
    sub_question_tasks = [process_sub_question(user_input, question, openai, update_status, idx, len(sub_questions)) for idx, question in enumerate(sub_questions, 1)]
    logger.info(f"Created {len(sub_question_tasks)} tasks for sub-question processing")
    
    results = await asyncio.gather(*sub_question_tasks)
    logger.info("All sub-questions processed")
    logger.info(f"Received {len(results)} results from sub-question processing")

    full_normal = [result[0] for result in results]
    full_pdf = [result[1] for result in results]
    logger.info(f"Extracted {len(full_normal)} normal analyses and {len(full_pdf)} PDF analyses")

    progress.current_step += 1
    progress.status = ResearchStatus.SYNTHESIZING_RESULTS
    progress.details = "Synthesizing final results"
    update_status(progress)
    logger.info("Starting final synthesis")
    full_final_answer = await final_synthesis(full_pdf, full_normal, user_input, format_notes, openai)
    logger.info("Final synthesis completed")
    logger.info(f"Final answer length: {len(full_final_answer['answer']) if isinstance(full_final_answer, dict) and 'answer' in full_final_answer else 'N/A'}")

    progress.current_step = total_steps
    progress.status = ResearchStatus.COMPLETED
    progress.details = "Research completed successfully"
    update_status(progress)
    logger.info("Research completed successfully")
    return full_final_answer
