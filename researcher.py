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
from models import SubQuestion, CompleteAnalysis, PDFAnalysis
from openai import OpenAI


def research(user_input: str):
    openai = OpenAI(api_key="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")

    fulll_pdf = []
    fulll_normal = []
    sub_questions_response = generate_sub_questions(user_input, openai)
    sub_questions = sub_questions_response.choices[0].message.parsed.questions
    format_notes = sub_questions_response.choices[0].message.parsed.format_notes
    for question in sub_questions:
        keywords = keyword_generator(user_input, question.question, openai)
        extracted_keywords = keywords.choices[0].message.parsed.keywords
        # extracted_file_keywords = file_keywords_result.choices[0].message.parsed.keywords
        print(f"Keywords == {extracted_keywords}")
        list_of_processed = []
        for keyword in extracted_keywords:
            processed = process_keyword(keyword, user_input, question.question, openai)
            print(f"Processed = {processed}")
            list_of_processed.append(processed)
        complete_analysis = CompleteAnalysis(analysis=[point for result in list_of_processed for point in result.points])
        refined_analysis = final_analysis_refiner(complete_analysis, user_input, question.question, openai)
        print(f"REFINED ANALYSIS = {refined_analysis}")
        fulll_normal.append(refined_analysis)

        file_keywords = file_keyword_generator(question.question, openai)
        extracted_file_keywords = file_keywords.choices[0].message.parsed.keywords
        print(f"File Keywords == {extracted_file_keywords}")
        pdf_links = search_for_pdf_files(extracted_file_keywords)
        print(f"PDF LINKS = {pdf_links}")
        list_of_processed_files = []
        for pdf_link in pdf_links:
            pdf_text = convert_to_text(pdf_link)
            analysed = analyze_with_gemini(pdf_text, user_input, question.question, openai)
            if len(analysed) < 2000:
                print(f"About to skip this one coz here's the analysis: {analysed}")
                continue
            print(f"PDF ANALYSIS = {analysed}")
            pdf_analysis = PDFAnalysis(url=pdf_link, analysis=analysed)
            list_of_processed_files.append(pdf_analysis)
        full_pdf_summary = summarize_pdf_analyses(list_of_processed_files, user_input, question.question, openai)
        print(f"FULL PDF ANALYSIS = {full_pdf_summary}")
        fulll_pdf.append(full_pdf_summary)

        answer = synthesize_combined_analysis(refined_analysis, full_pdf_summary, question.question, format_notes, openai)
        question.answer = answer["answer"]
        question.references = answer["references"]

    full_final_answer = final_synthesis(fulll_pdf, fulll_normal, user_input)

    return full_final_answer