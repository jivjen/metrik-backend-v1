from pydantic import BaseModel, Field
from typing import List, Optional, Callable, Dict, Tuple
import typing_extensions as typing
from enum import Enum

class SubQuestion(BaseModel):
    question: str = Field(description="The sub-question")
    answer: str = Field(description="The answer to the sub-question")
    references: List[str] = Field(description="List of references for this sub-question")

class SubQuestionGeneration(BaseModel):
    questions: List[SubQuestion] = Field(description="List of generated sub-questions")
    format_notes: str = Field(description="Notes on the requested output format")

class SearchKeywords(BaseModel):
    keywords: List[str] = Field(description="List of google search keywords")

class AnalyzedPoint(BaseModel):
    point: str = Field(description="A condensed point from the search result")
    reference: str = Field(description="The reference for this point")

class AnalyzedSearchPoints(BaseModel):
    points: List[AnalyzedPoint] = Field(description="List of condensed points with references")

class CompleteAnalysis(BaseModel):
    analysis: List[AnalyzedPoint] = Field(description="Complete analysis of the search results")

class RefinedAnalysis(BaseModel):
    refined_analysis: str = Field(description="Refined and condensed analysis with inline references")
    references: List[str] = Field(description="List of all references used in the analysis")

class FileSearchKeywords(BaseModel):
    keywords: List[str] = Field(description="List of Google search keywords for specific file types")

class KeywordGeneration(BaseModel):
    keywords: List[str] = Field(description="List of generated keywords")

class SearchResultAnalysis(BaseModel):
    points: List[AnalyzedPoint] = Field(description="List of analyzed points from the search result")

class FinalAnalysisRefinement(BaseModel):
    refined_analysis: str = Field(description="Refined and condensed analysis")
    references: List[str] = Field(description="List of references used in the analysis")

class PDFAnalysis(BaseModel):
    url: str = Field(description="URL of the PDF")
    # filename: str = Field(description="Filename of the PDF")
    analysis: str = Field(description="Analysis of the PDF content")

class JobProgress(BaseModel):
    stage: str = Field(description="Current stage of the job")
    sub_questions: Optional[List[str]] = Field(description="Generated sub-questions", default=None)
    completed_sub_questions: Optional[int] = Field(description="Number of completed sub-questions", default=None)
    total_sub_questions: Optional[int] = Field(description="Total number of sub-questions", default=None)

class FinalReport(BaseModel):
    main_question: str = Field(description="The main research question")
    sub_questions: List[SubQuestion] = Field(description="List of sub-questions with their answers")
    pdf_analyses: List[PDFAnalysis] = Field(description="Analyses of PDF documents")
    final_summary: str = Field(description="Final summary answering the main question")
    all_references: List[str] = Field(description="List of all references used in the report")

class GeminiPDFAnalysisResponse(typing.TypedDict):
    analysis: str

class ReformatResponseAnalysis(BaseModel):
    analysis: str = Field(description="The entire analysis")

class GeminiPDFSummariseResponse(typing.TypedDict):
    summary: str
    references: List[str]

class GeminiAnswerResponse(typing.TypedDict):
    answer: str
    references: List[str]

class ReformatSummaryResponse(BaseModel):
    summary: str = Field(description="The entire analysis")
    references: List[str] = Field(description="The references")

class ReformatAnswerResponse(BaseModel):
    answer: str = Field(description="The entire answer")
    references: List[str] = Field(description="The references")

class ResearchStatus(str, Enum):
    STARTED = "Research started"
    GENERATING_SUB_QUESTIONS = "Generating research questions"
    PROCESSING_SUB_QUESTIONS = "Analyzing research questions"
    GENERATING_KEYWORDS = "Generating search terms"
    KEYWORDS_GENERATED = "Search terms generated"
    PROCESSING_KEYWORDS = "Analyzing search terms"
    SEARCHING_ONLINE = "Searching online resources"
    SEARCH_COMPLETED = "Online search completed"
    SEARCH_ERROR = "Error in online search"
    ANALYZING_SEARCH_RESULT = "Analyzing search results"
    SEARCH_ANALYSIS_COMPLETED = "Search analysis completed"
    SEARCHING_DOCUMENTS = "Searching for relevant documents"
    DOCUMENTS_FOUND = "Relevant documents found"
    PROCESSING_DOCUMENTS = "Analyzing documents"
    DOCUMENT_ANALYSIS_COMPLETED = "Document analysis completed"
    COMBINING_ANALYSES = "Combining all analyses"
    REFINING_ANALYSIS = "Refining final analysis"
    SYNTHESIZING_RESULTS = "Synthesizing research results"
    SUB_QUESTION_COMPLETED = "Research question answered"
    COMPLETED = "Research completed"
    FAILED = "Research failed"
    DOCUMENT_SEARCH_KEYWORD = "Searching documents for keyword"
    DOCUMENT_SEARCH_KEYWORD_COMPLETED = "Completed search for keyword"
    DOCUMENT_SEARCH_ERROR = "Error searching documents for keyword"
    DOCUMENT_SEARCH_COMPLETED = "Document search completed"
    ONLINE_SEARCH = "Searching online resources"
    ONLINE_SEARCH_COMPLETED = "Completed online resource search"
    ONLINE_SEARCH_ERROR = "Error in online resource search"
    ONLINE_SEARCH_FAILED = "Failed to search online resources"
    KEYWORD_ANALYSIS_COMPLETED = "Completed analysis for keyword"
    PRIMARY_CONVERSION = "Attempting primary conversion method"
    PRIMARY_CONVERSION_COMPLETED = "Successfully converted using primary method"
    PRIMARY_CONVERSION_FAILED = "Failed to convert using primary method"
    PRIMARY_CONVERSION_ERROR = "Error during primary conversion method"
    SECONDARY_CONVERSION = "Attempting secondary conversion method"
    SECONDARY_CONVERSION_COMPLETED = "Successfully converted using secondary method"
    SECONDARY_CONVERSION_ERROR = "Error during secondary conversion method"
    DOCUMENT_DOWNLOAD = "Downloading document"
    DOCUMENT_DOWNLOAD_FAILED = "Failed to download document"
    DOCUMENT_TEXT_EXTRACTION = "Extracting text from document"
    DOCUMENT_PAGE_EXTRACTED = "Extracted text from document page"
    CONVERSION_FALLBACK = "Falling back to alternative conversion method"
    DOCUMENT_CONVERSION_COMPLETED = "Completed document conversion"
    PDF_CONVERSION_COMPLETED = "Completed PDF conversion"
    SUMMARIZING_DOCUMENT = "Summarizing document analyses"

class ResearchProgress(BaseModel):
    total_steps: int
    current_step: int
    status: ResearchStatus
    details: str
