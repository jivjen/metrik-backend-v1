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
    STARTED = "Started"
    GENERATING_SUB_QUESTIONS = "Generating sub-questions"
    PROCESSING_SUB_QUESTIONS = "Processing sub-questions"
    GENERATING_KEYWORDS = "Generating keywords"
    KEYWORDS_GENERATED = "Keywords generated"
    PROCESSING_KEYWORDS = "Processing keywords"
    PROCESSING_KEYWORD = "Processing keyword"
    TAVILY_SEARCH = "Tavily search"
    TAVILY_SEARCH_COMPLETED = "Tavily search completed"
    TAVILY_SEARCH_ERROR = "Tavily search error"
    TAVILY_SEARCH_FAILED = "Tavily search failed"
    ANALYZING_SEARCH_RESULT = "Analyzing search result"
    KEYWORD_ANALYSIS_COMPLETED = "Keyword analysis completed"
    KEYWORDS_PROCESSED = "Keywords processed"
    SEARCHING_PDFS = "Searching for PDFs"
    PDF_SEARCH_STARTED = "PDF search started"
    PDF_SEARCH_KEYWORD = "PDF search keyword"
    PDF_SEARCH_KEYWORD_COMPLETED = "PDF search keyword completed"
    PDF_SEARCH_ERROR = "PDF search error"
    PDF_SEARCH_COMPLETED = "PDF search completed"
    PDFS_FOUND = "PDFs found"
    PROCESSING_PDFS = "Processing PDFs"
    PDF_CONVERSION_STARTED = "PDF conversion started"
    JINA_AI_CONVERSION = "Jina AI conversion"
    JINA_AI_CONVERSION_COMPLETED = "Jina AI conversion completed"
    JINA_AI_CONVERSION_FAILED = "Jina AI conversion failed"
    JINA_AI_CONVERSION_ERROR = "Jina AI conversion error"
    PYMUPDF_CONVERSION = "PyMuPDF conversion"
    PDF_DOWNLOAD = "PDF download"
    PDF_DOWNLOAD_FAILED = "PDF download failed"
    PDF_TEXT_EXTRACTION = "PDF text extraction"
    PDF_PAGE_EXTRACTED = "PDF page extracted"
    PYMUPDF_CONVERSION_COMPLETED = "PyMuPDF conversion completed"
    PYMUPDF_CONVERSION_ERROR = "PyMuPDF conversion error"
    CONVERSION_FALLBACK = "Conversion fallback"
    PDF_CONVERSION_COMPLETED = "PDF conversion completed"
    COMBINING_ANALYSES = "Combining analyses"
    REFINING_ANALYSIS = "Refining analysis"
    SYNTHESIZING_RESULTS = "Synthesizing results"
    SUB_QUESTION_COMPLETED = "Sub-question completed"
    COMPLETED = "Completed"
    FAILED = "Failed"
