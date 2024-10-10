from pydantic import BaseModel, Field
from typing import List, Optional, Callable, Dict, Tuple
import typing_extensions as typing

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