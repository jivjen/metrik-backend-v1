from openai.types.chat import ParsedChatCompletion
from openai import OpenAI
from models import FileSearchKeywords

def file_keyword_generator(user_input: str, client: OpenAI) -> ParsedChatCompletion[FileSearchKeywords]:
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
                You are a professional Google search researcher specializing in finding PDF documents.
                Given an input query, your task is to generate 5 unique Google search keywords that will help find relevant PDFs related to the topic.
                Follow these strict guidelines:
                1. Every keyword MUST include 'filetype:pdf' to ensure only PDF results.
                2. Focus on market reports, industry analyses, company presentations, and research papers.
                3. Target high-quality sources like consulting firms, industry associations, and academic institutions.
                4. Include specific years or date ranges to find the most recent and relevant documents.
                5. Use advanced search operators like intitle:, inurl:, or site: to refine results when appropriate.
                6. Ensure each keyword is unique and covers a different aspect of the topic.
                7. Do not use quotation marks or other special characters unless part of a specific search operator.
                8. Tailor keywords to find in-depth, data-rich documents that a top market consultant would value.

                Example format:
                - market size electrical switches India filetype:pdf
                - electrical switch industry report 2022 filetype:pdf
                - site:.org Indian electrical components market analysis filetype:pdf
                """},
                {"role": "user", "content": f"Please generate the PDF-specific search keywords for the following input query: {user_input}"}
            ],
            response_format=FileSearchKeywords
        )
        return response
    except Exception as e:
        print(f"Error generating file keywords: {str(e)}")
        return None
