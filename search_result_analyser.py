from openai import OpenAI
from models import AnalyzedSearchPoints, SearchResultAnalysis


def search_result_analyzer(search_result: str, user_input: str, sub_question: str, client: OpenAI) -> AnalyzedSearchPoints:
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"""
                You are an expert market researcher and consultant specializing in data analysis and insight extraction.
                Your task is to analyze search results related to the specific sub-question: {sub_question} keeping in mind the context of the main query: {user_input}.
                A previous AI agent has generated the search results and data for the analysis.
                You will receive the entire dump of this data.
                Your responsibilities:

                1. Thoroughly review the provided search result data.
                2. Extract all potentially relevant information related to the sub-question: {sub_question} and the main query: {user_input}.
                3. Extrapolate insights and connections from the data, even if not explicitly stated.
                4. Condense the information into clear, concise points.
                5. For each point, provide a reference to its source
                6. In case of deductions or extrapolations of info please explain the approach and thought process behind coming up with that deduction

                Prioritize depth and breadth of analysis. Extrapolate required info that answers the sub-question or main query, drawing connections where possible. Do not invent information, but interpret and analyze deeply.
                Your output should be a comprehensive summary of all potentially relevant information found in the search results, including extrapolated insights, with each point clearly linked to its source and labeled appropriately.
                """},
                {"role": "user", "content": f"Main query: {user_input} \nSub-question: {sub_question} \nHere is the search result chunk: {search_result}"}
            ],
            response_format=SearchResultAnalysis
        )
        result = AnalyzedSearchPoints(points=response.choices[0].message.parsed.points)
        return result
    except Exception as e:
        print(f"Error analyzing search results: {str(e)}")
        return None