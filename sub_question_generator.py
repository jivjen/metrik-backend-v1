from openai.types.chat import ParsedChatCompletion
from openai import OpenAI
from models import SubQuestionGeneration 

def generate_sub_questions(user_input: str, client: OpenAI) -> ParsedChatCompletion[SubQuestionGeneration]:
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
                You are an expert researcher and critical thinker. Your task is to analyze the user's input and generate a comprehensive set of sub-questions. Follow these guidelines:

                1. If the user has provided specific sub-questions or topics, incorporate them into your output by making it super specific and detailed.
                2. If the user hasn't provided enough sub-questions (less than 5), add more to ensure comprehensive coverage.
                3. Ensure all aspects of the topic are covered such that the user can understand the topic in depth. Ex. historical, current, future, technological, societal, economic, etc.
                4. Consider both broad, overarching questions and specific, detailed inquiries.
                5. Frame questions to encourage deep, insightful responses rather than simple facts.
                6. If the user input specifies a particular output format or structure directly or indirectly, make sure the questions are framed accordingly and also that they are noted in your response.
                7. Aim for a maximum of 5 sub-questions including the extrapolated ones from the user input to keep the research focused.
                """},
                {"role": "user", "content": f"Analyze and generate sub-questions for the following user input: {user_input}"}
            ],
            response_format=SubQuestionGeneration
        )
        return response
    except Exception as e:
        print(f"Error generating sub-questions: {str(e)}")
        return None