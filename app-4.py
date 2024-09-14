import openai
import instructor
from pydantic import BaseModel, model_validator

client = instructor.from_openai(openai.OpenAI())

class ValidationResult(BaseModel):
    is_valid: bool

class Response(BaseModel):
    really_smart_reasoning: str
    correct_answer: str

    @model_validator(mode="after")
    def validate_answer(self):
        result = client.chat.completions.create(
            model="gpt-4o",
            response_model=ValidationResult,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert validator in math and physics. "
                        "Your task is to determine if the given reasoning and answer align correctly."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Does the following reasoning align with the provided answer?\n\n"
                        f"Reasoning: {self.really_smart_reasoning}\n\n"
                        f"Answer: {self.correct_answer}\n\n"
                        "Please respond with 'true' if they align, or 'false' if they don't."
                    )
                },
            ]
        )
        print(result)
        # You might want to handle the validation result here
        if not result.is_valid:
            raise ValueError("The reasoning and answer do not align.")
        return self

def complex_reason(input: str) -> Response:
    response = client.chat.completions.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "system",
                "content": "You are Fausto, answer the question as if you are a super expert in math and physics",
            },
            {
                "role": "user", 
                "content": input
            },
        ]
    )
    print(response)
    return response

print(complex_reason("What is my name?"))