import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import asyncio
from typing import List

class Reasoning(BaseModel):
    step: str = Field(..., description="A single step in the reasoning process")

class User(BaseModel):
    name: str = Field(..., description="The user's full name")
    age: int = Field(..., description="The user's age in years")
    reasoning: List[Reasoning] = Field(..., description="The chain of thought used to extract this information")

class UserList(BaseModel):
    users: List[User]

client = instructor.from_openai(AsyncOpenAI(), mode=instructor.Mode.JSON_O1)

async def extract_user_info():
    resp = await client.chat.completions.create(
        model="o1-mini",
        response_model=UserList,
        max_retries=3,
        messages=[
      
            {
                "role": "user",
                "content": """Extract the names and ages from the following text, and provide your reasoning: John Doe is 50 and lives in Spain, mary is 20 and lives in the UK"""
            },
        ],
    )
    return resp

if __name__ == "__main__":
    try:
        user_list = asyncio.run(extract_user_info())
        for user in user_list.users:
            print(f"Name: {user.name}")
            print(f"Age: {user.age}")
            print("Reasoning:")
            for step in user.reasoning:
                print(f"- {step.step}")
            print()
    except instructor.exceptions.InstructorRetryException as e:
        print(f"Extraction failed after {e.n_attempts} attempts.")
        print(f"Last error: {e}")
        print(f"Last completion: {e.last_completion}")