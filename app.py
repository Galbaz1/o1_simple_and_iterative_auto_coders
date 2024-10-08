import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import asyncio
from typing import List
import json
from termcolor import colored

class Reflection(BaseModel):
    reflection: str = Field(..., description="The entire reflection that went before arriving at the conclusion")

class Object(BaseModel):
    object: str = Field(..., description="The name of the object")
    reflection: List[Reflection] 
    color: str = Field(..., description="The color of the object")

client = instructor.from_openai(AsyncOpenAI(), mode=instructor.Mode.JSON_O1)

async def extract_object_info(obj: str):
    resp = await client.chat.completions.create(
        model="o1-preview",
        response_model=Object,
        max_retries=3,
        messages=[
            {
                "role": "user",
                "content": f"""Describe the exact color of the object: {obj}."""
            },
        ],
    )
    
    # Extract the completion_tokens_details
    completion_tokens_details = resp.usage.completion_tokens_details
    
    # Convert the pydantic model to a dictionary
    object_info = json.loads(resp.model_dump_json())
    
    # Add the completion_tokens_details to the object_info
    object_info['completion_tokens_details'] = completion_tokens_details
    
    return object_info

async def process_objects():
    objects = ["moonstone", "plutonium", "Pearl"]
    for obj in objects:
        try:
            object_info = await extract_object_info(obj)
            print(colored(json.dumps(object_info, indent=2), 'cyan'))
        except instructor.exceptions.InstructorRetryException as e:
            error_response = {
                "error": f"Extraction failed for {obj}",
                "attempts": e.n_attempts,
                "last_error": str(e),
                "last_completion": e.last_completion
            }
            print(colored(json.dumps(error_response, indent=2), 'red'))

if __name__ == "__main__":
    asyncio.run(process_objects())