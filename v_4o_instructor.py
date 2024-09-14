import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import asyncio
from typing import List
import json
from termcolor import colored
import random

# List of available colors for termcolor
COLORS = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan']

class Reflection(BaseModel):
    reflection: str = Field(..., description="The entire reflection that went before arriving at the conclusion")

class Object(BaseModel):
    object: str = Field(..., description="The name of the object")
    reflection: List[Reflection] 
    color: str = Field(..., description="The color of the object")

client = instructor.from_openai(AsyncOpenAI())

async def extract_object_info(obj: str):
    resp = await client.chat.completions.create(
        model="gpt-4o",
        response_model=Object,
        temperature=0.0,
        max_retries=3,
        messages=[
            {
                "role": "user",
                "content": f"""Describe the exact color of the object: {obj}."""
            },
        ],
    )
    return json.loads(resp.model_dump_json())

async def process_objects():
    objects = ["moonstone", "plutonium", "Pearl"]
    tasks = [extract_object_info(obj) for obj in objects]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    used_colors = set()
    for obj, result in zip(objects, results):
        # Choose a random color that hasn't been used recently
        available_colors = list(set(COLORS) - used_colors)
        if not available_colors:
            available_colors = COLORS
        color = random.choice(available_colors)
        used_colors.add(color)
        if len(used_colors) > len(COLORS) // 2:
            used_colors.pop()

        if isinstance(result, Exception):
            error_response = {
                "error": f"Extraction failed for {obj}",
                "last_error": str(result),
            }
            print(colored(json.dumps(error_response, indent=2), color))
        else:
            print(colored(json.dumps(result, indent=2), color))

if __name__ == "__main__":
    asyncio.run(process_objects())