from instructor import from_openai, Mode, exceptions
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import asyncio
from typing import List
import json
from termcolor import colored

# Define a Reflection model to represent a single reflection
class Reflection(BaseModel):
    reflection: str = Field(..., description="The entire reflection that went before arriving at the conclusion")

# Define an Object model to represent an object with its properties
class Object(BaseModel):
    object: str = Field(..., description="The name of the object")
    reflection: List[Reflection] 
    color: str = Field(..., description="The color of the object")

# Initialize the OpenAI client with instructor wrapper
client = from_openai(AsyncOpenAI(), mode=Mode.TOOLS_STRICT)

# Asynchronous function to extract information about a given object
async def extract_object_info(obj: str):
    """
    Extracts color information for a given object.

    Args:
        obj (str): The name of the object to analyze.

    Returns:
        dict: A dictionary containing the extracted object information.
    """
    resp = await client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        response_model=Object,
        max_retries=3,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "You are a color expert. Describe the exact color of the object."
            },
            {
                "role": "user",
                "content": f"""Describe the exact color of the object: {obj}."""
            },
        ],
    )
    return json.loads(resp.model_dump_json())

# Main function to process multiple objects concurrently
async def process_objects():
    """
    Processes a list of objects concurrently using asyncio tasks.
    """
    objects = ["moonstone", "plutonium", "pearl"]
    tasks = []
    for obj in objects:
        tasks.append(asyncio.create_task(process_single_object(obj)))
    await asyncio.gather(*tasks)

# Function to process a single object and handle potential exceptions
async def process_single_object(obj):
    """
    Processes a single object, extracting its information and handling potential exceptions.

    Args:
        obj (str): The name of the object to process.
    """
    try:
        object_info = await extract_object_info(obj)
        # Print the 'color' key in blue, and the rest in yellow
        color_value = object_info.pop('color', None)
        print(colored(json.dumps(object_info, indent=2), 'yellow'))
        if color_value:
            print(colored(f'  "color": "{color_value}"', 'blue'))
        print()  # Add a blank line between objects
    except exceptions.InstructorRetryException as e:
        error_response = {
            "error": f"Extraction failed for {obj}",
            "attempts": e.n_attempts,
            "last_error": str(e),
            "last_completion": e.last_completion
        }
        print(colored(json.dumps(error_response, indent=2), 'yellow'))
        print()  # Add a blank line between objects

# Entry point of the script
if __name__ == "__main__":
    asyncio.run(process_objects())