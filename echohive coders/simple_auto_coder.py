from openai import OpenAI
from termcolor import colored
import re

client = OpenAI()

# Take user input for the task
user_task = input("Enter the task for the AI (e.g., 'create a fully functional tower defense game in pygame without using any external files'): ")

response = client.chat.completions.create(
    model="o1-mini",
    messages=[
        {
            "role": "user", 
            "content": user_task
        }
    ]
)

# Extract code from the response
code_match = re.search(r'```python\n(.*?)```', response.choices[0].message.content, re.DOTALL)
if code_match:
    extracted_code = code_match.group(1)
    
    # Save the extracted code to a .py file
    with open('generated_code.py', 'w') as file:
        file.write(extracted_code)
    
    print(colored("Code extracted and saved to generated_code.py", "green"))
else:
    print(colored("No Python code found in the response", "red"))

print(colored(f"Completion Tokens: {response.usage.completion_tokens}", "blue"))
print(colored(f"Prompt Tokens: {response.usage.prompt_tokens}", "yellow"))
print(colored(f"Total Tokens: {response.usage.total_tokens}", "red"))
print(colored(f"Reasoning Tokens: {response.usage.completion_tokens_details['reasoning_tokens']}", "magenta"))