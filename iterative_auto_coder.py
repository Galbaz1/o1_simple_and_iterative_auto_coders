# Import necessary libraries
import sys
from openai import OpenAI
from termcolor import colored
import re
import subprocess
import time
import threading
import os
import psutil

# Set the number of iterations for code improvement
n = 10

# Initialize OpenAI client
client = OpenAI()
model = "o1-preview" 
#model = "o1-mini"

# Function to run code with a timeout
def run_code(code, timeout=5):
    print(colored("Running code...", "cyan"))
    def target():
        # Create and run a temporary Python file
        temp_file = os.path.join("responses", "temp_code.py")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        process = subprocess.Popen(['python', temp_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        # Clean up temporary file
        os.remove(temp_file)
        
        return stdout, stderr, process.returncode

    # Run the code in a separate thread with a timeout
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        print(colored("Code execution timed out after 5 seconds", "yellow"))
        # Terminate the process if it's still running
        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        return None, "Timeout", None
    return target()

# Function to improve code iteratively
def improve_code(initial_code, n, user_task):
    print(colored(f"Starting code improvement process for {n} iterations...", "cyan"))
    
    # Ensure responses folder exists
    responses_folder = "responses"
    os.makedirs(responses_folder, exist_ok=True)
    
    for i in range(n):
        print(colored(f"\nIteration {i+1}/{n}:", "cyan"))
        
        # Run the current code and check for errors
        print(colored("  Running current code...", "cyan"))
        stdout, stderr, returncode = run_code(initial_code)
        
        # If there's an error, ask AI to fix it
        if stderr and stderr != "Timeout":
            print(colored(f"  Error detected in iteration {i+1}:", "red"))
            print(stderr)
            print(colored("  Requesting error fix from AI...", "cyan"))
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"The following code produced an error. Please fix it:\n\n{initial_code}\n\nError:\n{stderr}. please return the full fixed code in its entirety."
                    }
                ]
            )
        else:
            # If no error, ask AI to improve the code
            print(colored("  No errors detected. Requesting code improvement from AI...", "cyan"))
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Please improve the following Python code for {user_task}. Make sure to think of all features and write as much code to cover as many features as possible. This is iteration {i+1} of {n}:\n\n{initial_code}"
                    }
                ]
            )
        
        # Extract improved code from AI response
        code_match = re.search(r'```python\n(.*?)```', response.choices[0].message.content, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            print(colored(f"  Code improved - Iteration {i+1}/{n}", "green"))
            
            # Save the improved code
            print(colored("  Saving iteration code...", "cyan"))
            iteration_filename = f'{user_task[:20].replace(" ", "_").lower()}_iteration_{i+1}.py'
            with open(os.path.join(responses_folder, iteration_filename), 'w', encoding='utf-8') as file:
                file.write(code)
            print(colored(f"Iteration {i+1} saved to {os.path.join(responses_folder, iteration_filename)}", "blue"))
            
            # Save non-code text from AI response
            print(colored("  Saving non-code text...", "cyan"))
            non_code_text = re.sub(r'```python\n.*?```', '', response.choices[0].message.content, flags=re.DOTALL)
            non_code_filename = f'{user_task[:20].replace(" ", "_").lower()}_iteration_{i+1}.txt'
            with open(os.path.join(responses_folder, non_code_filename), 'w', encoding='utf-8') as file:
                file.write(non_code_text.strip())
            print(colored(f"Non-code text saved to {os.path.join(responses_folder, non_code_filename)}", "blue"))
            
            # Print token usage information
            print(colored("  Token usage for this iteration:", "cyan"))
            print(colored(f"Iteration {i+1} Completion Tokens: {response.usage.completion_tokens}", "blue"))
            print(colored(f"Iteration {i+1} Prompt Tokens: {response.usage.prompt_tokens}", "yellow"))
            print(colored(f"Iteration {i+1} Total Tokens: {response.usage.total_tokens}", "red"))
            print(colored(f"Iteration {i+1} Reasoning Tokens: {response.usage.completion_tokens_details.reasoning_tokens}", "magenta"))
            
            initial_code = code  # Update the code for the next iteration
        else:
            print(colored(f"  No Python code found in the response for iteration {i+1}", "red"))
            break
        
        # Allow user to add instructions between iterations
        print(colored("\nBefore next iteration:", "cyan"))
        print(colored("  Press Enter within 3 seconds to add an instruction, or wait to continue:", "cyan"))
        
        start_time = time.time()
        user_pressed_enter = False
        while time.time() - start_time < 3:
            if os.name == 'nt':  # For Windows
                import msvcrt
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\r':  # Enter key
                        user_pressed_enter = True
                        break
            else:  # For Unix-based systems
                import select
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    key = sys.stdin.read(1)
                    if key == '\n':  # Enter key
                        user_pressed_enter = True
                        break
            time.sleep(0.1)
        
        if user_pressed_enter:
            user_input = input(colored("  Enter additional instruction: ", "cyan"))
            if user_input:
                print(colored(f"  Adding user instruction: {user_input}", "green"))
                user_task += f" Additionally, {user_input}"
            else:
                print(colored("  No instruction added.", "yellow"))
        else:
            print(colored("  No input received, continuing to next iteration.", "yellow"))
    
    return initial_code  # Return the final improved code

# Main program execution
print(colored("Welcome to the Iterative Auto Coder!", "green"))
user_task = input(colored("Enter the task for the AI (e.g., 'create a fully functional tower defense game in pygame without using any external files'): ", "cyan"))
user_task += " Please think long and carefully about all the features that are necessary and implement them all without mistakes. think carefully to avoid any errors."

# Ensure responses folder exists
responses_folder = "responses"
os.makedirs(responses_folder, exist_ok=True)

print(colored("\nRequesting initial code from AI...", "cyan"))
# Get initial code from AI
initial_response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user", 
            "content": user_task
        }
    ]
)

# Extract and save initial code
initial_code_match = re.search(r'```python\n(.*?)```', initial_response.choices[0].message.content, re.DOTALL)
if initial_code_match:
    initial_code = initial_code_match.group(1)
    
    print(colored("\nSaving initial code...", "cyan"))
    initial_filename = f'{user_task[:20].replace(" ", "_").lower()}_initial.py'
    with open(os.path.join(responses_folder, initial_filename), 'w', encoding='utf-8') as file:
        file.write(initial_code)
    print(colored(f"Initial code saved to {os.path.join(responses_folder, initial_filename)}", "green"))
    
    print(colored("Saving initial non-code text...", "cyan"))
    initial_non_code_text = re.sub(r'```python\n.*?```', '', initial_response.choices[0].message.content, flags=re.DOTALL)
    initial_text_filename = f'{user_task[:20].replace(" ", "_").lower()}_initial.txt'
    with open(os.path.join(responses_folder, initial_text_filename), 'w', encoding='utf-8') as file:
        file.write(initial_non_code_text.strip())
    print(colored(f"Initial non-code text saved to {os.path.join(responses_folder, initial_text_filename)}", "green"))
    
    # Print token usage for initial response
    print(colored("\nInitial response token usage:", "cyan"))
    print(colored(f"Initial Completion Tokens: {initial_response.usage.completion_tokens}", "blue"))
    print(colored(f"Initial Prompt Tokens: {initial_response.usage.prompt_tokens}", "yellow"))
    print(colored(f"Initial Total Tokens: {initial_response.usage.total_tokens}", "red"))
    print(colored(f"Initial Reasoning Tokens: {initial_response.usage.completion_tokens_details.reasoning_tokens}", "magenta"))
    
    # Allow user to add instructions before starting iterations
    print(colored("\nBefore starting iterations:", "cyan"))
    print(colored("  Press Enter within 3 seconds to add an instruction, or wait to continue:", "cyan"))
    
    start_time = time.time()
    user_pressed_enter = False
    while time.time() - start_time < 3:
        if os.name == 'nt':  # For Windows
            import msvcrt
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'\r':  # Enter key
                    user_pressed_enter = True
                    break
        else:  # For Unix-based systems
            import select
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if rlist:
                key = sys.stdin.read(1)
                if key == '\n':  # Enter key
                    user_pressed_enter = True
                    break
        time.sleep(0.1)
    
    if user_pressed_enter:
        user_input = input(colored("  Enter additional instruction: ", "cyan"))
        if user_input:
            print(colored(f"  Adding user instruction: {user_input}", "green"))
            user_task += f" Additionally, {user_input}"
        else:
            print(colored("  No instruction added.", "yellow"))
    else:
        print(colored("  No input received, continuing to iterations.", "yellow"))
    
    # Start the code improvement process
    print(colored("\nStarting code improvement process...", "cyan"))
    final_code = improve_code(initial_code, n, user_task)
    
    # Save the final improved code
    print(colored("\nSaving final code...", "cyan"))
    final_filename = f'{user_task[:20].replace(" ", "_").lower()}_final.py'
    with open(os.path.join(responses_folder, final_filename), 'w', encoding='utf-8') as file:
        file.write(final_code)
    
    print(colored(f"Final code extracted and saved to {os.path.join(responses_folder, final_filename)} after {n} iterations", "green"))
else:
    print(colored("\nNo Python code found in the initial response", "red"))
    print(colored("Initial response token usage:", "cyan"))
    print(colored(f"Initial Completion Tokens: {initial_response.usage.completion_tokens}", "blue"))
    print(colored(f"Initial Prompt Tokens: {initial_response.usage.prompt_tokens}", "yellow"))
    print(colored(f"Initial Total Tokens: {initial_response.usage.total_tokens}", "red"))
    print(colored(f"Initial Reasoning Tokens: {initial_response.usage.completion_tokens_details.reasoning_tokens}", "magenta"))

print(colored("\nThank you for using the Iterative Auto Coder!", "green"))