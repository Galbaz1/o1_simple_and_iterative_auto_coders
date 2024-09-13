# Auto Code Generation and Improvement

This project contains two Python scripts that utilize OpenAI's API to generate and improve code based on user-provided tasks.

## simple_auto_coder.py

This script is a basic implementation of code generation using OpenAI's API. Here's how it works:

1. It prompts the user to enter a task description.
2. It sends this task to OpenAI's API to generate Python code.
3. The generated code is extracted from the API response.
4. The code is saved to a file named 'generated_code.py'.
5. It prints token usage statistics for the API call.

This script is straightforward and generates code in a single pass.

## iterative_auto_coder.py

This script is an advanced implementation that generates initial code and iteratively improves it. Here's its functionality:

1. It prompts the user to enter a task description.
2. It sends this task to OpenAI's API to generate initial Python code.
3. The initial code and explanatory text are saved to separate files.
4. It then enters an improvement loop (default 10 iterations):
   - In each iteration, it sends the current code back to the API for improvement.
   - The improved code and explanatory text are saved for each iteration.
5. After all iterations, the final improved code is saved.
6. It prints token usage statistics for each API call, including:
   - Completion tokens
   - Prompt tokens
   - Total tokens
   - Reasoning tokens

Key features:
- Uses the OpenAI API (specifically the "o1-mini" model)
- Requires proper setup of API credentials
- Uses the `termcolor` library for colored console output
- Employs regular expressions for parsing API responses
- Saves each iteration of code improvement separately
- Provides detailed token usage statistics for each iteration

This script allows for progressive refinement of the generated code, potentially resulting in higher quality output. It's particularly useful for complex coding tasks that benefit from iterative improvement.

Both scripts use the OpenAI API and require proper setup of API credentials. They also use the `termcolor` library for colored console output and regular expressions for parsing the API responses.
