# main.py

import sys
import subprocess
import pkg_resources

# Check and install missing packages
required = {'fastapi', 'PyPDF2', 'openai', 'tiktoken', 'uvicorn'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print(f"Installing missing packages: {missing}")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

import os
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
import openai
import PyPDF2

app = FastAPI()

# Set your OpenAI API key as an environment variable or replace with your API key
openai.api_key = os.getenv("OPENAI_API_KEY") or "YOUR_OPENAI_API_KEY"

def num_tokens_from_string(string: str, encoding_name: str = 'gpt2') -> int:
    """Returns the number of tokens in a text string."""
    import tiktoken  # Ensure tiktoken is installed
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

@app.post("/summarize_pdf/")
async def summarize_pdf(file: UploadFile = File(...)):
    # Validate that the uploaded file is a PDF
    if file.content_type != 'application/pdf':
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a PDF file."
        )

    try:
        # Read the file contents
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(BytesIO(contents))

        # Extract text from each page of the PDF
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() or ""

        # Calculate the total number of tokens in the text
        total_tokens = num_tokens_from_string(text)

        # Set model token limits
        max_model_tokens = 8192  # Adjust if using a different model with a different token limit
        max_tokens_per_prompt = 6000  # Reserve tokens for the response and other overhead

        # Split text into chunks if it's too long
        if total_tokens > max_tokens_per_prompt:
            chunks = []
            current_chunk = ""
            current_tokens = 0
            sentences = text.split('. ')  # Simple sentence tokenizer

            for sentence in sentences:
                token_count = num_tokens_from_string(sentence)
                if current_tokens + token_count <= max_tokens_per_prompt:
                    current_chunk += sentence + '. '
                    current_tokens += token_count
                else:
                    chunks.append(current_chunk)
                    current_chunk = sentence + '. '
                    current_tokens = token_count

            if current_chunk:
                chunks.append(current_chunk)
        else:
            chunks = [text]

        # Summarize each chunk
        summaries = []

        for chunk in chunks:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes documents.",
                },
                {
                    "role": "user",
                    "content": f"Please provide a concise summary of the following text:\n\n{chunk}",
                },
            ]

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=500,
                temperature=0.5,
            )

            summary = response['choices'][0]['message']['content'].strip()
            summaries.append(summary)

        # Combine summaries if there are multiple chunks
        if len(summaries) > 1:
            combined_summaries = ' '.join(summaries)
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes text.",
                },
                {
                    "role": "user",
                    "content": f"Please provide a concise summary of the following text:\n\n{combined_summaries}",
                },
            ]

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=500,
                temperature=0.5,
            )

            final_summary = response['choices'][0]['message']['content'].strip()
        else:
            final_summary = summaries[0]

        return {"summary": final_summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
