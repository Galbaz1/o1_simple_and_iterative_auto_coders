import sys
import subprocess
import pkg_resources

# Check and install missing packages
required = {
    'fastapi',
    'PyMuPDF',
    'openai',
    'tiktoken',
    'uvicorn',
    'python-multipart',
    'jinja2',
    'aiofiles',
}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print(f"Installing missing packages: {missing}")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

import os
import logging
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from starlette.middleware.sessions import SessionMiddleware

import openai
import fitz  # PyMuPDF
from tiktoken import get_encoding
from typing import List

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Summarizer API", version="1.0")

# Set your OpenAI API key as an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key or openai.api_key == "YOUR_OPENAI_API_KEY":
    raise Exception("Please set your OpenAI API key in the environment variable 'OPENAI_API_KEY'.")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Session middleware (if needed for future features)
app.add_middleware(SessionMiddleware, secret_key="!secret")

def num_tokens_from_string(text: str, encoding_name: str = 'cl100k_base') -> int:
    """Returns the number of tokens in a text string."""
    encoding = get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def split_text(text: str, max_tokens: int) -> List[str]:
    """Split text into chunks not exceeding max_tokens."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence += '. '
        token_count = num_tokens_from_string(sentence)
        if current_tokens + token_count <= max_tokens:
            current_chunk += sentence
            current_tokens += token_count
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = token_count

    if current_tokens > max_tokens:
        raise ValueError("A single sentence exceeds the maximum token limit.")

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.post("/summarize_pdf/", response_class=HTMLResponse)
async def summarize_pdf(request: Request, file: UploadFile = File(...)):
    # Validate that the uploaded file is a PDF
    if file.content_type != 'application/pdf':
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a PDF file."
        )

    # Limit file size to prevent overloading the server (e.g., 10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, detail="File too large. Please upload a file smaller than 10MB."
        )

    try:
        # Read the file contents
        pdf_file = BytesIO(contents)

        # Use PyMuPDF (fitz) to read and extract text from the PDF
        doc = fitz.open(stream=pdf_file, filetype="pdf")

        # Extract text from each page
        text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += page.get_text()

        # Clean up the text
        text = text.strip().replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())

        if not text:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF doesn't contain extractable text."
            )

        # Calculate the total number of tokens in the text
        total_tokens = num_tokens_from_string(text)

        # Set model token limits
        max_model_tokens = 8192  # For GPT-4
        max_tokens_per_prompt = 6000  # Reserve tokens for the response and overhead

        # Split text into chunks if it's too long
        if total_tokens > max_tokens_per_prompt:
            chunks = split_text(text, max_tokens_per_prompt)
        else:
            chunks = [text]

        summaries = []
        for idx, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {idx+1}/{len(chunks)}")

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes documents.",
                },
                {
                    "role": "user",
                    "content": f"Please provide a detailed but concise summary of the following text:\n\n{chunk}",
                },
            ]

            attempt = 0
            max_retries = 3

            while attempt < max_retries:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=messages,
                        max_tokens=500,
                        temperature=0.5,
                    )
                    summary = response['choices'][0]['message']['content'].strip()
                    summaries.append(summary)
                    break  # Exit the retry loop
                except openai.error.OpenAIError as e:
                    logger.error(f"OpenAI API error: {e}")
                    attempt += 1
                    if attempt >= max_retries:
                        raise HTTPException(status_code=500, detail="Failed to get response from OpenAI API.")
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    raise HTTPException(status_code=500, detail="An unexpected error occurred.")

        # Combine summaries if there are multiple chunks
        if len(summaries) > 1:
            combined_summaries = ' '.join(summaries)
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that combines summaries.",
                },
                {
                    "role": "user",
                    "content": f"Please combine the following summaries into a single coherent summary:\n\n{combined_summaries}",
                },
            ]

            attempt = 0
            max_retries = 3

            while attempt < max_retries:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=messages,
                        max_tokens=500,
                        temperature=0.5,
                    )
                    final_summary = response['choices'][0]['message']['content'].strip()
                    break  # Exit the retry loop
                except openai.error.OpenAIError as e:
                    logger.error(f"OpenAI API error: {e}")
                    attempt += 1
                    if attempt >= max_retries:
                        raise HTTPException(status_code=500, detail="Failed to get response from OpenAI API.")
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    raise HTTPException(status_code=500, detail="An unexpected error occurred.")
        else:
            final_summary = summaries[0]

        # Determine the response format
        if request.headers.get('accept', '').find('text/html') != -1:
            # Render the summary in an HTML template
            return templates.TemplateResponse("summary.html", {"request": request, "summary": final_summary})
        else:
            # Return JSON response
            return JSONResponse(content={"summary": final_summary})

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
