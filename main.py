# main.py

import os
import asyncio
import secrets
import time
from collections import defaultdict
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import aiofiles
import fitz  # PyMuPDF
import openai
import tiktoken
from decouple import config
from openai import AsyncOpenAI  # Add this import

# Load environment variables
openai.api_key = config('OPENAI_API_KEY')

# Initialize OpenAI client
client = AsyncOpenAI(api_key=openai.api_key)  # Add this line

# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Update rate limiting variables
user_requests = defaultdict(list)
user_requests_lock = asyncio.Lock()
RATE_LIMIT = 5  # Max requests
RATE_PERIOD = 60  # Time window in seconds

async def is_rate_limited(ip_address):
    current_time = time.time()
    async with user_requests_lock:
        request_times = user_requests[ip_address]
        # Filter out requests that are older than RATE_PERIOD
        request_times = [t for t in request_times if current_time - t < RATE_PERIOD]
        user_requests[ip_address] = request_times
        if len(request_times) >= RATE_LIMIT:
            return True
        user_requests[ip_address].append(current_time)
    return False

# Helper function to count tokens
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    client_ip = request.client.host
    if await is_rate_limited(client_ip):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Rate limit exceeded. Please try again later."},
            status_code=429
        )
    if file.content_type != 'application/pdf':
        return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid file type. Only PDFs are allowed."}, status_code=400)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        return templates.TemplateResponse("index.html", {"request": request, "error": "File too large. Maximum size is 10 MB."}, status_code=400)
    # Save the uploaded file
    temp_filename = f"{secrets.token_hex(8)}.pdf"
    async with aiofiles.open(temp_filename, 'wb') as out_file:
        await out_file.write(content)
    # Extract text from PDF
    text = ""
    try:
        with fitz.open(temp_filename) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid PDF file."}, status_code=400)
    finally:
        # Remove the temporary file
        try:
            os.remove(temp_filename)
        except Exception:
            pass
    # Summarize using OpenAI API
    try:
        summary = await summarize_text(text)
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Error summarizing text."}, status_code=500)
    return templates.TemplateResponse("result.html", {"request": request, "summary": summary})

async def summarize_text(text):
    max_token_per_chunk = 2000
    token_count = num_tokens_from_string(text)
    encoding = tiktoken.get_encoding("cl100k_base")
    if token_count <= max_token_per_chunk:
        prompt = f"Please provide a concise summary of the following text:\n\n{text}"
        response = await client.chat.completions.create(  # Updated method call
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        summary = response.choices[0].message.content.strip()  # Updated attribute access
        return summary
    else:
        # Split text into smaller chunks
        tokens = encoding.encode(text)
        chunks = [tokens[i:i + max_token_per_chunk] for i in range(0, len(tokens), max_token_per_chunk)]
        summaries = []
        for chunk in chunks:
            chunk_text = encoding.decode(chunk)
            prompt = f"Please summarize the following text:\n\n{chunk_text}"
            response = await client.chat.completions.create(  # Updated method call
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
            )
            chunk_summary = response.choices[0].message.content.strip()  # Updated attribute access
            summaries.append(chunk_summary)
        # Combine summaries
        combined_text = ' '.join(summaries)
        # Summarize the combined summaries
        final_prompt = f"Please provide a concise summary of the following text:\n\n{combined_text}"
        final_response = await client.chat.completions.create(  # Updated method call
            model="gpt-4o",
            messages=[{"role": "user", "content": final_prompt}],
        )
        final_summary = final_response.choices[0].message.content.strip()  # Updated attribute access
        return final_summary
# Run the application (use: uvicorn main:app --reload)

