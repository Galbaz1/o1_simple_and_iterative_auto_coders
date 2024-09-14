import os
import asyncio
import secrets
import time
from collections import defaultdict

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import aiofiles
import fitz  # PyMuPDF
from openai import AsyncOpenAI  # Add this import
import tiktoken
from decouple import config

# Load environment variables
openai_api_key = config('OPENAI_API_KEY')

# Initialize AsyncOpenAI client
client = AsyncOpenAI(api_key=openai_api_key)

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting variables
user_requests = defaultdict(list)
user_requests_lock = asyncio.Lock()
RATE_LIMIT = 5  # Max requests
RATE_PERIOD = 60  # Time window in seconds

async def is_rate_limited(ip_address):
    current_time = time.time()
    async with user_requests_lock:
        request_times = user_requests[ip_address]
        request_times = [t for t in request_times if current_time - t < RATE_PERIOD]
        user_requests[ip_address] = request_times
        if len(request_times) >= RATE_LIMIT:
            return True
        user_requests[ip_address].append(current_time)
    return False

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

@app.post("/api/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    client_ip = request.client.host
    if await is_rate_limited(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10 MB.")

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
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid PDF file.")
    finally:
        try:
            os.remove(temp_filename)
        except Exception:
            pass

    # Summarize using OpenAI API
    try:
        summary = await summarize_text(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing text: {str(e)}")

    return {"summary": summary}

async def summarize_text(text):
    max_token_per_chunk = 2000
    encoding = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoding.encode(text))

    if token_count <= max_token_per_chunk:
        prompt = f"Please provide a concise summary of the following text:\n\n{text}"
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        summary = response.choices[0].message.content.strip()
        return summary
    else:
        # Split text into smaller chunks
        tokens = encoding.encode(text)
        chunks = [tokens[i:i + max_token_per_chunk] for i in range(0, len(tokens), max_token_per_chunk)]
        summaries = []
        for chunk in chunks:
            chunk_text = encoding.decode(chunk)
            prompt = f"Please summarize the following text:\n\n{chunk_text}"
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
            )
            chunk_summary = response.choices[0].message.content.strip()
            summaries.append(chunk_summary)
        # Combine summaries
        combined_text = ' '.join(summaries)
        # Summarize the combined summaries
        final_prompt = f"Please provide a concise summary of the following text:\n\n{combined_text}"
        final_response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": final_prompt}],
        )
        final_summary = final_response.choices[0].message.content.strip()
        return final_summary

