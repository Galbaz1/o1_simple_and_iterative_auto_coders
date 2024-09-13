# Ensure to run this script in a virtual environment where you have admin privileges to install packages

import sys
import subprocess
import os
import logging
from io import BytesIO
from typing import List, Optional
from pathlib import Path
from datetime import datetime

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
    'itsdangerous',
    'python-decouple',
    'python-docx',
    'aiohttp',
    'fastapi_login',
    'passlib[bcrypt]',
    'sqlalchemy',
    'asyncpg',
    'aiosqlite',
    'ratelimit',
    'pydantic',
}

# Use importlib.metadata (Python 3.8+) or fallback to pkg_resources
try:
    if sys.version_info >= (3, 8):
        from importlib.metadata import distributions
    else:
        from importlib_metadata import distributions  # type: ignore

    installed = {dist.metadata['Name'].lower() for dist in distributions()}
except ImportError:
    import pkg_resources
    installed = {pkg.key for pkg in pkg_resources.working_set}

missing = required - installed

if missing:
    print(f"Installing missing packages: {missing}")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

import uvicorn
from fastapi import (
    FastAPI, File, HTTPException, UploadFile, Request, Depends,
    BackgroundTasks, Form
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi_login import LoginManager
from fastapi.security import OAuth2PasswordRequestForm
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, select
from sqlalchemy.ext.declarative import declarative_base
from ratelimit import limits
from fastapi.exceptions import RequestValidationError
from starlette.middleware.sessions import SessionMiddleware  # <-- Corrected import
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
import openai
import fitz  # PyMuPDF
import docx2txt
import tiktoken
from decouple import config
from secrets import token_urlsafe
from pydantic import BaseModel
import asyncio
import aiofiles

# Load environment variables
OPENAI_API_KEY = config('OPENAI_API_KEY')
SECRET_KEY = config('SECRET_KEY', default=token_urlsafe(32))
DATABASE_URL = config('DATABASE_URL', default='sqlite+aiosqlite:///./summaries.db')
ALLOWED_HOSTS = config('ALLOWED_HOSTS', default='*').split(',')
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Summarizer API", version="2.0")

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY
if not openai.api_key or openai.api_key == "YOUR_OPENAI_API_KEY":
    raise Exception("Please set your OpenAI API key in the environment variable 'OPENAI_API_KEY'.")

# Database setup using SQLAlchemy with AsyncIO
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

# User model for authentication
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), unique=True, index=True, nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    hashed_password = Column(String(150), nullable=False)

# Summary model to store summaries
class Summary(Base):
    __tablename__ = 'summaries'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    original_filename = Column(String(255))
    summary_text = Column(String, nullable=True)
    status = Column(String(50), nullable=False, default='pending')
    error = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create database tables on startup
@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Set up password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Set up login manager
manager = LoginManager(SECRET_KEY, token_url='/auth/token', use_cookie=True)
manager.cookie_name = "access_token"

@manager.user_loader()
async def load_user(username: str):
    async with async_session() as session:
        result = await session.execute(
            select(User).where(User.username == username)
        )
        user = result.scalars().first()
        return user

# Set up templates
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(Path(BASE_DIR, "templates")))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Session middleware for session management
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Rate limiting middleware
class RateLimiterMiddleware(BaseHTTPMiddleware):
    LIMIT_CALLS = 100
    WINDOW_SIZE = 60  # seconds

    def __init__(self, app):
        super().__init__(app)
        self.calls = {}
        self.lock = asyncio.Lock()

    async def dispatch(self, request, call_next):
        client_ip = request.client.host
        async with self.lock:
            if client_ip not in self.calls:
                self.calls[client_ip] = []
            current_time = asyncio.get_event_loop().time()
            self.calls[client_ip] = [call_time for call_time in self.calls[client_ip] if call_time > current_time - self.WINDOW_SIZE]
            if len(self.calls[client_ip]) >= self.LIMIT_CALLS:
                raise HTTPException(status_code=429, detail="Too Many Requests")
            self.calls[client_ip].append(current_time)
        response = await call_next(request)
        return response

app.add_middleware(RateLimiterMiddleware)

def num_tokens_from_string(text: str, model_name: str = 'gpt-4') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
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

async def extract_text_from_pdf(contents: bytes) -> str:
    """Extract text from PDF using PyMuPDF."""
    pdf_file = BytesIO(contents)
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        text += page.get_text()
    return text

async def extract_text_from_docx(contents: bytes) -> str:
    """Extract text from DOCX using python-docx."""
    file_stream = BytesIO(contents)
    text = docx2txt.process(file_stream)
    return text

async def extract_text_from_txt(contents: bytes) -> str:
    """Extract text from TXT file."""
    text = contents.decode('utf-8')
    return text

async def summarize_text(chunk: str) -> str:
    """Summarize text using OpenAI GPT-4."""
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
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=messages,
                max_tokens=500,
                temperature=0.5,
            )
            summary = response['choices'][0]['message']['content'].strip()
            return summary
        except openai.error.RateLimitError as e:
            logger.error(f"OpenAI API rate limit error: {e}")
            attempt += 1
            await asyncio.sleep(2 ** attempt)
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            attempt += 1
            await asyncio.sleep(2 ** attempt)
            if attempt >= max_retries:
                raise HTTPException(status_code=500, detail="Failed to get response from OpenAI API.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "detail": exc.detail},
        status_code=exc.status_code,
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "detail": exc.errors()},
        status_code=400
    )

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = await manager.get_current_user(request)
    return templates.TemplateResponse("upload_form.html", {"request": request, "user": user})

@app.post("/upload/", response_class=HTMLResponse)
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    summary_level: str = Form("detailed"),
    language: Optional[str] = Form("en"),
    current_user=Depends(manager)
):
    # Validate file type
    if file.content_type not in ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']:
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a PDF, DOCX, or TXT file."
        )

    # Limit file size (e.g., 20MB)
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, detail="File too large. Please upload a file smaller than 20MB."
        )

    # Save the file asynchronously
    upload_dir = Path(BASE_DIR, "uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / f"{token_urlsafe(8)}_{file.filename}"
    async with aiofiles.open(file_path, 'wb') as out_file:
        await out_file.write(contents)

    # Create a summary record with status 'pending'
    async with async_session() as session:
        new_summary = Summary(
            user_id=current_user.id,
            original_filename=file.filename,
            status='pending',
        )
        session.add(new_summary)
        await session.commit()
        await session.refresh(new_summary)
        summary_id = new_summary.id

    # Start background task for processing
    background_tasks.add_task(
        process_file, contents, file.content_type, file.filename, summary_level, language, summary_id
    )

    return RedirectResponse(url=f"/summary/{summary_id}", status_code=303)

async def process_file(contents, content_type, filename, summary_level, language, summary_id):
    try:
        if content_type == 'application/pdf':
            text = await extract_text_from_pdf(contents)
        elif content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            text = await extract_text_from_docx(contents)
        elif content_type == 'text/plain':
            text = await extract_text_from_txt(contents)
        else:
            raise Exception("Unsupported file type.")

        # Clean up the text
        text = text.strip().replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())

        if not text:
            raise Exception("The uploaded file doesn't contain extractable text.")

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

        # Summarize each chunk concurrently
        summaries = await asyncio.gather(*(summarize_text(chunk) for chunk in chunks))

        # Combine summaries if there are multiple chunks
        if len(summaries) > 1:
            combined_summaries = ' '.join(summaries)
            final_summary = await summarize_text(combined_summaries)
        else:
            final_summary = summaries[0]

        # Update summary record in database
        async with async_session() as session:
            summary_record = await session.get(Summary, summary_id)
            if summary_record:
                summary_record.summary_text = final_summary
                summary_record.status = 'completed'
                await session.commit()
            else:
                logger.error(f"Summary record with id {summary_id} not found.")

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        # Update summary record with error
        async with async_session() as session:
            summary_record = await session.get(Summary, summary_id)
            if summary_record:
                summary_record.status = 'error'
                summary_record.error = str(e)
                await session.commit()

@app.get("/summary/{summary_id}", response_class=HTMLResponse)
async def view_summary(request: Request, summary_id: int, current_user=Depends(manager)):
    async with async_session() as session:
        summary_record = await session.get(Summary, summary_id)
        if not summary_record:
            raise HTTPException(status_code=404, detail="Summary not found.")
        if summary_record.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to view this summary.")

    return templates.TemplateResponse(
        "summary.html",
        {"request": request, "summary": summary_record}
    )

@app.get("/summaries/", response_class=HTMLResponse)
async def list_summaries(request: Request, current_user=Depends(manager)):
    async with async_session() as session:
        result = await session.execute(
            select(Summary).where(Summary.user_id == current_user.id).order_by(Summary.created_at.desc())
        )
        summaries = result.scalars().all()
    return templates.TemplateResponse(
        "summaries.html",
        {"request": request, "summaries": summaries}
    )

# Authentication routes
@app.post('/auth/register')
async def register(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):
    async with async_session() as session:
        result = await session.execute(
            select(User).where((User.username == username) | (User.email == email))
        )
        existing_user = result.scalars().first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username or email already registered.")
        hashed_password = pwd_context.hash(password)
        user = User(username=username, email=email, hashed_password=hashed_password)
        session.add(user)
        await session.commit()
    return JSONResponse(content={"message": "User registered successfully."})

@app.post('/auth/token')
async def login(data: OAuth2PasswordRequestForm = Depends()):
    username = data.username
    password = data.password
    async with async_session() as session:
        result = await session.execute(
            select(User).where(User.username == username)
        )
        user = result.scalars().first()

    if not user or not pwd_context.verify(password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials.")
    access_token = manager.create_access_token(
        data={"sub": username},
        expires=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    response = JSONResponse(content={"access_token": access_token, "token_type": "bearer"})
    manager.set_cookie(response, access_token)
    return response

@app.get('/auth/logout')
async def logout(request: Request):
    response = RedirectResponse(url="/")
    response.delete_cookie(key=manager.cookie_name)
    return response

# Main entry point
if __name__ == "__main__":
    uvicorn.run("improved_script:app", host="0.0.0.0", port=8000, reload=True)
