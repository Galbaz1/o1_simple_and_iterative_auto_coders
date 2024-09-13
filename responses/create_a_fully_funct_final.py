# main.py

import os
import asyncio
import secrets
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File, Depends, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi_login import LoginManager
from passlib.context import CryptContext
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.future import select
import aiofiles
import fitz  # PyMuPDF
import openai
import tiktoken
from decouple import config

# Load environment variables
openai.api_key = config('OPENAI_API_KEY')
SECRET = config('SECRET_KEY', default='your-secret-key')
DATABASE_URL = config('DATABASE_URL', default='sqlite+aiosqlite:///./test.db')

# Define lifespan function
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create the database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # You can add any cleanup code here

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Set up authentication
manager = LoginManager(SECRET, token_url='/auth/token', use_cookie=True)
manager.cookie_name = "access_token"

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Database setup
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)

# User model
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), unique=True, index=True)
    hashed_password = Column(String(256))

# Dependency to get DB session
async def get_db():
    async with SessionLocal() as session:
        yield session

# Authentication functions
@manager.user_loader()
async def load_user(username: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(User).where(User.username == username)
    )
    user = result.scalars().first()
    return user

# Rate limiting variables and lock
user_requests = defaultdict(list)
user_requests_lock = asyncio.Lock()
RATE_LIMIT = 5  # Max requests
RATE_PERIOD = 60  # Time window in seconds

async def is_rate_limited(user_id):
    current_time = time.time()
    async with user_requests_lock:
        request_times = user_requests[user_id]
        # Filter out requests that are older than RATE_PERIOD
        request_times = [t for t in request_times if current_time - t < RATE_PERIOD]
        user_requests[user_id] = request_times
        if len(request_times) >= RATE_LIMIT:
            return True
        user_requests[user_id].append(current_time)
    return False

# Helper function to count tokens
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

# Routes
@app.post('/auth/register')
async def register(username: str = Form(...), password: str = Form(...), db: AsyncSession = Depends(get_db)):
    # Check if username already exists
    result = await db.execute(select(User).where(User.username == username))
    existing_user = result.scalars().first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_password = pwd_context.hash(password)
    new_user = User(username=username, hashed_password=hashed_password)
    db.add(new_user)
    await db.commit()
    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

@app.post('/auth/token')
async def login(request: Request, username: str = Form(...), password: str = Form(...), db: AsyncSession = Depends(get_db)):
    user = await load_user(username, db)
    if not user or not pwd_context.verify(password, user.hashed_password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"}, status_code=400)
    access_token = manager.create_access_token(data={'sub': user.username})
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    manager.set_cookie(response, access_token)
    return response

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie(manager.cookie_name)
    return response

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request, user=Depends(manager)):
    return templates.TemplateResponse("index.html", {"request": request, "user": user})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...), user=Depends(manager)):
    if await is_rate_limited(user.username):
        return templates.TemplateResponse("index.html", {"request": request, "user": user, "error": "Rate limit exceeded. Please try again later."}, status_code=429)
    if file.content_type != 'application/pdf':
        return templates.TemplateResponse("index.html", {"request": request, "user": user, "error": "Invalid file type. Only PDFs are allowed."}, status_code=400)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        return templates.TemplateResponse("index.html", {"request": request, "user": user, "error": "File too large. Maximum size is 10 MB."}, status_code=400)
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
        return templates.TemplateResponse("index.html", {"request": request, "user": user, "error": "Invalid PDF file."}, status_code=400)
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
        return templates.TemplateResponse("index.html", {"request": request, "user": user, "error": "Error summarizing text."}, status_code=500)
    return templates.TemplateResponse("result.html", {"request": request, "summary": summary, "user": user})

async def summarize_text(text):
    max_token_per_chunk = 2000
    token_count = num_tokens_from_string(text)
    encoding = tiktoken.get_encoding("cl100k_base")
    if token_count <= max_token_per_chunk:
        prompt = f"Please provide a concise summary of the following text:\n\n{text}"
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        summary = response['choices'][0]['message']['content'].strip()
        return summary
    else:
        # Split text into smaller chunks
        tokens = encoding.encode(text)
        chunks = [tokens[i:i + max_token_per_chunk] for i in range(0, len(tokens), max_token_per_chunk)]
        summaries = []
        for chunk in chunks:
            chunk_text = encoding.decode(chunk)
            prompt = f"Please summarize the following text:\n\n{chunk_text}"
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
            )
            chunk_summary = response['choices'][0]['message']['content'].strip()
            summaries.append(chunk_summary)
        # Combine summaries
        combined_text = ' '.join(summaries)
        # Summarize the combined summaries
        final_prompt = f"Please provide a concise summary of the following text:\n\n{combined_text}"
        final_response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": final_prompt}],
        )
        final_summary = final_response['choices'][0]['message']['content'].strip()
        return final_summary

# Run the application (use: uvicorn main:app --reload)
