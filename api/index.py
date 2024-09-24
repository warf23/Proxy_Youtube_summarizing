import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import validators
from fastapi.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi
import re
import logging
import requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Define a request body model
class TranscriptRequest(BaseModel):
  url: str
  language: str = "English"

# Language options
language_codes = {'English': 'en', 'Arabic': 'ar', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 
              'Italian': 'it', 'Portuguese': 'pt', 'Chinese': 'zh', 'Japanese': 'ja', 'Korean': 'ko'}

# Webshare proxy configuration
PROXY_HOST = os.getenv("PROXY_HOST")
PROXY_PORT = os.getenv("PROXY_PORT")
PROXY_USER = os.getenv("PROXY_USER")
PROXY_PASS = os.getenv("PROXY_PASS")

proxies = {
  "http": f"socks5://{PROXY_USER}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}",
  "https": f"socks5://{PROXY_USER}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}"
}

@app.get("/api")   
def read_root():
  return {"Hello": "World"}

def get_youtube_transcript(video_id, language='en'):
  try:
      transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
      return " ".join([entry['text'] for entry in transcript])
  except Exception as e:
      logger.error(f"Error fetching YouTube transcript: {str(e)}")
      return None

def get_youtube_video_id(url):
  video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
  if video_id_match:
      return video_id_match.group(1)
  return None

@app.post("/api/transcript")
async def get_transcript(request: TranscriptRequest):
  url = request.url
  language = request.language

  # Validate input
  if not validators.url(url):
      raise HTTPException(status_code=400, detail="Invalid URL")

  if language not in language_codes:
      raise HTTPException(status_code=400, detail="Invalid language")

  try:
      if "youtube.com" in url or "youtu.be" in url:
          video_id = get_youtube_video_id(url)
          if video_id:
              content = get_youtube_transcript(video_id, language_codes.get(language, 'en'))
              if content:
                  return {"transcript": content}
              else:
                  raise HTTPException(status_code=404, detail="Transcript not found")
          else:
              raise HTTPException(status_code=400, detail="Invalid YouTube URL")
      else:
          raise HTTPException(status_code=400, detail="URL is not a YouTube link")
  
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))
