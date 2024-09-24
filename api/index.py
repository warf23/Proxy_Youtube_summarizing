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
import random

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

# List of proxies
PROXY_LIST = [
  "38.154.227.167:5868:ltfazxrm:y69z38mzjh2y",
  "45.127.248.127:5128:ltfazxrm:y69z38mzjh2y",
  "207.244.217.165:6712:ltfazxrm:y69z38mzjh2y",
  "64.64.118.149:6732:ltfazxrm:y69z38mzjh2y",
  "167.160.180.203:6754:ltfazxrm:y69z38mzjh2y",
  "104.239.105.125:6655:ltfazxrm:y69z38mzjh2y",
  "198.105.101.92:5721:ltfazxrm:y69z38mzjh2y",
  "154.36.110.199:6853:ltfazxrm:y69z38mzjh2y",
  "204.44.69.89:6342:ltfazxrm:y69z38mzjh2y",
  "206.41.172.74:6634:ltfazxrm:y69z38mzjh2y"
]

def get_random_proxy():
  proxy = random.choice(PROXY_LIST)
  host, port, user, password = proxy.split(':')
  return {
      "http": f"socks5://{user}:{password}@{host}:{port}",
      "https": f"socks5://{user}:{password}@{host}:{port}"
  }

@app.get("/api")   
def read_root():
  return {"Hello": "World"}

def get_youtube_transcript(video_id, language='en'):
  try:
      proxies = get_random_proxy()
      transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language], proxies=proxies)
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
