import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import validators
from fastapi.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi
import re
import logging
import random
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredURLLoader

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

# Define request body model
class SummarizeRequest(BaseModel):
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

# Define the prompt template
prompt_template = PromptTemplate(
  input_variables=["text", "language"],
  template="""
Please provide a concise and informative summary in the language {language} of the following content. The summary should be approximately 300 words and should highlight the main points, key arguments, and any significant conclusions or insights presented in the content. Ensure that the summary is clear and easy to understand for someone who has not accessed the original content.

Content:
{text}
"""
)

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

def Url_website(url):
  loader = UnstructuredURLLoader(
      urls=[url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0"}
  )
  documents = loader.load()
  return " ".join([doc.page_content for doc in documents])

def summarize_with_groq(content, language):
  try:
      groq_api_key = os.getenv("GROQ_API_KEY")
      if not groq_api_key:
          raise ValueError("Groq API key not found in environment variables")

      llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it")

      text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
      
      if isinstance(content, str):
          texts = text_splitter.split_text(content)
          docs = [Document(page_content=t) for t in texts]
      elif isinstance(content, list) and all(isinstance(doc, Document) for doc in content):
          docs = content
      else:
          raise ValueError("Invalid content type. Expected string or list of Documents.")

      chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt_template)
      summary = chain.run(input_documents=docs, language=language)

      return summary
  except Exception as e:
      logger.error(f"Error summarizing content: {str(e)}")
      raise

@app.post("/api/summarize")
async def get_transcript_and_summarize(request: SummarizeRequest):
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
              transcript = get_youtube_transcript(video_id, language_codes.get(language, 'en'))
              if transcript:
                  summary = summarize_with_groq(transcript, language)
                  return {"summary": summary}
              else:
                  raise HTTPException(status_code=404, detail="Transcript not found")
          else:
              raise HTTPException(status_code=400, detail="Invalid YouTube URL")
      else:
          text = Url_website(url)
          summary = summarize_with_groq(text, language)
          return {"summary": summary}
  
  except Exception as e:
      logger.error(f"Error in get_transcript_and_summarize: {str(e)}")
      raise HTTPException(status_code=500, detail=str(e))

# Run with Uvicorn
if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
