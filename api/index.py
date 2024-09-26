from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain

app = FastAPI()

class UrlInput(BaseModel):
  url: HttpUrl
  groq_api_key: str
  language: str

# Language options
language_codes = {
  'English': 'en', 'Arabic': 'ar', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
  'Italian': 'it', 'Portuguese': 'pt', 'Chinese': 'zh', 'Japanese': 'ja', 'Korean': 'ko'
}

# Define the prompt template
prompt_template = """
Content Summary Request

Language: {language}
Word Count: Approximately 300 words
Source: {text}

Objective:
Provide a concise yet comprehensive summary of the given content in the specified language. The summary should be accessible to readers unfamiliar with the original material.

Key Focus Areas:
1. Main points and central themes
2. Key arguments and supporting evidence
3. Significant conclusions or findings
4. Notable insights or implications
5. Methodologies used (if applicable)

Summary Guidelines:
- Begin with a brief introduction contextualizing the content.
- Organize information logically, using clear transitions between ideas.
- Prioritize the most crucial information from the source material.
- Maintain objectivity, avoiding personal interpretations or biases.
- Include relevant statistics, data points, or examples that substantiate main ideas.
- Conclude with the overarching message or significance of the content.

Additional Considerations:
- Identify any limitations, potential biases, or areas of controversy in the source material.
- Highlight any unique or innovative aspects of the content.
- If relevant, briefly mention the credibility or expertise of the source.

Formatting:
- Use clear, concise language appropriate for the target audience.
- Employ bullet points or numbered lists for clarity when presenting multiple related points.
- Include subheadings if it enhances readability and organization.

Note: Ensure the summary stands alone as an informative piece, providing value even without access to the original content.
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text", "language"])

def fetch_webpage(url: str) -> BeautifulSoup:
  try:
      response = requests.get(url)
      response.raise_for_status()
      return BeautifulSoup(response.content, 'html.parser')
  except requests.RequestException as e:
      raise HTTPException(status_code=400, detail=f"Error fetching the webpage: {str(e)}")

def extract_key_info(soup: BeautifulSoup) -> dict:
  for element in soup(['header', 'footer', 'nav', 'aside']):
      element.decompose()

  paragraphs = [p.text.strip() for p in soup.find_all('p') if p.text.strip()]
  headings = [h.text.strip() for h in soup.find_all(['h1', 'h2', 'h3']) if h.text.strip()]

  return {
      'paragraphs': paragraphs,
      'headings': headings
  }

def create_documents(extracted_info: dict) -> list:
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len,
  )

  all_text = "\n\n".join(extracted_info['headings'] + extracted_info['paragraphs'])
  chunks = text_splitter.split_text(all_text)
  documents = [Document(page_content=chunk) for chunk in chunks]

  return documents

@app.post("/extract_and_summarize")
async def extract_and_summarize(url_input: UrlInput):
  soup = fetch_webpage(url_input.url)
  extracted_info = extract_key_info(soup)
  documents = create_documents(extracted_info)

  # Initialize the language model
  model = ChatGroq(groq_api_key=url_input.groq_api_key, model_name="llama-3.2-90b-text-preview")
  chain = load_summarize_chain(model, chain_type="stuff", prompt=prompt)

  # Generate summary
  summary = chain.run(input_documents=documents, language=url_input.language)

  return {
      "num_documents": len(documents),
      "summary": summary
  }

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
