from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = 'llama3.2'

class Website:
    def __init__(self, url, wait_time = 10):
        self.url = url

        options = Options()

        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument('--headless')

        # service = Service(PATH_TO_CHROME_DRIVER)
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        driver.implicitly_wait(wait_time)
        page_source = driver.page_source
        logging.info(f"Page source retrieved: {url}")
        driver.quit()

        soup = BeautifulSoup(page_source, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.get_text(separator="\n", strip=True)

def create_query(texts):
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant that extracts the most relevant keywords and key skills from the given job description. The job descriptions will be for machine learning engineers or data scientists. Return only the extracted keywords as a comma-separated list with no additional text."
        },
        {
            "role": "user", 
            "content": f"{texts}"
        },
    ]
    return messages

def main():
    url = "url" 
    website = Website(url)
    query = create_query(website.text)
    # Request non-streaming JSON to avoid chunked output
    response = requests.post(OLLAMA_API, headers=HEADERS, json={
        "model": MODEL,
        "messages": query,
        "stream": False
    })
    # Parse and format the response to a human-friendly list
    try:
        res_json = response.json()
        # Support both OpenAI-style and Ollama-style responses
        choices = res_json.get("choices")
        if choices:
            content = choices[0].get("message", {}).get("content", "")
        else:
            content = res_json.get("message", {}).get("content", "")
    except ValueError:
        logging.error("Response is not valid JSON")
        print(response.text)
        return
    # Format content into list
    print("Extracted Keywords:")
    if content:
        if "," in content:
            items = [kw.strip() for kw in content.split(",") if kw.strip()]
        else:
            items = [kw.strip() for kw in content.splitlines() if kw.strip()]
        for kw in items:
            print(f"- {kw}")
    else:
        print("No content found in response.")

if __name__ == '__main__':
    main()
