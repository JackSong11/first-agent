import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("LLM_API_KEY"),
    base_url=os.environ.get("LLM_BASE_URL"),
)
