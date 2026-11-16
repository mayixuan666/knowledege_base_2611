import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

@dataclass
class MineruConfig():
    base_url:str
    api_token:str
mineru_config = MineruConfig(
    base_url=os.getenv("MINERU_BASE_URL",""),  #后边双引号是默认值，读不到前边的就走后边默认值
    api_token=os.getenv("MINERU_API_TOKEN","")
)