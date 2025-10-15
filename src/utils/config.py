import os
from dotenv import load_dotenv

# Load .env automatically
load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    DATA_PATH = os.getenv("DATA_PATH", "data/")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @staticmethod
    def summary():
        return {
            "data_path": Config.DATA_PATH,
            "log_level": Config.LOG_LEVEL,
            "api_keys": {
                "openai": bool(Config.OPENAI_API_KEY),
                "huggingface": bool(Config.HF_API_TOKEN)
            }
        }