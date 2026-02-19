import os

from dotenv import load_dotenv

load_dotenv()

class Settings:
    ENVIRONMENT = os.getenv("ENVIRONMENT")
    AZURE_WEBJOBS_STORAGE = os.getenv("AzureWebJobsStorage")  
    POSTGRES_CONN_STR = os.getenv("POSTGRES_CONN_STR")
    UAMI_CLIENT_ID = os.getenv("UAMI_CLIENT_ID")  # User-assigned managed identity
    EMAIL_STORAGE_ACCOUNT_NAME = os.getenv("EMAIL_STORAGE_ACCOUNT_NAME")
    EMAIL_STORAGE_ACCOUNT_KEY = os.getenv("EMAIL_STORAGE_ACCOUNT_KEY")
    EMAIL_STORAGE_ACCOUNT_URL = os.getenv("EMAIL_STORAGE_ACCOUNT_URL")
    MODELS_BLOB_CONTAINER_NAME = "models"
    ENABLE_SLACK = os.getenv("ENABLE_SLACK", "False")
    SLACK_TOKEN = os.getenv("SLACK_TOKEN")