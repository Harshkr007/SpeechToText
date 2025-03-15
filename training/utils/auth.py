from huggingface_hub import login
import os
from dotenv import load_dotenv

def setup_huggingface_auth():
    """Setup Hugging Face authentication"""
    load_dotenv()
    
    # Get token from environment variable or prompt user
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("Please enter your Hugging Face token (from https://huggingface.co/settings/tokens):")
        token = input().strip()
    
    try:
        login(token)
        print("Successfully logged in to Hugging Face")
    except Exception as e:
        raise Exception(f"Failed to login to Hugging Face: {str(e)}")
