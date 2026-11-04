"""Gemini client helper for Lukas's AI Agent."""

from __future__ import annotations #For annotations to help with type hints and other features.

import logging #core python library for logging messages and errors.
import os #core python library for interacting with the operating system like accessing environment variables.
import time #Python library to handle time-related operations like delays.
import warnings #Python library to handle warnings and suppress them.
from pathlib import Path #Python library to handle file paths and directories.

from google import genai #Googles AI API library specifically their Gemini python SDK (This is an external library, hence it should be added to requirements.txt)
from google.genai import errors, types #SDK configuratio classes

# Suppress noisy automatic-function-calling notices from google-genai.
logging.getLogger("google_genai.models").setLevel(logging.ERROR) #.getlogger retrives a logger for google genai models and sets the level to ERROR to suppress the noisy automatic-function-calling notices.
#Setting the level to Error suppresses any lower serverity messages like INFO and WARNING
warnings.filterwarnings("ignore", message=".*automatic function calling.*") #This tells python what to do with warning messages. A warning is different from an error
#where the code still runs but there is a potential issue that should be addressed. The above filterwarning is used to ignore any automatic function calling notices.

MODEL_NAME = "gemini-3.5-flash-lite" #The model name is the name of the model we are using to generate the response.

REPO_ROOT = Path(__file__).resolve().parent.parent #The repo root is the root of the repository.


def _load_dotenv() -> None: #Function responsible for loadiing the environment variables from the .env file, returning nothing.
    """Load environment variables from .env files without overwriting existing vars."""
    candidate_paths = [
        REPO_ROOT / ".env",
        REPO_ROOT.parent / ".env",
    ]
    for env_path in candidate_paths: #For each path in the candidate paths
        if not env_path.exists(): #If that path does not exist, continue with the next path.
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines(): #read_text reads the files content into a string and splitlines splits the string into multiple lines
            line = line.strip()
            #For each of the lines, we first strip them, then if the line if empty of starts with a # (representing a comment) or doesnt contain an =, we skip it
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=") #partition splits a string at the first occurent of a =, so the key is the variable name and value is its value
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                os.environ.setdefault(key, value)


_load_dotenv() #Module initialization function to load the environment variables from the .env file.

