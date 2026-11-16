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

_client: genai.Client | None = None #initializing the client to None, which will be set to a Gemini client object later.


def _get_client() -> genai.Client:
    """Return a singleton Gemini client configured from environment variables."""
    global _client #This is a global variable that is used to store the client object. Hence any changes to it will be reflected globally to the above definittion too
    if _client is None:
        _load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") #If the client is not found, get the API key
        if not api_key:
            raise RuntimeError(
                "Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment or .env file."
            )
        _client = genai.Client(api_key=api_key) #set the client to the Gemini client object using the API key.
    return _client


def query_llm(prompt: str, temperature: float = 0.0) -> str:
    """Send a prompt to Gemini and return the raw string response."""
    client = _get_client() #get the client object using the above function
    config = types.GenerateContentConfig( #creating a config object to configure the model's behavior
        temperature=temperature,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True), #disable automatic function calling, which is calling other tools like a weather API

    )

    delay = 3.0 #Add delay to the response to avoid overwhelming the model.
    for attempt in range(6): #We attempt 6 times to generate the response, if it fails, we raise an error.
        try:
            response = client.models.generate_content( #Generate the response frzom the model using the prompt, model and config
                model=MODEL_NAME,
                contents=prompt,
                config=config,
            )
            return (response.text or "").strip() #Return the response as a string, stripping any whitespace around it.
        except errors.APIError as exc: #Catch an error if any occur
            status_code = getattr(exc, "code", None) #get the error status code
            if status_code in (429, 503) and attempt < 5: #If the error code is 429 (rate limit error) or 503 (service unavailable), we sleep for the delay and try again.
                time.sleep(delay)
                delay *= 2.0
                continue
            raise #if we are past the 5 attempts, we raise an error and return an empty string.
    return ""
