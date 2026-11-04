"""Gemini client helper for Lukas's AI Agent."""

from __future__ import annotations #For annotations to help with type hints and other features.

import logging #core python library for logging messages and errors.
import os #core python library for interacting with the operating system like accessing environment variables.
import time #Python library to handle time-related operations like delays.
import warnings #Python library to handle warnings and suppress them.
from pathlib import Path #Python library to handle file paths and directories.

from google import genai #Googles AI API library specifically their Gemini python SDK
from google.genai import errors, types #SDK configuratio classes

