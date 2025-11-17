import getpass
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from langchain.chains import RetrievalQA
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama  # или другая LLM
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass(
    "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"
)
from dotenv import load_dotenv

load_dotenv()
