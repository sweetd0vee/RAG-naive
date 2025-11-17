import getpass
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load environment variables from .env file
load_dotenv()

hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")


class SimpleRAG:
    def __init__(self, pdf_path: str, persist_directory: str = "./chroma_db"):
        self.pdf_path = pdf_path
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.qa_chain = None
        self.embeddings = None
        self.llm = None

        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_and_process_document(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Загрузка и обработка документа"""

        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF файл не найден: {self.pdf_path}")

        self.logger.info("Загружаю документ...")

        # Загрузка PDF
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        if not documents:
            raise ValueError("Документ не содержит текста")

        # Разбиение на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        self.logger.info(f"Создано {len(chunks)} чанков из документа")

        # Фильтрация и валидация чанков
        valid_chunks = self._validate_chunks(chunks)
        self.logger.info(f"Используется {len(valid_chunks)} валидных чанков")

        # Создание эмбеддингов
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Создание векторной базы
        self._create_vectorstore(valid_chunks)

        return valid_chunks

    def _validate_chunks(self, chunks: List[Document]) -> List[Document]:
        """Валидация и очистка чанков"""
        valid_chunks = []

        for i, chunk in enumerate(chunks):
            # Проверка содержания
            if not chunk.page_content or not chunk.page_content.strip():
                continue

            # Очистка текста
            cleaned_content = self._clean_text(chunk.page_content)
            if len(cleaned_content) < 10:  # Минимальная длина
                continue

            # Обновление метаданных
            chunk.page_content = cleaned_content
            if 'page' not in chunk.metadata:
                chunk.metadata['page'] = 0
            chunk.metadata['source'] = os.path.basename(self.pdf_path)
            chunk.metadata['chunk_id'] = i

            valid_chunks.append(chunk)

        return valid_chunks

    def _clean_text(self, text: str) -> str:
        """Очистка текста"""
        import re

        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text)
        # Удаление специальных символов (опционально)
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        return text.strip()

    def _create_vectorstore(self, chunks: List[Document]):
        """Создание векторного хранилища"""
        # Очистка старой базы
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)

        self.logger.info("Создание векторной базы...")

        try:
            # Способ 1: Использование Chroma
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.logger.info("Векторная база создана успешно")

        except Exception as e:
            self.logger.warning(f"Chroma не сработал: {e}")
            self.logger.info("Пробую использовать FAISS...")
            self._create_vectorstore_faiss(chunks)

    def _create_vectorstore_faiss(self, chunks: List[Document]):
        """Резервный способ с FAISS"""
        try:
            from langchain_community.vectorstores import FAISS

            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            # Сохраняем FAISS индекс
            self.vectorstore.save_local(self.persist_directory)
            self.logger.info("Векторная база создана с использованием FAISS")

        except Exception as e:
            self.logger.error(f"FAISS также не сработал: {e}")
            raise RuntimeError("Не удалось создать векторное хранилище")

    def setup_qa_chain(self, model_name: str = "llama2", search_k: int = 3,
                       temperature: float = 0.1, max_tokens: int = 512):
        """Настройка цепочки вопрос-ответ"""

        if not self.vectorstore:
            raise ValueError("Сначала загрузите документы!")

        # Улучшенный промпт
        prompt_template = """Ты - помощник, отвечающий на вопросы на основе предоставленного контекста.

Контекст:
{context}

Вопрос: {question}

Инструкции:
1. Ответь строго на основе предоставленного контекста
2. Если ответа нет в контексте, скажи "В предоставленных документах нет информации для ответа на этот вопрос"
3. Будь точным и лаконичным
4. Используй маркированные списки если уместно
5. Не придумывай информацию, которой нет в контексте

Ответ:"""

        # Создание промпта с использованием правильного класса
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        try:
            # Инициализация LLM с параметрами
            self.llm = Ollama(
                model=model_name,
                temperature=temperature,
                num_predict=max_tokens
            )
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели {model_name}: {e}")
            raise

        # Настройка ретривера
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": search_k,
                "score_threshold": 0.3  # Более мягкий порог
            }
        )

        # Создание QA цепи
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": PROMPT,
            },
            return_source_documents=True,
            verbose=False  # Отключаем подробное логирование для продакшена
        )

        self.logger.info(f"Цепочка QA успешно настроена с моделью {model_name}")

    def ask_question(self, question: str) -> Dict[str, Any]:
        """Задать вопрос системе RAG"""
        if not self.qa_chain:
            raise ValueError("Сначала настройте QA цепочку!")

        self.logger.info(f"Вопрос: {question}")

        try:
            result = self.qa_chain({"query": question})

            # Форматированный вывод
            print(f"\n🤖 Ответ: {result['result']}")

            if result['source_documents']:
                print(f"\n📚 Использованные источники ({len(result['source_documents'])}:")
                for i, doc in enumerate(result['source_documents']):
                    page = doc.metadata.get('page', 'N/A')
                source = doc.metadata.get('source', 'Unknown')
                preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                print(f"   {i + 1}. Страница {page} | {source}")
                print(f"      📄 {preview}")
                print()
            else:
                print("\n⚠️  Источники не найдены")

            return {
                "answer": result['result'],
                "source_documents": result['source_documents'],
                "question": question
            }

        except Exception as e:
            self.logger.error(f"Ошибка при обработке вопроса: {e}")
            return {
                "answer": "Извините, произошла ошибка при обработке вашего вопроса.",
                "source_documents": [],
                "question": question,
                "error": str(e)
            }

    def search_similar(self, query: str, k: int = 3) -> List[Document]:
        """Поиск похожих документов без использования LLM"""
        if not self.vectorstore:
            raise ValueError("Векторное хранилище не инициализировано!")

        return self.vectorstore.similarity_search(query, k=k)

    def get_document_info(self) -> Dict[str, Any]:
        """Получение информации о загруженном документе"""
        if not self.vectorstore:
            return {"status": "Документы не загружены"}

        try:
            # Для Chroma
            if hasattr(self.vectorstore, '_collection'):
                collection = self.vectorstore._collection
                count = collection.count() if collection else 0
            # Для FAISS - приблизительный подсчет
            elif hasattr(self.vectorstore, 'index'):
                count = self.vectorstore.index.ntotal if hasattr(self.vectorstore.index, 'ntotal') else "Unknown"
            else:
                count = "Unknown"

            return {
                "document_path": self.pdf_path,
                "vector_store": self.persist_directory,
                "document_count": count,
                "embedding_model": "all-MiniLM-L6-v2",
                "status": "Загружено"
            }
        except Exception as e:
            self.logger.error(f"Ошибка получения информации: {e}")
            return {
                "document_path": self.pdf_path,
                "status": "Ошибка получения информации"
            }

    def clear_vectorstore(self):
        """Очистка векторного хранилища"""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            self.logger.info("Векторное хранилище очищено")
        self.vectorstore = None
        self.qa_chain = None
