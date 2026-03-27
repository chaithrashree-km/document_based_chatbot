import uuid
import os
import csv
import logging
from unstructured.partition.auto import partition
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.db.Vector_Database import VectorDatabase
from app.Config import Config
from app.utils.Text_Helper import clean_text

class Upload:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    config = Config()
    database = VectorDatabase()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".xls", ".xlsx",".jpg", ".jpeg", ".png", ".docx", ".pptx", ".html", ".md"}

    def _extract(self, file_path: str):
        try:
            elements = partition(filename=file_path)
        except Exception as e:
            logging.error(f"Failed to partition {file_path}: {e}")
            _, ext = os.path.splitext(file_path)

            if ext.lower() == ".csv":
               return self._extract_csv_fallback(file_path)
            return []

        pages: dict[int, list[str]] = {}
        for el in elements:
            page_num = (el.metadata.page_number or 1) if el.metadata else 1
            text = str(el).strip()
            if text:
                pages.setdefault(page_num, []).append(text)

        result = []
        for page_num, texts in sorted(pages.items()):
            combined_text = "\n".join(texts)
            result.append((combined_text, page_num))
        return result

    def _extract_csv_fallback(self, file_path: str):
       try:
          rows = []
          with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                line = ", ".join(
                    f"{k}: {v}" for k, v in row.items() if v and v.strip()
                )
                if line:
                    rows.append(line)

          if not rows:
            logging.warning(f"CSV fallback: no content in {file_path}")
            return []

          logging.info(f"CSV fallback extracted {len(rows)} rows from {file_path}")
          return [("\n".join(rows), 1)]

       except Exception as e:
          logging.error(f"CSV fallback failed for {file_path}: {e}")
          return []

    def upload_documents(self, path: str):
        BATCH_SIZE = 100
        batch_texts = []
        batch_sources = []

        if os.path.isfile(path):
            logging.info(f"{path} is a file.")
            files = [path]
        elif os.path.isdir(path):
            logging.info(f"{path} is a directory.")
            files = [
                os.path.join(root, f)
                for root, _, filenames in os.walk(path)
                for f in filenames
            ]
            logging.info(f"Files found: {len(files)}")
        else:
            logging.error(f"Path does not exist: {path}")
            return {"message": "Invalid path provided."}

        logging.info(f"Files to process: {len(files)}")

        embedding_dimension = self.model.get_sentence_embedding_dimension()
        self.database.create_collection(embedding_dimension)

        def flush_batch():
            if not batch_texts:
                return
            embeddings = self.model.encode(batch_texts).tolist()
            points = [
                {
                    "id": str(uuid.uuid4()),
                    "vector": embeddings[i],
                    "payload": {
                        "text": batch_texts[i],
                        "source": batch_sources[i]["source"],
                        "page": batch_sources[i]["page"],
                        "folder": batch_sources[i]["folder"],
                    },
                }
                for i in range(len(batch_texts))
            ]
            self.database.client.upsert(
                collection_name=self.config.COLLECTION_NAME,
                points=points
            )
            logging.info(f"Inserted batch of {len(batch_texts)} chunks")
            batch_texts.clear()
            batch_sources.clear()

        for file_path in files:
            _, ext = os.path.splitext(file_path)
            if ext.lower() not in self.SUPPORTED_EXTENSIONS:
                logging.info(f"Skipping unsupported file: {file_path}")
                continue

            file = os.path.basename(file_path)
            folder = os.path.dirname(file_path)
            logging.info(f"Extracting: {file_path}")

            pages = self._extract(file_path)
            if not pages:
                logging.warning(f"No content extracted from {file_path}")
                continue

            for page_text, page_number in pages:
                cleaned = clean_text(page_text)
                for chunk in self.splitter.split_text(cleaned):
                    batch_texts.append(chunk)
                    batch_sources.append({
                        "source": file,
                        "page": page_number,
                        "folder": folder
                    })
                    if len(batch_texts) == BATCH_SIZE:
                        flush_batch()

        flush_batch()  
        return {"message": "Chunks ingested successfully."}
