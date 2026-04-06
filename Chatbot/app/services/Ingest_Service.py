import uuid
import os
import csv
import logging
import hashlib  
import fitz
import pdfplumber
import numpy as np
import easyocr
from io import BytesIO
from PIL import Image
from unstructured.partition.auto import partition
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.db.Vector_Database import VectorDatabase
from app.Config import Config
from app.utils.Text_Helper import clean_text

class Upload:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    config = Config()
    database = VectorDatabase()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".xls", ".xlsx", ".jpg", ".jpeg", ".png", ".docx", ".pptx", ".html", ".md"}

    _ocr_reader = None

    @classmethod
    def _get_ocr_reader(cls):
        if cls._ocr_reader is None:
            logging.info("Loading EasyOCR model (CPU)...")
            cls._ocr_reader = easyocr.Reader(["en"], gpu=False)
            logging.info("EasyOCR ready.")
        return cls._ocr_reader

    def _analyse_page(self, fitz_page, plumber_page) -> dict:
        raw_text = fitz_page.get_text("text").strip()
        has_text = len(raw_text) > 80
        has_tables = False
        try:
            has_tables = bool(plumber_page.extract_tables())
        except Exception:
            pass
        has_images = bool(fitz_page.get_images(full=True))
        drawings = fitz_page.get_drawings()
        has_charts = len(drawings) > 10
        return {
            "has_text": has_text,
            "has_tables": has_tables,
            "has_images": has_images,
            "has_charts": has_charts,
            "text": raw_text,
            "drawings": drawings,
        }

    @staticmethod
    def _table_to_markdown(table: list) -> str:
        if not table:
            return ""
        header = [str(c).strip() if c else "" for c in table[0]]
        rows = table[1:]
        lines = []
        if any(header):
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join("---" for _ in header) + " |")
        for row in rows:
            cells = [str(c).strip() if c else "" for c in row]
            if any(cells):
                lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    @staticmethod
    def _rasterize_region(fitz_page, clip_rect=None, dpi: int = 200) -> np.ndarray:
        scale = dpi / 72
        mat = fitz.Matrix(scale, scale)
        pix = fitz_page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            arr = arr[:, :, :3]
        return arr

    def _ocr_array(self, img_array: np.ndarray) -> str:
        ocr = self._get_ocr_reader()
        results = ocr.readtext(img_array, detail=0, paragraph=True)
        return " ".join(results).strip()

    def _extract_chart_text(self, fitz_page, drawings: list) -> str:
        if not drawings:
            return ""
        rects = [fitz.Rect(d["rect"]) for d in drawings if d.get("rect")]
        if not rects:
            return ""
        union_rect = rects[0]
        for r in rects[1:]:
            union_rect = union_rect | r
        padding = 40
        page_rect = fitz_page.rect
        expanded = fitz.Rect(
            max(0, union_rect.x0 - padding),
            max(0, union_rect.y0 - padding),
            min(page_rect.width, union_rect.x1 + padding),
            min(page_rect.height, union_rect.y1 + padding),
        )
        text_in_region = fitz_page.get_text("text", clip=expanded).strip()
        img_array = self._rasterize_region(fitz_page, clip_rect=expanded, dpi=200)
        ocr_text = self._ocr_array(img_array)
        parts = []
        if text_in_region:
            parts.append(f"[Chart labels from text layer]\n{text_in_region}")
        if ocr_text and ocr_text not in text_in_region:
            parts.append(f"[Chart labels from OCR]\n{ocr_text}")
        return "\n".join(parts)

    def _extract_image_text(self, doc, fitz_page, image_list: list) -> list:
        parts = []
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                pil_image = Image.open(BytesIO(img_bytes)).convert("RGB")
                img_array = np.array(pil_image)
                ocr_text = self._ocr_array(img_array)
                if ocr_text:
                    parts.append(f"[Embedded image {img_idx + 1} OCR]\n{ocr_text}")
            except Exception as e:
                logging.warning(f"  Image {img_idx + 1} OCR failed: {e}")
        return parts

    def _extract_complex_pdf(self, file_path: str) -> list:
        results = []
        try:
            doc = fitz.open(file_path)
            plumber = pdfplumber.open(file_path)
        except Exception as e:
            logging.error(f"Could not open PDF {file_path}: {e}")
            return []
        for page_index in range(len(doc)):
            page_num = page_index + 1
            fitz_page = doc[page_index]
            plumber_page = plumber.pages[page_index]
            page_parts = []
            info = self._analyse_page(fitz_page, plumber_page)
            logging.info(
                f"  Page {page_num}: text={info['has_text']} "
                f"tables={info['has_tables']} images={info['has_images']} "
                f"charts={info['has_charts']}"
            )
            if info["has_text"]:
                page_parts.append(f"[Text]\n{info['text']}")
            if info["has_tables"]:
                try:
                    for t_idx, table in enumerate(plumber_page.extract_tables()):
                        md = self._table_to_markdown(table)
                        if md:
                            page_parts.append(f"[Table {t_idx + 1}]\n{md}")
                except Exception as e:
                    logging.warning(f"  Table extraction failed page {page_num}: {e}")
            if info["has_charts"]:
                chart_text = self._extract_chart_text(fitz_page, info["drawings"])
                if chart_text:
                    page_parts.append(f"[Chart/graph]\n{chart_text}")
            if info["has_images"]:
                image_parts = self._extract_image_text(
                    doc, fitz_page, fitz_page.get_images(full=True)
                )
                page_parts.extend(image_parts)
            if not page_parts:
                logging.info(f"  Page {page_num}: no content — full-page OCR fallback")
                try:
                    arr = self._rasterize_region(fitz_page, dpi=200)
                    ocr_text = self._ocr_array(arr)
                    if ocr_text:
                        page_parts.append(f"[Full page OCR]\n{ocr_text}")
                except Exception as e:
                    logging.warning(f"  Full-page OCR failed page {page_num}: {e}")
            if page_parts:
                results.append(("\n\n".join(page_parts), page_num))
        doc.close()
        plumber.close()
        return results

    def _is_complex_pdf(self, file_path: str) -> bool:
        try:
            doc = fitz.open(file_path)
            for page in doc:
                if page.get_images(full=True):
                    return True
                if len(page.get_drawings()) > 10:
                    return True
            doc.close()
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    if page.extract_tables():
                        return True
        except Exception:
            pass
        return False

    def _extract_image_fallback(self, file_path: str) -> list:
        try:
            img_array = np.array(Image.open(file_path).convert("RGB"))
            ocr_text = self._ocr_array(img_array)
            if ocr_text:
                logging.info(f"Image OCR extracted {len(ocr_text)} chars from {file_path}")
                return [(ocr_text, 1)]
        except Exception as e:
            logging.error(f"Image fallback failed for {file_path}: {e}")
        return []

    def _extract(self, file_path: str) -> list:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext in {".jpg", ".jpeg", ".png"}:
            return self._extract_image_fallback(file_path)
        if ext == ".pdf":
            if self._is_complex_pdf(file_path):
                logging.info(f"Complex PDF detected: {file_path}")
                return self._extract_complex_pdf(file_path)
        try:
            elements = partition(filename=file_path)
        except Exception as e:
            logging.error(f"Failed to partition {file_path}: {e}")
            if ext == ".csv":
                return self._extract_csv_fallback(file_path)
            return []
        pages: dict = {}
        for el in elements:
            page_num = (el.metadata.page_number or 1) if el.metadata else 1
            text = str(el).strip()
            if text:
                pages.setdefault(page_num, []).append(text)
        return [
            ("\n".join(texts), page_num)
            for page_num, texts in sorted(pages.items())
        ]

    def _extract_csv_fallback(self, file_path: str) -> list:
        try:
            rows = []
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    line = ", ".join(f"{k}: {v}" for k, v in row.items() if v and v.strip())
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

    def upload_documents(self, path: str, file_hash: str = None):
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
                        "document_id": batch_sources[i]["document_id"],
                        "version": batch_sources[i]["version"],
                        "file_hash": batch_sources[i]["file_hash"]  
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
            document_id = str(uuid.uuid4())
            _, ext = os.path.splitext(file_path)
            if ext.lower() not in self.SUPPORTED_EXTENSIONS:
                logging.info(f"Skipping unsupported file: {file_path}")
                continue

            file = os.path.basename(file_path)
            folder = os.path.dirname(file_path)
            logging.info(f"Extracting: {file_path}")

            if file_hash is not None:
                computed_hash = file_hash
            else:
                with open(file_path, "rb") as f:
                    computed_hash = hashlib.sha256(f.read()).hexdigest()
                logging.info(f"Computed hash for {file}: {computed_hash}")

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
                        "folder": folder,
                        "document_id": document_id,
                        "version": 1,
                        "file_hash": computed_hash 
                    })
                    if len(batch_texts) == BATCH_SIZE:
                        flush_batch()

        flush_batch()
        return {"message": "Chunks ingested successfully."}
