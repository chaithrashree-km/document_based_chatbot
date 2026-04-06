"""
Additional tests to cover the missing lines in app/services/Ingest_Service.py.

Missing lines from coverage report:
  33-37   -> _get_ocr_reader
  40-50   -> _analyse_page
  61-73   -> _table_to_markdown
  77-83   -> _rasterize_region
  86-88   -> _ocr_array
  91-115  -> _extract_chart_text
  118-131 -> _extract_image_text
  134-184 -> _extract_complex_pdf
  187-201 -> _is_complex_pdf
  204-212 -> _extract_image_fallback
  228     -> _extract_csv_fallback branch inside _extract (csv partition failure)
  242-257 -> _extract_csv_fallback method

These tests are written to work alongside the existing test_IngestService.py
(Document 2) which already handles upload_documents and _extract routing.
Run both files together:
    pytest tests/app/services/test_IngestService.py
           tests/app/services/test_IngestService_coverage.py -v --cov=app/services/Ingest_Service
"""

import numpy as np
import pytest
from io import BytesIO
from unittest.mock import MagicMock, patch, call
from app.services.Ingest_Service import Upload


class TestUpload:
    def setup_method(self):
        self.upload = Upload()

        self.upload.database = MagicMock()
        self.upload.database.client = MagicMock()
        self.upload.database.create_collection = MagicMock()

        self.upload.model = MagicMock()
        self.upload.model.get_sentence_embedding_dimension.return_value = 384
        self.upload.model.encode.return_value = np.array([[0.1] * 384])

        self.upload.splitter = MagicMock()
        self.upload.splitter.split_text.return_value = ["chunk1"]

        self.upload.config = MagicMock()
        self.upload.config.COLLECTION_NAME = "test_collection"

        Upload._ocr_reader = None

    def teardown_method(self):
        Upload._ocr_reader = None

    def test_get_ocr_reader_creates_reader_when_none(self):
        """First call must instantiate easyocr.Reader with cpu=False."""
        mock_reader = MagicMock()
        with patch("app.services.Ingest_Service.easyocr.Reader",
                   return_value=mock_reader) as mock_cls:
            reader = Upload._get_ocr_reader()

        mock_cls.assert_called_once_with(["en"], gpu=False)
        assert reader is mock_reader
        assert Upload._ocr_reader is mock_reader

    def test_get_ocr_reader_returns_cached_reader_on_second_call(self):
        """Subsequent calls must NOT construct a new Reader."""
        existing = MagicMock()
        Upload._ocr_reader = existing

        with patch("app.services.Ingest_Service.easyocr.Reader") as mock_cls:
            reader = Upload._get_ocr_reader()

        mock_cls.assert_not_called()
        assert reader is existing

    def _make_fitz_page(self, text="", images=None, drawings=None):
        page = MagicMock()
        page.get_text.return_value = text
        page.get_images.return_value = images or []
        page.get_drawings.return_value = drawings or []
        page.rect = MagicMock(width=612, height=792)
        return page

    def _make_plumber_page(self, tables=None, raise_exc=False):
        page = MagicMock()
        if raise_exc:
            page.extract_tables.side_effect = Exception("table error")
        else:
            page.extract_tables.return_value = tables or []
        return page

    def test_analyse_page_has_text_true(self):
        fp = self._make_fitz_page(text="a" * 81)
        pp = self._make_plumber_page()
        result = self.upload._analyse_page(fp, pp)
        assert result["has_text"] is True

    def test_analyse_page_has_text_false_short(self):
        fp = self._make_fitz_page(text="short")
        pp = self._make_plumber_page()
        result = self.upload._analyse_page(fp, pp)
        assert result["has_text"] is False

    def test_analyse_page_has_text_false_empty(self):
        fp = self._make_fitz_page(text="")
        pp = self._make_plumber_page()
        result = self.upload._analyse_page(fp, pp)
        assert result["has_text"] is False

    def test_analyse_page_has_tables_true(self):
        fp = self._make_fitz_page()
        pp = self._make_plumber_page(tables=[[["h1", "h2"]]])
        result = self.upload._analyse_page(fp, pp)
        assert result["has_tables"] is True

    def test_analyse_page_has_tables_false_no_tables(self):
        fp = self._make_fitz_page()
        pp = self._make_plumber_page(tables=[])
        result = self.upload._analyse_page(fp, pp)
        assert result["has_tables"] is False

    def test_analyse_page_has_tables_false_on_exception(self):
        """extract_tables raising must be silently swallowed → False."""
        fp = self._make_fitz_page()
        pp = self._make_plumber_page(raise_exc=True)
        result = self.upload._analyse_page(fp, pp)
        assert result["has_tables"] is False

    def test_analyse_page_has_images_true(self):
        fp = self._make_fitz_page(images=[(1,)])
        pp = self._make_plumber_page()
        result = self.upload._analyse_page(fp, pp)
        assert result["has_images"] is True

    def test_analyse_page_has_images_false(self):
        fp = self._make_fitz_page(images=[])
        pp = self._make_plumber_page()
        result = self.upload._analyse_page(fp, pp)
        assert result["has_images"] is False

    def test_analyse_page_has_charts_true_more_than_10_drawings(self):
        fp = self._make_fitz_page(drawings=list(range(11)))
        pp = self._make_plumber_page()
        result = self.upload._analyse_page(fp, pp)
        assert result["has_charts"] is True

    def test_analyse_page_has_charts_false_exactly_10_drawings(self):
        fp = self._make_fitz_page(drawings=list(range(10)))
        pp = self._make_plumber_page()
        result = self.upload._analyse_page(fp, pp)
        assert result["has_charts"] is False

    def test_analyse_page_returns_text_and_drawings(self):
        drawings = [{"rect": (0, 0, 10, 10)}]
        fp = self._make_fitz_page(text="hello world", drawings=drawings)
        pp = self._make_plumber_page()
        result = self.upload._analyse_page(fp, pp)
        assert result["text"] == "hello world"
        assert result["drawings"] is drawings

    def test_table_to_markdown_empty_list(self):
        assert Upload._table_to_markdown([]) == ""

    def test_table_to_markdown_header_and_rows(self):
        table = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
        md = Upload._table_to_markdown(table)
        assert "| Name | Age |" in md
        assert "| --- | --- |" in md
        assert "| Alice | 30 |" in md
        assert "| Bob | 25 |" in md

    def test_table_to_markdown_header_only(self):
        md = Upload._table_to_markdown([["Col1", "Col2"]])
        assert "| Col1 | Col2 |" in md
        assert "---" in md

    def test_table_to_markdown_none_cells_become_empty_string(self):
        md = Upload._table_to_markdown([[None, "B"], ["C", None]])
        assert "|  | B |" in md

    def test_table_to_markdown_all_blank_header_skips_separator(self):
        md = Upload._table_to_markdown([["", ""], ["a", "b"]])
        assert "---" not in md

    def test_table_to_markdown_all_empty_data_row_skipped(self):
        md = Upload._table_to_markdown([["H1", "H2"], ["", ""], ["x", "y"]])
        lines = [ln for ln in md.splitlines() if ln.strip()]
        # header line + separator + one data row = 3
        assert len(lines) == 3

    def _make_pixmap(self, h, w, n):
        pix = MagicMock()
        pix.height = h
        pix.width = w
        pix.n = n
        pix.samples = bytes([128] * (h * w * n))
        return pix

    def test_rasterize_region_rgb_shape(self):
        page = MagicMock()
        page.get_pixmap.return_value = self._make_pixmap(4, 6, 3)
        arr = Upload._rasterize_region(page, dpi=72)
        assert arr.shape == (4, 6, 3)

    def test_rasterize_region_rgba_alpha_stripped(self):
        page = MagicMock()
        page.get_pixmap.return_value = self._make_pixmap(2, 3, 4)
        arr = Upload._rasterize_region(page, dpi=72)
        assert arr.shape == (2, 3, 3)

    def test_rasterize_region_passes_clip_rect(self):
        page = MagicMock()
        page.get_pixmap.return_value = self._make_pixmap(1, 1, 3)
        clip = MagicMock()
        Upload._rasterize_region(page, clip_rect=clip, dpi=72)
        kwargs = page.get_pixmap.call_args[1]
        assert kwargs["clip"] is clip

    def test_rasterize_region_dpi_affects_scale(self):
        """Passing dpi=144 doubles the scale vs dpi=72; just verify no error."""
        page = MagicMock()
        page.get_pixmap.return_value = self._make_pixmap(1, 1, 3)
        arr = Upload._rasterize_region(page, dpi=144)
        assert arr.shape == (1, 1, 3)

    def test_ocr_array_returns_joined_text(self):
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = ["Hello", "World"]
        with patch.object(Upload, "_get_ocr_reader", return_value=mock_reader):
            result = self.upload._ocr_array(np.zeros((5, 5, 3), dtype=np.uint8))
        assert result == "Hello World"

    def test_ocr_array_returns_empty_string_when_no_text(self):
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        with patch.object(Upload, "_get_ocr_reader", return_value=mock_reader):
            result = self.upload._ocr_array(np.zeros((5, 5, 3), dtype=np.uint8))
        assert result == ""

    def test_ocr_array_strips_whitespace(self):
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = ["  hi  "]
        with patch.object(Upload, "_get_ocr_reader", return_value=mock_reader):
            result = self.upload._ocr_array(np.zeros((5, 5, 3), dtype=np.uint8))
        assert result == "hi"

    def test_ocr_array_calls_readtext_with_correct_params(self):
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        with patch.object(Upload, "_get_ocr_reader", return_value=mock_reader):
            self.upload._ocr_array(img)
        mock_reader.readtext.assert_called_once_with(img, detail=0, paragraph=True)

    def test_extract_chart_text_empty_drawings_returns_empty(self):
        result = self.upload._extract_chart_text(MagicMock(), [])
        assert result == ""

    def test_extract_chart_text_drawings_without_rect_returns_empty(self):
        result = self.upload._extract_chart_text(
            MagicMock(), [{"no_rect": True}, {"also_no_rect": True}]
        )
        assert result == ""

    def test_extract_chart_text_includes_text_layer_section(self):
        page = MagicMock()
        page.rect = MagicMock(width=612, height=792)
        page.get_text.return_value = "axis label"

        mock_rect = MagicMock()
        mock_rect.__or__ = lambda s, o: s
        mock_rect.x0, mock_rect.y0 = 10, 10
        mock_rect.x1, mock_rect.y1 = 200, 200

        with patch("app.services.Ingest_Service.fitz.Rect", return_value=mock_rect), \
             patch.object(self.upload, "_rasterize_region",
                          return_value=np.zeros((5, 5, 3), dtype=np.uint8)), \
             patch.object(self.upload, "_ocr_array", return_value="different ocr"):
            result = self.upload._extract_chart_text(
                page, [{"rect": (10, 10, 200, 200)}]
            )

        assert "[Chart labels from text layer]" in result
        assert "axis label" in result

    def test_extract_chart_text_includes_ocr_section_when_different(self):
        page = MagicMock()
        page.rect = MagicMock(width=612, height=792)
        page.get_text.return_value = "axis label"

        mock_rect = MagicMock()
        mock_rect.__or__ = lambda s, o: s
        mock_rect.x0, mock_rect.y0 = 0, 0
        mock_rect.x1, mock_rect.y1 = 100, 100

        with patch("app.services.Ingest_Service.fitz.Rect", return_value=mock_rect), \
             patch.object(self.upload, "_rasterize_region",
                          return_value=np.zeros((5, 5, 3), dtype=np.uint8)), \
             patch.object(self.upload, "_ocr_array", return_value="ocr result"):
            result = self.upload._extract_chart_text(
                page, [{"rect": (0, 0, 100, 100)}]
            )

        assert "[Chart labels from OCR]" in result
        assert "ocr result" in result

    def test_extract_chart_text_skips_ocr_duplicate_of_text_layer(self):
        page = MagicMock()
        page.rect = MagicMock(width=612, height=792)
        page.get_text.return_value = "same text"

        mock_rect = MagicMock()
        mock_rect.__or__ = lambda s, o: s
        mock_rect.x0, mock_rect.y0 = 0, 0
        mock_rect.x1, mock_rect.y1 = 50, 50

        with patch("app.services.Ingest_Service.fitz.Rect", return_value=mock_rect), \
             patch.object(self.upload, "_rasterize_region",
                          return_value=np.zeros((5, 5, 3), dtype=np.uint8)), \
             patch.object(self.upload, "_ocr_array", return_value="same text"):
            result = self.upload._extract_chart_text(
                page, [{"rect": (0, 0, 50, 50)}]
            )

        assert "[Chart labels from OCR]" not in result

    def test_extract_chart_text_empty_text_layer_and_ocr_yields_empty_parts(self):
        page = MagicMock()
        page.rect = MagicMock(width=612, height=792)
        page.get_text.return_value = ""

        mock_rect = MagicMock()
        mock_rect.__or__ = lambda s, o: s
        mock_rect.x0, mock_rect.y0 = 0, 0
        mock_rect.x1, mock_rect.y1 = 50, 50

        with patch("app.services.Ingest_Service.fitz.Rect", return_value=mock_rect), \
             patch.object(self.upload, "_rasterize_region",
                          return_value=np.zeros((5, 5, 3), dtype=np.uint8)), \
             patch.object(self.upload, "_ocr_array", return_value=""):
            result = self.upload._extract_chart_text(
                page, [{"rect": (0, 0, 50, 50)}]
            )

        assert result == ""

    def test_extract_chart_text_union_rect_over_multiple_drawings(self):
        """Two drawings → union_rect | r branch executes."""
        page = MagicMock()
        page.rect = MagicMock(width=612, height=792)
        page.get_text.return_value = ""

        r1 = MagicMock()
        r1.__or__ = lambda s, o: s
        r1.x0, r1.y0, r1.x1, r1.y1 = 0, 0, 50, 50

        with patch("app.services.Ingest_Service.fitz.Rect", return_value=r1), \
             patch.object(self.upload, "_rasterize_region",
                          return_value=np.zeros((5, 5, 3), dtype=np.uint8)), \
             patch.object(self.upload, "_ocr_array", return_value=""):
            result = self.upload._extract_chart_text(
                page,
                [{"rect": (0, 0, 50, 50)}, {"rect": (60, 60, 100, 100)}]
            )

        assert isinstance(result, str)

    def test_extract_image_text_returns_ocr_for_valid_image(self):
        doc = MagicMock()
        doc.extract_image.return_value = {"image": b"\x89PNG\r\n\x1a\n"}
        mock_pil = MagicMock()
        mock_pil.convert.return_value = mock_pil

        with patch("app.services.Ingest_Service.Image.open", return_value=mock_pil), \
             patch("app.services.Ingest_Service.np.array",
                   return_value=np.zeros((5, 5, 3), dtype=np.uint8)), \
             patch.object(self.upload, "_ocr_array", return_value="ocr text"):
            parts = self.upload._extract_image_text(doc, MagicMock(), [(1,)])

        assert len(parts) == 1
        assert "[Embedded image 1 OCR]" in parts[0]
        assert "ocr text" in parts[0]

    def test_extract_image_text_skips_empty_ocr_result(self):
        doc = MagicMock()
        doc.extract_image.return_value = {"image": b"\x00" * 10}
        mock_pil = MagicMock()
        mock_pil.convert.return_value = mock_pil

        with patch("app.services.Ingest_Service.Image.open", return_value=mock_pil), \
             patch("app.services.Ingest_Service.np.array",
                   return_value=np.zeros((5, 5, 3), dtype=np.uint8)), \
             patch.object(self.upload, "_ocr_array", return_value=""):
            parts = self.upload._extract_image_text(doc, MagicMock(), [(1,)])

        assert parts == []

    def test_extract_image_text_handles_extract_image_exception(self):
        doc = MagicMock()
        doc.extract_image.side_effect = Exception("corrupt xref")
        parts = self.upload._extract_image_text(doc, MagicMock(), [(99,)])
        assert parts == []

    def test_extract_image_text_numbers_images_sequentially(self):
        doc = MagicMock()
        doc.extract_image.return_value = {"image": b"\x00" * 10}
        mock_pil = MagicMock()
        mock_pil.convert.return_value = mock_pil

        with patch("app.services.Ingest_Service.Image.open", return_value=mock_pil), \
             patch("app.services.Ingest_Service.np.array",
                   return_value=np.zeros((5, 5, 3), dtype=np.uint8)), \
             patch.object(self.upload, "_ocr_array", return_value="text"):
            parts = self.upload._extract_image_text(doc, MagicMock(), [(1,), (2,)])

        assert "[Embedded image 1 OCR]" in parts[0]
        assert "[Embedded image 2 OCR]" in parts[1]

    def test_extract_image_text_empty_image_list_returns_empty(self):
        parts = self.upload._extract_image_text(MagicMock(), MagicMock(), [])
        assert parts == []

    def _build_complex_pdf_mocks(self, pages_cfg):
        fitz_pages, plumber_pages = [], []
        for cfg in pages_cfg:
            fp = self._make_fitz_page(
                text=cfg.get("text", ""),
                images=cfg.get("images", []),
                drawings=cfg.get("drawings", []),
            )
            pp = MagicMock()
            if cfg.get("raise_tables"):
                pp.extract_tables.side_effect = Exception("table error")
            else:
                pp.extract_tables.return_value = cfg.get("tables", [])
            fitz_pages.append(fp)
            plumber_pages.append(pp)

        doc = MagicMock()
        doc.__len__ = MagicMock(return_value=len(pages_cfg))
        doc.__getitem__ = MagicMock(side_effect=lambda i: fitz_pages[i])

        plumber_obj = MagicMock()
        plumber_obj.pages = plumber_pages

        return doc, plumber_obj

    def test_extract_complex_pdf_returns_empty_on_open_failure(self):
        with patch("app.services.Ingest_Service.fitz.open",
                   side_effect=Exception("cannot open")):
            result = self.upload._extract_complex_pdf("bad.pdf")
        assert result == []

    def test_extract_complex_pdf_text_page(self):
        doc, plumber = self._build_complex_pdf_mocks([{"text": "a" * 90}])
        with patch("app.services.Ingest_Service.fitz.open", return_value=doc), \
             patch("app.services.Ingest_Service.pdfplumber.open", return_value=plumber):
            results = self.upload._extract_complex_pdf("f.pdf")
        assert len(results) == 1
        assert "[Text]" in results[0][0]
        assert results[0][1] == 1

    def test_extract_complex_pdf_table_page(self):
        doc, plumber = self._build_complex_pdf_mocks(
            [{"tables": [[["H1", "H2"], ["v1", "v2"]]]}]
        )
        with patch("app.services.Ingest_Service.fitz.open", return_value=doc), \
             patch("app.services.Ingest_Service.pdfplumber.open", return_value=plumber):
            results = self.upload._extract_complex_pdf("f.pdf")
        assert "[Table 1]" in results[0][0]

    def test_extract_complex_pdf_table_extraction_exception_handled(self):
        """Table extraction raising must not crash the method; other content survives."""
        doc, plumber = self._build_complex_pdf_mocks(
            [{"text": "a" * 90, "raise_tables": True}]
        )
        with patch("app.services.Ingest_Service.fitz.open", return_value=doc), \
             patch("app.services.Ingest_Service.pdfplumber.open", return_value=plumber):
            results = self.upload._extract_complex_pdf("f.pdf")
        # Text was present so result is non-empty; no crash
        assert any("[Text]" in r[0] for r in results)

    def test_extract_complex_pdf_chart_page(self):
        doc, plumber = self._build_complex_pdf_mocks(
            [{"drawings": list(range(12))}]
        )
        with patch("app.services.Ingest_Service.fitz.open", return_value=doc), \
             patch("app.services.Ingest_Service.pdfplumber.open", return_value=plumber), \
             patch.object(self.upload, "_extract_chart_text",
                          return_value="chart content"):
            results = self.upload._extract_complex_pdf("f.pdf")
        assert "[Chart/graph]" in results[0][0]
        assert "chart content" in results[0][0]

    def test_extract_complex_pdf_chart_skipped_when_text_empty(self):
        """_extract_chart_text returning '' must NOT add a Chart/graph section."""
        doc, plumber = self._build_complex_pdf_mocks(
            [{"text": "a" * 90, "drawings": list(range(12))}]
        )
        with patch("app.services.Ingest_Service.fitz.open", return_value=doc), \
             patch("app.services.Ingest_Service.pdfplumber.open", return_value=plumber), \
             patch.object(self.upload, "_extract_chart_text", return_value=""):
            results = self.upload._extract_complex_pdf("f.pdf")
        assert "[Chart/graph]" not in results[0][0]

    def test_extract_complex_pdf_image_page(self):
        doc, plumber = self._build_complex_pdf_mocks(
            [{"images": [(1,)]}]
        )
        with patch("app.services.Ingest_Service.fitz.open", return_value=doc), \
             patch("app.services.Ingest_Service.pdfplumber.open", return_value=plumber), \
             patch.object(self.upload, "_extract_image_text",
                          return_value=["[Embedded image 1 OCR]\nimage text"]):
            results = self.upload._extract_complex_pdf("f.pdf")
        assert "image text" in results[0][0]

    def test_extract_complex_pdf_full_page_ocr_fallback_with_text(self):
        doc, plumber = self._build_complex_pdf_mocks([{"text": ""}])
        with patch("app.services.Ingest_Service.fitz.open", return_value=doc), \
             patch("app.services.Ingest_Service.pdfplumber.open", return_value=plumber), \
             patch.object(self.upload, "_rasterize_region",
                          return_value=np.zeros((5, 5, 3), dtype=np.uint8)), \
             patch.object(self.upload, "_ocr_array", return_value="full page ocr"):
            results = self.upload._extract_complex_pdf("f.pdf")
        assert "[Full page OCR]" in results[0][0]
        assert "full page ocr" in results[0][0]

    def test_extract_complex_pdf_full_page_ocr_fallback_empty_ocr_no_result(self):
        doc, plumber = self._build_complex_pdf_mocks([{"text": ""}])
        with patch("app.services.Ingest_Service.fitz.open", return_value=doc), \
             patch("app.services.Ingest_Service.pdfplumber.open", return_value=plumber), \
             patch.object(self.upload, "_rasterize_region",
                          return_value=np.zeros((5, 5, 3), dtype=np.uint8)), \
             patch.object(self.upload, "_ocr_array", return_value=""):
            results = self.upload._extract_complex_pdf("f.pdf")
        assert results == []

    def test_extract_complex_pdf_full_page_ocr_rasterize_exception_handled(self):
        doc, plumber = self._build_complex_pdf_mocks([{"text": ""}])
        with patch("app.services.Ingest_Service.fitz.open", return_value=doc), \
             patch("app.services.Ingest_Service.pdfplumber.open", return_value=plumber), \
             patch.object(self.upload, "_rasterize_region",
                          side_effect=Exception("raster fail")):
            results = self.upload._extract_complex_pdf("f.pdf")
        assert results == []

    def test_extract_complex_pdf_doc_and_plumber_closed(self):
        doc, plumber = self._build_complex_pdf_mocks([{"text": "a" * 90}])
        with patch("app.services.Ingest_Service.fitz.open", return_value=doc), \
             patch("app.services.Ingest_Service.pdfplumber.open", return_value=plumber):
            self.upload._extract_complex_pdf("f.pdf")
        doc.close.assert_called_once()
        plumber.close.assert_called_once()

    def test_extract_complex_pdf_page_number_starts_at_1(self):
        doc, plumber = self._build_complex_pdf_mocks([{"text": "a" * 90}])
        with patch("app.services.Ingest_Service.fitz.open", return_value=doc), \
             patch("app.services.Ingest_Service.pdfplumber.open", return_value=plumber):
            results = self.upload._extract_complex_pdf("f.pdf")
        assert results[0][1] == 1

    def test_extract_complex_pdf_multiple_pages(self):
        doc, plumber = self._build_complex_pdf_mocks(
            [{"text": "a" * 90}, {"text": "b" * 90}]
        )
        with patch("app.services.Ingest_Service.fitz.open", return_value=doc), \
             patch("app.services.Ingest_Service.pdfplumber.open", return_value=plumber):
            results = self.upload._extract_complex_pdf("f.pdf")
        assert len(results) == 2
        assert results[0][1] == 1
        assert results[1][1] == 2

    def _run_is_complex(self, fitz_pages, plumber_pages):
        doc = MagicMock()
        doc.__iter__ = MagicMock(return_value=iter(fitz_pages))

        plumber_pdf = MagicMock()
        plumber_pdf.pages = plumber_pages

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=plumber_pdf)
        ctx.__exit__ = MagicMock(return_value=False)

        with patch("app.services.Ingest_Service.fitz.open", return_value=doc), \
             patch("app.services.Ingest_Service.pdfplumber.open", return_value=ctx):
            return self.upload._is_complex_pdf("f.pdf")

    def test_is_complex_pdf_true_when_page_has_images(self):
        page = MagicMock()
        page.get_images.return_value = [(1,)]
        page.get_drawings.return_value = []
        assert self._run_is_complex([page], []) is True

    def test_is_complex_pdf_true_when_page_has_many_drawings(self):
        page = MagicMock()
        page.get_images.return_value = []
        page.get_drawings.return_value = list(range(11))
        assert self._run_is_complex([page], []) is True

    def test_is_complex_pdf_true_when_plumber_finds_tables(self):
        fitz_page = MagicMock()
        fitz_page.get_images.return_value = []
        fitz_page.get_drawings.return_value = []

        plumber_page = MagicMock()
        plumber_page.extract_tables.return_value = [[["a"]]]

        assert self._run_is_complex([fitz_page], [plumber_page]) is True

    def test_is_complex_pdf_false_for_plain_text_pdf(self):
        fitz_page = MagicMock()
        fitz_page.get_images.return_value = []
        fitz_page.get_drawings.return_value = []

        plumber_page = MagicMock()
        plumber_page.extract_tables.return_value = []

        assert self._run_is_complex([fitz_page], [plumber_page]) is False

    def test_is_complex_pdf_false_on_fitz_open_exception(self):
        with patch("app.services.Ingest_Service.fitz.open",
                   side_effect=Exception("cannot open")):
            result = self.upload._is_complex_pdf("bad.pdf")
        assert result is False

    def test_extract_image_fallback_returns_ocr_text(self):
        mock_pil = MagicMock()
        mock_pil.convert.return_value = mock_pil
        with patch("app.services.Ingest_Service.Image.open", return_value=mock_pil), \
             patch("app.services.Ingest_Service.np.array",
                   return_value=np.zeros((5, 5, 3), dtype=np.uint8)), \
             patch.object(self.upload, "_ocr_array", return_value="extracted text"):
            result = self.upload._extract_image_fallback("img.jpg")
        assert result == [("extracted text", 1)]

    def test_extract_image_fallback_returns_empty_when_ocr_empty(self):
        mock_pil = MagicMock()
        mock_pil.convert.return_value = mock_pil
        with patch("app.services.Ingest_Service.Image.open", return_value=mock_pil), \
             patch("app.services.Ingest_Service.np.array",
                   return_value=np.zeros((5, 5, 3), dtype=np.uint8)), \
             patch.object(self.upload, "_ocr_array", return_value=""):
            result = self.upload._extract_image_fallback("blank.png")
        assert result == []

    def test_extract_image_fallback_returns_empty_on_exception(self):
        with patch("app.services.Ingest_Service.Image.open",
                   side_effect=Exception("bad image")):
            result = self.upload._extract_image_fallback("corrupt.jpg")
        assert result == []

    def test_extract_csv_partition_failure_calls_csv_fallback(self):
        with patch("app.services.Ingest_Service.partition",
                   side_effect=Exception("partition fail")), \
             patch.object(self.upload, "_extract_csv_fallback",
                          return_value=[("row1, row2", 1)]) as mock_fb:
            result = self.upload._extract("data.csv")

        mock_fb.assert_called_once_with("data.csv")
        assert result == [("row1, row2", 1)]

    def test_extract_non_csv_partition_failure_returns_empty(self):
        with patch("app.services.Ingest_Service.partition",
                   side_effect=Exception("partition fail")):
            result = self.upload._extract("document.txt")
        assert result == []

    def test_extract_csv_fallback_valid_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25\n", encoding="utf-8")
        result = self.upload._extract_csv_fallback(str(f))
        assert len(result) == 1
        text, page = result[0]
        assert "Alice" in text
        assert "Bob" in text
        assert page == 1

    def test_extract_csv_fallback_header_only_returns_empty(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("name,age\n", encoding="utf-8")
        result = self.upload._extract_csv_fallback(str(f))
        assert result == []

    def test_extract_csv_fallback_blank_value_rows_skipped(self, tmp_path):
        f = tmp_path / "sparse.csv"
        f.write_text("name,age\n,\nAlice,30\n", encoding="utf-8")
        result = self.upload._extract_csv_fallback(str(f))
        assert len(result) == 1
        assert "Alice" in result[0][0]

    def test_extract_csv_fallback_returns_empty_on_io_exception(self):
        with patch("builtins.open", side_effect=Exception("io error")):
            result = self.upload._extract_csv_fallback("missing.csv")
        assert result == []

    def test_extract_csv_fallback_multiple_rows_joined_by_newline(self, tmp_path):
        f = tmp_path / "multi.csv"
        f.write_text("col\nrow1\nrow2\nrow3\n", encoding="utf-8")
        result = self.upload._extract_csv_fallback(str(f))
        assert len(result) == 1
        lines = result[0][0].splitlines()
        assert len(lines) == 3

    def test_extract_csv_fallback_row_formatted_as_key_value(self, tmp_path):
        f = tmp_path / "kv.csv"
        f.write_text("city,country\nParis,France\n", encoding="utf-8")
        result = self.upload._extract_csv_fallback(str(f))
        assert "city: Paris" in result[0][0]
        assert "country: France" in result[0][0]