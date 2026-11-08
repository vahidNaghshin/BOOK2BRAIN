# Book2Brain

Turn books into a searchable knowledge base. Books are uploaded as PDF, EPUB or DOCX files, stored safely on disk, and tracked in PostgreSQL (with pgvector for later phases).

**Status:** Day 3 — document parsing and OCR. Uploaded books can be converted into clean, normalized text blocks that keep their source location (page numbers for PDF, chapter references for EPUB). Chapter detection and passage segmentation come next.

## Prerequisites

- Git
- Python 3.12
- Docker with Docker Compose
- **Tesseract OCR** (system package, used for scanned PDF pages):
  - macOS: `brew install tesseract`
  - Debian/Ubuntu: `sudo apt install tesseract-ocr`
  - Windows: install the [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki) and add it to `PATH`
  - Other languages than English need extra data: `brew install tesseract-lang` or `sudo apt install tesseract-ocr-<lang>`
- About 1 GB of free disk space and internet access the first time a PDF is processed: Docling downloads its layout and table models (about 500 MB) once. See [Docling models](#docling-models).

Check Tesseract with `tesseract --version` and `tesseract --list-langs`. If it is missing, the app still works for digital PDFs, EPUB and DOCX; only documents that need OCR fail, with a message telling you what to install.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/vahidNaghshin/BOOK2BRAIN.git
cd BOOK2BRAIN
```

### 2. Create the Python environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and replace `change-me` with a password of your choice. Use the same password in both `POSTGRES_PASSWORD` and `DATABASE_URL`.

| Variable | Description |
|---|---|
| `POSTGRES_DB` | Database name (`book2brain`) |
| `POSTGRES_USER` | Application database user |
| `POSTGRES_PASSWORD` | Database password (required) |
| `POSTGRES_PORT` | Port published on localhost (default `5432`) |
| `DATABASE_URL` | SQLAlchemy connection URL used by the app and by migrations |
| `STORAGE_ROOT` | Where original uploaded files are stored (default `storage`; relative paths are resolved from the project root, absolute paths can be anywhere) |
| `MAX_UPLOAD_SIZE_MB` | Maximum upload size in MB (default `100`) |
| `LOG_LEVEL` | Logging level (default `INFO`) |
| `OCR_LANGUAGE` | Tesseract language(s), e.g. `eng` or `eng+deu` (default `eng`; the language data must be installed) |
| `OCR_DPI` | Resolution pages are rendered at before OCR, 72–600 (default `300`) |
| `OCR_ENABLED` | Set to `false` to skip scanned pages instead of running OCR (default `true`) |
| `OCR_MIN_TEXT_CHARS` | A page with fewer letters/digits than this and a page image is treated as scanned (default `25`) |
| `OCR_MAX_GARBLED_RATIO` | A page whose extracted text is more garbled than this fraction goes to OCR (default `0.3`) |
| `OCR_MIN_CONFIDENCE` | OCR text below this confidence (0–100) never replaces already-extracted text (default `30`) |
| `MIN_CONTENT_CHARACTERS` | A document with less extracted text than this is not marked completed (default `20`) |
| `PROCESSING_STALE_MINUTES` | A run with no result after this long is treated as interrupted and can be retried (default `120`) |
| `TEST_DATABASE_URL` | Optional database for the test suite (see [Tests](#tests)) |

If you change `POSTGRES_PORT`, change the port in `DATABASE_URL` too. Special characters in the password must be URL-encoded in `DATABASE_URL`.

To accept uploads larger than 100 MB, raise `MAX_UPLOAD_SIZE_MB` **and** `maxUploadSize` in `.streamlit/config.toml` (Streamlit's own cap).

### 4. Start the database

```bash
docker compose up -d
docker compose ps
```

Wait until the `db` service shows `healthy`. Data is kept in the `pgdata` Docker volume.

### 5. Create the database tables

```bash
alembic upgrade head
```

This runs the migrations in `database/migrations/`, creating the `books`, `documents`, `processing_runs`, `processing_errors` and `extracted_blocks` tables and enabling the pgvector extension. Run it again after pulling new migrations; it is safe to repeat.

### 6. Launch the application

```bash
streamlit run app/main.py
```

Open http://localhost:8501. The page shows the title, version and database status, with three tabs:

- **Upload** — choose a PDF, EPUB or DOCX file, enter the book metadata (title is required) and upload it.
- **Library** — all uploaded documents with title, author, format, size, upload date and processing status, plus a details view for each book and its documents.
- **Processing** — pick a document, start (or retry) processing, watch its status and progress, and preview the extracted text. See [Processing](#processing).

## Uploads

- Allowed types: `.pdf`, `.epub`, `.docx`. The file **content** is checked, not just the extension or the browser-supplied MIME type.
- Files above `MAX_UPLOAD_SIZE_MB`, empty files and invalid files are rejected with a clear message.
- Every file is hashed with SHA-256. Uploading the same file again is rejected as a duplicate: no second copy is stored and no second record is created.
- A book is identified by its title and author (case-insensitive). Uploading another format of the same book (same title and author) attaches a new document to the existing book; the existing book's metadata is left unchanged.
- Files are stored under `STORAGE_ROOT/original/<document-uuid>/<document-uuid>.<ext>`. The original filename is kept in the database only. If saving the file or the database record fails, neither is kept.

## Processing

Open the **Processing** tab, pick a document and press **Start processing**. The work runs in the background, so you can leave the page; the tab refreshes itself while a run is active. Processing turns the stored file into an ordered list of **blocks** (headings, paragraphs, list items, tables, captions) saved in PostgreSQL.

| Format | Parsed with | Source location kept on every block |
|---|---|---|
| PDF | Docling (structure and text), Tesseract (scanned pages) | page number, position on the page, element reference |
| EPUB | EbookLib (spine items in reading order) | chapter/spine item id, file name, chapter title, element index |
| DOCX | Docling | element reference and index; **no page numbers** (Word has no fixed pages) |

Page numbers are only stored where the format has real pages; they are never guessed. Blocks also carry a heading `section_title` (the nearest heading before them) and, for headings, the heading level. Chapter detection and passage segmentation are not part of this step.

### Status and workflow

A document moves through `uploaded` → `pending` → `processing` → `completed` or `failed`. While processing, the run reports its stage: `parsing`, then `ocr` (only when needed), then `normalizing`. Every attempt is recorded as a processing run, and failures are stored with their cause and diagnostic details.

- Results are saved in a single transaction: a failed run never leaves partial text behind.
- A document is **not** marked completed if almost nothing was extracted (`MIN_CONTENT_CHARACTERS`).
- Failed documents show the reason and can be **retried**; earlier attempts and errors are kept as history. A completed document cannot be processed again yet.
- If the app stops mid-run, the run is treated as interrupted after `PROCESSING_STALE_MINUTES` and the document can be retried.

### Scanned PDFs and OCR

PDFs are assessed **page by page**, so mixed digital/scanned files work. A page goes to Tesseract only if its extracted text is missing or too short (`OCR_MIN_TEXT_CHARS`) **and** the page contains an image, or if the extracted text is garbled (`OCR_MAX_GARBLED_RATIO`). Blank pages are skipped. OCR runs at `OCR_DPI` in `OCR_LANGUAGE`, keeps the original page number, and records its confidence. OCR text never replaces text that was already extracted when it is shorter or below `OCR_MIN_CONFIDENCE`. The Processing tab shows which pages used OCR, which failed, and any warnings.

OCR is a separate service (`app/parsing/ocr_service.py`) behind an `OcrEngine` interface, so another engine can be added without changing the processing workflow.

### Text normalization

All formats share one deterministic normalizer: Unicode NFC, consistent newlines, tidy whitespace, no excessive blank lines, and removal of extraction artifacts (soft hyphens, zero-width characters, ligatures such as "ﬁ"). It does not merge paragraphs, drop punctuation, numbering, citations or table content, or rewrite text. When normalization changed a block, the original text is kept (`raw_text`); tick **Show original text** in the preview to inspect it.

### Docling models

On the first PDF it processes, Docling downloads its layout and table models (about 500 MB, cached under `~/.cache/huggingface`). This needs internet access once; afterwards processing works offline. `docling-tools models download` (see its `--help`) can fetch Docling's models ahead of time, but the first PDF run downloads whatever is missing automatically.

If the download fails with `CERTIFICATE_VERIFY_FAILED`, your network probably inspects TLS traffic (corporate proxy). Point Python at a certificate bundle that includes your company's root certificate for that first run, for example `export SSL_CERT_FILE=/path/to/ca-bundle.pem` (and `REQUESTS_CA_BUNDLE` to the same file). EPUB and DOCX processing do not need the models.

## Common tasks

| Task | Command |
|---|---|
| Stop the database | `docker compose stop` |
| Delete the database and its data | `docker compose down -v` |
| Apply migrations | `alembic upgrade head` |
| Show the current migration | `alembic current` |
| Create a migration after changing models | `alembic revision --autogenerate -m "describe change"` (review the generated file) |
| Roll back the last migration | `alembic downgrade -1` |
| Run tests | `pytest` |

If you change the database password in `.env` after the first start, run `docker compose down -v` and start the database again, because the password is only applied when the volume is created.

## Tests

```bash
docker compose up -d   # tests need PostgreSQL
pytest
```

Database tests use a separate database named after yours with a `_test` suffix (for example `book2brain_test`). It is created automatically and built with the real migrations, and your data is never touched. To use a different one, set `TEST_DATABASE_URL`; its name must contain `_test`. Uploaded files in tests go to temporary directories, and all test documents (PDF, EPUB, DOCX, scans) are generated in code, so no books are stored in the repository.

- Tests that use the real Docling PDF models are skipped automatically if the models cannot be loaded (for example offline on a fresh machine). Run PDF processing once, or `docling-tools models download`, to enable them.
- Tests marked `ocr` use a real Tesseract installation and are skipped if it is missing. Run only them with `pytest -m ocr`, or leave them out with `pytest -m "not ocr"`. All other OCR tests use a fake engine.

## Troubleshooting the database connection

Check the connection from the project folder, with the venv active:

```bash
python -m database.connection
```

The app shows the same message under the title.

| Message contains | Cause and fix |
|---|---|
| `password authentication failed` | The password in `DATABASE_URL` doesn't match the one the database was created with. Make both passwords in `.env` match. If you changed the password after the first start, run `docker compose down -v` and `docker compose up -d`. This deletes the database data. |
| `password authentication failed` or `role "book2brain" does not exist`, with a local PostgreSQL installed | Another PostgreSQL server is already using port 5432, so the app connects to it instead of the container. Check with `lsof -nP -iTCP:5432 -sTCP:LISTEN`. Stop the other server, or set `POSTGRES_PORT=5433` and use port 5433 in `DATABASE_URL`, then run `docker compose up -d`. |
| `Connection refused` | Nothing is listening on the port in `DATABASE_URL`. Check that `docker compose ps` shows the port mapping (for example `0.0.0.0:5432->5432/tcp`) and that the port matches `POSTGRES_PORT`. |
| `DATABASE_URL is not set` | `.env` is missing. Run the app from the project folder after `cp .env.example .env`. |
| `relation "books" does not exist` | The tables haven't been created. Run `alembic upgrade head`. |
| `relation "extracted_blocks" does not exist` or `column ... stage does not exist` | The Day 3 migration hasn't been applied. Run `alembic upgrade head`. |

## Troubleshooting processing

Failed documents show the reason in the Processing tab, with the error code and details.

| Error code | Cause and fix |
|---|---|
| `ocr_unavailable` | Tesseract isn't installed, or the `OCR_LANGUAGE` data is missing. Follow the install hint in the message (see [Prerequisites](#prerequisites)), then press **Retry**. Or set `OCR_ENABLED=false` to skip scanned pages. |
| `encrypted_document` | The PDF is password-protected. Remove the password and upload the file again. |
| `corrupt_document` | The file is damaged or isn't a valid PDF/EPUB/DOCX. Re-export or re-download it. |
| `empty_document` / `no_meaningful_content` | No (or almost no) text was found. A PDF of blank pages, or of images that OCR could not read, ends here. Try a higher `OCR_DPI`. |
| `unexpected_error` | A bug or environment problem. The traceback is in the error details and the app log. |
| Download fails with `CERTIFICATE_VERIFY_FAILED` | See [Docling models](#docling-models). |

## Project layout

```
app/
  main.py          Streamlit entry point
  config.py        Settings (environment variables)
  schemas.py       Pydantic models (book metadata, read models)
  ui/              Streamlit pages: upload.py, library.py, processing.py
  services/        Business logic: book_service, document_service, storage_service,
                   file_validation, document_processing_service, job_runner
  parsing/         Parsers and OCR, independent of the UI and the database:
                   models.py (output models), base.py, parser_factory.py,
                   pdf_parser.py, epub_parser.py, docx_parser.py, ocr_service.py,
                   scan_detection.py, text_normalizer.py
  repositories/    Database queries
database/
  connection.py    Engine, sessions, connection check
  models/          SQLAlchemy models (Book, Document, ProcessingRun, ProcessingError, ExtractedBlock)
  migrations/      Alembic migrations (configured by alembic.ini)
  init/            Scripts run when the Docker database is first created
ingestion/         Reserved for later phases
storage/           Default location of uploaded files (git-ignored)
tests/             pytest suite
```

The layers are kept separate: the UI calls services, services call parsers, repositories and the storage backend, and only repositories talk to the database. Parsers return structured results (`ParsingResult`) and never touch the database or the UI. Storage sits behind the `StorageBackend` interface in `storage_service.py`, so cloud storage can be added later.

## Known limitations

- A completed document cannot be re-processed yet (for example after changing OCR settings); only failed documents can be retried.
- Chapter detection, section hierarchy and passage segmentation are not implemented (Day 4). `section_title` is only the nearest preceding heading.
- Page headers and footers detected by Docling are left out of PDF text. Text inside images that are not full-page scans is not OCR'd.
- DOCX and EPUB have no page numbers; blocks are located by element and chapter references instead.
- One document is processed at a time, and the work runs inside the app process: restarting the app interrupts a run (it can then be retried).
