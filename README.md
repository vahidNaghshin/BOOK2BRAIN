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

## Equations

Mathematics is extracted as **LaTeX with its source location**, never as flattened text. Equations are found while a document is processed (the text is saved first, so a problem with an equation can never fail the document) and shown in the **Equations** tab: rendered, with the raw LaTeX, the original source, the page/region, the surrounding text, a crop image of the region, how it was recognized, and a validation status. The tab can re-extract a document's equations at any time without touching its text.

| Format | How equations are found | Notes |
|---|---|---|
| PDF (digital or scanned) | Docling's layout analysis finds formula regions (works without a text layer). Docling's formula model (`CodeFormulaV2`) turns them into LaTeX. Regions it can't read go to math OCR. | Display equations only; inline formulas in PDFs are not detected. Page number and bounding box kept. |
| EPUB | MathML (`<math>`), LaTeX in the text (`$..$`, `$$..$$`, `\(..\)`, `\[..\]`) and images with LaTeX alt text. Equation images go to math OCR. | Inline vs display is known. An embedded TeX annotation is used verbatim. |
| DOCX | Word equations (OMML) converted to LaTeX; the original OMML is kept. | An equation that can't be converted is stored as invalid with a reason instead of being dropped. |

**Validation.** Every equation is `valid`, `uncertain` or `invalid`. LaTeX from the document's own markup (MathML, OMML, TeX) is valid if it parses. Model output is `valid` only when the model reported a confidence at or above `EQUATION_MIN_CONFIDENCE` or the PDF's own text layer agrees with it; on a scanned page there is nothing to compare with, so those results stay `uncertain` ("needs review") by design. The recognition `confidence` is only ever what an engine reports, and Docling's formula model reports none, so it is shown as "not reported". Unsafe commands (`\input`, `\write`, ...) are rejected, not rendered.

**Normalization** is deterministic and idempotent: delimiters removed, spacing tidied (`\frac { 1 } { 2 }` → `\frac{1}{2}`), `\tag{n}` / `(n)` read as the equation number. The text as extracted is kept alongside (`original_latex`).

### Measured quality

Measured with this repository's tools on equations rendered from known LaTeX (20 equations: algebra, calculus, series, statistics, physics, chemistry, logic ...; `python -m tools.evaluate_equations`). "Exact" is strict: the LaTeX must match the ground truth after removing spaces and `\left`/`\right`, so equivalent but differently written LaTeX (`\lor` vs `\vee`) counts as a miss.

| Scenario | Equation regions found | Exact | Validation |
|---|---|---|---|
| Digital PDF | 16 of 20 | 15 of 16 found | 16 valid (text layer agrees) |
| Scanned PDF (image only) | 16 of 20 | 15 of 16 found | 16 uncertain (no text layer to confirm) |

The four that were missed are missed by Docling's layout model, which sometimes does not mark a formula region at all. The 16th of the found regions differs only by equivalent notation. Recognition by itself, on isolated equation images (`python -m tools.compare_math_engines --engine docling|pix2text`):

| Engine | Exact | Token similarity ≥ 0.95 | Mean similarity | Speed (CPU) |
|---|---|---|---|---|
| Docling formula model (default) | 18 of 20 | 19 of 20 | 0.985 | about 1.5 s per equation |
| Pix2Text 1.1.7 (optional) | 9 of 20 | 11 of 20 | 0.863 | about 1.3 s per equation |

The one real recognition error of the default engine was a lowercase `x` read as `X`; the other miss is equivalent notation. These are synthetic, cleanly rendered equations: real scans, handwriting, tight multi-line derivations and matrices will be harder, which is why model output is never reported as certain without evidence. Run `tools/evaluate_equations.py` on your own material to measure it.

### Speed and the formula model

Docling's formula model (about 600 MB, downloaded on first use like the layout models; see [Docling models](#docling-models)) runs only on pages that contain formula regions. With Docling's default settings it took about 194 s for one page with 5 equations on an Apple Silicon Mac (measured; the cause was not investigated further). `FORMULA_ENRICHMENT_DEVICE=cpu` with `FORMULA_ENRICHMENT_DTYPE=float32` (the defaults here) brought that to about 15 s, and the 10 formulas of the digital test document above to about 20 s. A book with hundreds of equations therefore takes minutes, not seconds; the work runs in the background.

### Pictures that are really equations

Layout analysis often labels an equation as a *picture*. Small, wide picture regions (up to 110 pt high) are therefore tried with math OCR and kept only if the result reads as mathematics (`EQUATION_SCAN_PICTURES`, at most `EQUATION_SCAN_PICTURES_MAX` per document; a real figure that gets tried is simply discarded). Kept ones are marked "recognised from a picture region".

### Optional: Pix2Text

`MATH_OCR_ENGINE=pix2text` uses [Pix2Text](https://github.com/breezedeus/pix2text) instead of Docling's model for the fallback route. It is **not** installed by default: it downgrades `transformers` and adds about 50 packages that conflict with Docling's, and it was less accurate here (table above). If you want it, install it in its own virtual environment and pin `rapidocr==3.9.2` (at the time of writing `rapidocr` 3.10 breaks Pix2Text's `cnstd` dependency). The app tells you if it is missing or broken.

### Equation settings

See `.env.example`: `EQUATION_EXTRACTION_ENABLED`, `FORMULA_ENRICHMENT_ENABLED/DEVICE/DTYPE`, `MATH_OCR_ENABLED/ENGINE`, `EQUATION_MIN_CONFIDENCE`, `EQUATION_STORE_CROPS`, `EQUATION_CROP_DPI`, `EQUATION_SCAN_PICTURES`, `EQUATION_SCAN_PICTURES_MAX`. If the formula model can't be loaded (no internet on first use), the Equations tab and the run summary say so, the document text is unaffected, and equations can be re-extracted later.

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

- Tests that use the real Docling PDF models are skipped automatically if the models cannot be loaded (for example offline on a fresh machine). Run PDF processing once, or `docling-tools models download`, to enable them. The equation tests that run the real formula model are skipped separately if that model (about 600 MB, downloaded on first use) is missing.
- Equation tests render their PDFs with matplotlib (a test-only dependency in `requirements.txt`).
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
| `relation "equations" does not exist` or `column ... equation_status does not exist` | The Day 3.1 migration hasn't been applied. Run `alembic upgrade head`. |

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
| Equations tab: `equation_extraction_failed`, `formula_enrichment_failed` | The text is fine. The formula model could not be loaded (first use needs internet; see [Docling models](#docling-models)) or crashed. Fix the cause and press **Re-extract equations**. Math OCR is tried as a fallback where it works. |
| Equations marked `uncertain` | Expected for model output with nothing to confirm it (scans, no reported confidence). Check them against the region image. |
| `unsupported_omml` | A Word equation uses a structure the converter can't handle. It is listed as invalid with its original OMML kept; the rest of the document is unaffected. |

## Project layout

```
app/
  main.py          Streamlit entry point
  config.py        Settings (environment variables)
  schemas.py       Pydantic models (book metadata, read models)
  ui/              Streamlit pages: upload.py, library.py, processing.py, equations.py
  services/        Business logic: book_service, document_service, storage_service,
                   file_validation, document_processing_service, job_runner
  parsing/         Parsers and OCR, independent of the UI and the database:
                   models.py (output models), base.py, parser_factory.py,
                   pdf_parser.py, epub_parser.py, docx_parser.py, ocr_service.py,
                   scan_detection.py, text_normalizer.py, omml.py (Word equations)
  parsing/equation/  Equation extraction: equation_service.py (orchestration), pdf_equations.py,
                   equation_extractor.py (EPUB/DOCX), equation_ocr.py (math OCR engines),
                   latex_normalizer.py, equation_validator.py, mathml_converter.py,
                   equation_detector.py, math_symbols.py, equation_models.py
  repositories/    Database queries
database/
  connection.py    Engine, sessions, connection check
  models/          SQLAlchemy models (Book, Document, ProcessingRun, ProcessingError, ExtractedBlock, Equation)
  migrations/      Alembic migrations (configured by alembic.ini)
  init/            Scripts run when the Docker database is first created
ingestion/         Reserved for later phases
storage/           Default location of uploaded files (git-ignored)
tests/             pytest suite
tools/             Quality measurement: evaluate_equations.py, compare_math_engines.py
```

The layers are kept separate: the UI calls services, services call parsers, repositories and the storage backend, and only repositories talk to the database. Parsers return structured results (`ParsingResult`) and never touch the database or the UI. Storage sits behind the `StorageBackend` interface in `storage_service.py`, so cloud storage can be added later.

## Known limitations

- A completed document cannot be re-processed yet (for example after changing OCR settings); only failed documents can be retried.
- Chapter detection, section hierarchy and passage segmentation are not implemented (Day 4). `section_title` is only the nearest preceding heading.
- Page headers and footers detected by Docling are left out of PDF text. Text inside images that are not full-page scans is not OCR'd.
- DOCX and EPUB have no page numbers; blocks are located by element and chapter references instead.
- One document is processed at a time, and the work runs inside the app process: restarting the app interrupts a run (it can then be retried).
- Equations: PDF inline formulas are not detected (only display equations); Docling's layout model misses some formula regions (4 of 20 in the synthetic test above); equation images in PDFs are recognized only inside regions the layout model finds; a model can still misread symbols (for example `x` as `X`), which is why unconfirmed model output is marked `uncertain`. Handwriting and degraded scans were not evaluated. Only the equation types listed in the table were tested; chemistry, units, and multi-line derivations are stored as ordinary equations without further structure.
- On scanned PDFs, an equation that layout analysis labels a picture (rather than a formula) is still read by Tesseract as garbled text in the page's text blocks; the equation itself is recovered separately in the Equations tab. The text blocks for such regions are not cleaned up.
- Docling can place a paragraph slightly out of order around formulas, so the "surrounding text" of an equation (taken from neighbouring blocks) can occasionally be off by a block.
- Re-extracting equations re-reads the document, so it takes as long as the first extraction.
