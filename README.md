# Book2Brain

Turn books into structured, traceable text for AI work. A book is uploaded as a PDF, EPUB or DOCX file, processed into clean text, a chapter/section hierarchy, paragraphs and AI-ready **passages**, and everything is stored in PostgreSQL with its original document, page and position, so every passage can be traced back to the source (with pgvector installed for later phases).

**Release v0.1.0 — Phase 1 foundation.** Upload a book, process it in one step, browse its chapters and sections, read its passages with their page references, and recover from failures without uploading again.

**Phase 2 — knowledge extraction (complete, with documented limits).** Passages are turned into typed, source-linked knowledge (concepts, principles, methods, frameworks, evidence, examples, equations) and relationships by a configurable LLM, with full provenance, retries, resumable runs and review flags, and can be inspected on the *Extraction* and *Knowledge* screens. Day 5 added the evaluation method and tools described in [Evaluating extraction](#evaluating-extraction-phase-2-day-5). Start with the [Phase 2 completion report](docs/phase2/completion-report.md). Retrieval, search and AI features are later phases (see [Known limitations](#known-limitations)).

## What it does

| Capability | Where |
|---|---|
| Upload PDF / EPUB / DOCX with validation, SHA-256 duplicate detection, original file kept | **Upload** |
| One processing pipeline: parse, OCR where needed, extract text, detect structure, identify paragraphs, generate passages, validate, with live stage tracking, failure reporting and retry | **Library**, **Processing** |
| Book library with status, stage, editions (several documents per book), delete | **Library** |
| Structure explorer: front matter, parts, chapters, sections, appendices, back matter, page ranges | **Structure** |
| Passage viewer: full passage text, chapter/section context, page references, navigation | **Passages** |
| Equation extraction as LaTeX with source location | **Equations** |

**Stack:** Python 3.12, Streamlit, PostgreSQL (+ pgvector), SQLAlchemy 2, Pydantic 2, Alembic, Docling (PDF/DOCX), EbookLib (EPUB), Tesseract (OCR), pytest. No task queue: processing runs in a background thread of the app.

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

This runs the migrations in `database/migrations/`, creating the `books`, `documents`, `processing_runs`, `processing_errors`, `extracted_blocks`, `equations`, `structural_nodes`, `paragraphs`, `passages`, `passage_paragraphs` and `structure_analyses` tables and enabling the pgvector extension. Run it again after pulling new migrations; it is safe to repeat.

### 6. Launch the application

```bash
streamlit run app/main.py
```

Open http://localhost:8501. The page shows the title, version and database status, and a screen selector:

- **Library** — every book and document with its status and current stage, the editions of each book, and actions: open the structure or passages, start/retry processing, see details, delete a document (deleting a book's last document removes the book). It refreshes itself while anything is processing.
- **Upload** — choose a PDF, EPUB or DOCX file, enter the book metadata (title is required; author, subtitle, publisher, year, ISBN, language and description are optional) and upload. **Start processing after upload** is on by default.
- **Structure** — the explorer: pick a book and one of its documents, expand and collapse the hierarchy, select a chapter or section. See [Book structure](#book-structure-and-passages).
- **Passages** — the viewer: pick a chapter and section, open a passage, read it in full with its page references and move to the previous/next one.
- **Processing** — details of a document's run: stages, errors, the extracted text preview and OCR information. See [Processing](#processing).
- **Equations** — review the equations found in a processed document. See [Equations](#equations).

The book, document and section you choose are remembered when you move between screens ("Open structure" in the Library lands on that document; "View passages of this section" lands on that section's passages).

## Using the app

1. **Upload a book** (Upload). A new title creates a book; the same title and author with another file adds a document ("edition") to the existing book. A file already uploaded is rejected as a duplicate.
2. **Monitor processing** (Library). The status reads *Uploaded*, *Queued*, *Processing: <stage>*, *Completed* (or *Completed with warnings* when some scanned pages could not be read) or *Failed during <stage>*. The Processing screen shows every stage with its outcome.
3. **Explore chapters and sections** (Structure). Front matter, back matter and parts start open; chapters are collapsed. Only structure that exists is shown. Each node shows its type, page range, confidence and passage counts; ⚠️ marks nodes with warnings.
4. **Inspect passages** (Passages). Choose a chapter and section ("Whole chapter" includes its sections). Open a passage to read the complete stored text, where it comes from and the paragraphs it was built from.
5. **Page references.** Pages are the PDF's own page numbers (the first page is 1), copied from the extracted text; they are not the numbers printed on the page. EPUB and DOCX have no fixed pages, so their passages say *page unknown*. A page range marked *partly known* means only some of the source text had a page. No page is ever guessed.
6. **When processing fails.** The Library and Processing screens show the failed stage and a plain-language message (for example *Structural detection failed.*). Press **Retry processing**: if the text was already extracted, the same run resumes from the failed stage without parsing the file again. Technical details are in the application log and in the Processing screen's error entries.
## Uploads

- Allowed types: `.pdf`, `.epub`, `.docx`. The file **content** is checked, not just the extension or the browser-supplied MIME type.
- Files above `MAX_UPLOAD_SIZE_MB`, empty files and invalid files are rejected with a clear message.
- Every file is hashed with SHA-256. Uploading the same file again is rejected as a duplicate: no second copy is stored and no second record is created.
- A book is identified by its title and author (case-insensitive). Uploading another format of the same book (same title and author) attaches a new document to the existing book; the existing book's metadata is left unchanged.
- Book metadata kept: title, author, subtitle, publisher, publication year, ISBN, language, description. Per document: original filename, format, size, SHA-256, upload time and a document id. Optional fields can be left empty; nothing is read from the file at upload time, and missing metadata never blocks an upload.
- Files are stored under `STORAGE_ROOT/original/<document-uuid>/<document-uuid>.<ext>`. The original filename is kept in the database only. If saving the file or the database record fails, neither is kept.

## Processing

Press **Start processing** in the Library (or tick it when uploading, or use the Processing screen). The work runs in a background thread, so you can leave the page; the screens refresh themselves while a run is active. The first stages turn the stored file into an ordered list of **blocks** (headings, paragraphs, list items, tables, captions) saved in PostgreSQL; the rest of the pipeline builds structure and passages from them.

| Format | Parsed with | Source location kept on every block |
|---|---|---|
| PDF | Docling (structure and text), Tesseract (scanned pages) | page number, position on the page, element reference |
| EPUB | EbookLib (spine items in reading order) | chapter/spine item id, file name, chapter title, element index |
| DOCX | Docling | element reference and index; **no page numbers** (Word has no fixed pages) |

Page numbers are only stored where the format has real pages; they are never guessed. Blocks also carry a heading `section_title` (the nearest heading before them) and, for headings, the heading level. A block the parser reports across a page break keeps its first page and the list of pages, which the structure step uses for page ranges.

### Pipeline, status and stages

Processing is one pipeline (`ProcessingPipeline.process_document(document_id)` in `app/services/pipeline_service.py`); the screens only start it and display its state.

```
document_registration → parsing → ocr (scanned pages only) → normalizing → text_extraction
  → equation_extraction → structural_detection → paragraph_identification
  → passage_segmentation → persistence → validation → completed
```

| Stage | What happens |
|---|---|
| `document_registration` | The document, its book and the stored original file are checked. |
| `parsing`, `ocr`, `normalizing` | The Day 3 parser extracts text blocks with their pages and positions; Tesseract reads scanned pages; text is normalized. |
| `text_extraction` | The blocks are saved (one transaction). From here a failure keeps the text. |
| `equation_extraction` | Equations are extracted (own status; they never decide whether the run succeeds). |
| `structural_detection`, `paragraph_identification`, `passage_segmentation`, `persistence` | The Day 4 services build the hierarchy, paragraphs and passages and save them in one transaction. |
| `validation` | `IntegrityValidator` checks the saved data (below). |
| `completed` | Set only after every stage, including validation, succeeded. |

**Status model.** A document is `uploaded` → `pending` → `processing` → `completed` or `failed`. The run records its `stage` separately (and keeps the failed stage when it fails), `started_at`, `completed_at`, `updated_at`, `retry_count`, an error message and `stage_results`, a per-stage record (status, times and stage-specific numbers). Every attempt is a processing run; failures are `processing_errors` rows with the run, document, stage, attempt number, error type, a user-facing message and developer diagnostics.

- Nothing is reported as completed before the structure and passages are saved *and* validated.
- A document is **not** completed if almost nothing was extracted (`MIN_CONTENT_CHARACTERS`).
- **Retry.** If the failure was after `text_extraction`, retrying **resumes the same run** (`retry_count` + 1) from the failed stage: the file is not parsed again and nothing is duplicated, because structure, paragraphs and passages are replaced in one transaction. If only `validation` failed, the saved structure is reused. If parsing failed, a retry starts a new run; the failed one stays as history.
- If the app stops mid-run, the run is treated as interrupted after `PROCESSING_STALE_MINUTES` and can be retried (and resumed, if its text was saved).
- A completed document cannot be processed again; use **Detect structure again** or **Regenerate passages** (Structure screen) to rebuild derived data from the stored text.

**Data integrity validation** (`app/services/integrity_service.py`) reads the saved records back and checks relationships, not counts (pages, paragraphs, nodes and passages are different things): the document belongs to its book and the run to its document; the file is in storage; text is stored and block pages are within the document; every node, paragraph and passage belongs to the right document and book; parents are in the same document, one level up and earlier in order; node sequence and passage sequence are consecutive; no duplicate nodes; page ranges are valid and within the document; passages are non-empty, counted, linked to paragraphs of their own section, and every section with text has passages; and the structure step's own validation found no errors. Any error fails the run at `validation` (the data stays for inspection); warnings (such as unreadable scanned pages) are kept with the run.

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

## Book structure and passages

The pipeline builds this for every book it processes (stages `structural_detection` to `persistence`). Nothing is parsed again: the structure services read the text blocks stored by the text stages, with their page numbers and positions, and build three separate layers. The **Structure** screen can also rebuild them for a document on demand (**Detect structure** / **Detect structure again**), which is how a book processed before this feature gets its structure.

| Layer | What it is | Table |
|---|---|---|
| **Structure** | The book's own organisation: front matter, parts, chapters, sections, subsections, appendices... Any depth, linked parent to child. | `structural_nodes` |
| **Paragraphs** | The source text of each node, one record per paragraph, list, table, equation or caption, with pages, offsets and bounding boxes. | `paragraphs` |
| **Passages** | Derived units sized for AI processing, linked to their node and to the paragraphs they were built from (`passage_paragraphs`). | `passages` |

Structure belongs to the *document*, not just the book, because editions differ in pagination and layout. `structure_analyses` holds one status/summary/validation record per document.

### The Structure and Passages screens

- **Document selection**: pick a book, then one of its documents (editions have their own structure, page numbers and passages; nothing is shown from another edition).
- **Table of contents**: an outline you can expand and collapse (front matter, back matter and parts start open, chapters collapsed). Selecting a node shows its type, pages, confidence and counts, the sections it contains, its **paragraphs** (source text) and, in a separate tab, a preview of its **passages**; **View passages of this section** opens the Passages screen on that section with token count and original page range. "Include subsections" widens either list to the whole subtree. Nodes with warnings are marked ⚠️.
- **Book details**: the title, subtitle and author *detected* next to the ones you entered, with confidence.
- **Warnings**: structured validation errors/warnings/info and the list of low-confidence or ambiguous nodes.
- **Passage settings**: target/min/max size, overlap and tokenizer. **Regenerate passages** replaces only the passages: the file is not parsed again and structure is not detected again.

### How structure is detected

Detection is deterministic; an LLM is optional and only sees what remains ambiguous (`app/structure/`).

1. **Table of contents** (`toc.py`): a "Contents" heading followed by entry lines (also dot-leader lines and table-shaped contents) is recognised. Those lines stay in the book as content of a *table of contents* node but are never mistaken for headings, and they corroborate real headings found later.
2. **Candidate scoring** (`heading_detection.py`): every block gets a score from independent evidence: the parser's own heading label, numbering patterns, well-known heading words, text length, capitalisation, sentence shape, size relative to body text (estimated from the bounding boxes), centring, being first on the page, being an isolated short line before body text, agreement with the table of contents, repeated numbering sequences and a style shared with other headings. Penalties apply to sentence-like text, math symbols, code-like lines, running page headers (the same text on nearby pages) and captions. Scores at or above `STRUCTURE_HEADING_ACCEPT_THRESHOLD` (0.70) are confident headings; scores between `…REJECT_THRESHOLD` (0.45) and that are **ambiguous**: kept, typed `other_heading` and flagged with a warning; lower scores stay body text. A rejected parser heading is kept as a paragraph with an info note.
3. **Numbering** (`numbering.py`): `CHAPTER 1`, `Chapter One`, `Chapter 1: Title`, `1. Title`, `1.1 Title`, `1.1.1.1 Title`, `PART II — Title`, `Part Three: Title`, `Appendix A — Title`, with arabic, roman and written numbers. A label alone (`CHAPTER 1`) followed by a title line is merged into one heading. Numbered headings are not trusted blindly: `Chapter 3 shows that…`, `2 Peter wrote…` and lone numbered lines score low.
4. **Unnumbered chapters**: styles (parser heading level plus size cluster) are ranked relative to the styles that do have numbering; with no numbering anywhere the most prominent style becomes the chapter level, the next sections, and so on.
5. **Front and back matter**: the title, subtitle and author are scored from the opening pages (heading label, size, file metadata), so a cover, blank page or half-title before the title page is fine; copyright pages are recognised from © / ISBN / "all rights reserved". Preface, foreword, acknowledgements, introduction, contents and dedication before the first chapter go under a *Front Matter* node; appendices, notes, glossary, bibliography, index and "about the author" after it go under *Back Matter*. In a numbered book `1. Introduction` is chapter 1, not front matter.
6. **Optional LLM** (`llm.py`): only ambiguous candidates are sent, in batches, with their surroundings; the model classifies existing text, it cannot add headings or rewrite titles (a proposed title must be part of the original heading). Failures are logged and reported as a warning, and the deterministic result stands. Off by default.

Node types: `book_title`, `front_matter`, `back_matter`, `preface`, `foreword`, `acknowledgement`, `introduction`, `part`, `chapter`, `section`, `subsection`, `subsubsection`, `appendix`, `notes`, `bibliography`, `glossary`, `other_heading`, plus `table_of_contents`, `copyright`, `dedication`, `index` and `about_author`. The column is a plain string, so adding a type needs no migration. A node keeps its `original_title` (the heading as written, even over two lines), a normalised `title`, `number`, `level` (depth), document-order `sequence`, pages, offsets and `confidence`; `metadata` records the signals, style and LLM reasoning. Nothing is dropped: text before the first heading goes to the front-matter node, and a document with no headings at all is kept under one node with a warning.

### Paragraphs

A paragraph that continues across a page break (the first block ends mid-sentence, the next starts lowercase on the following page) is stored once with its full text and page range (`metadata.continued_across_pages`, `block_spans` with each block's page and character range). Runs of list items become one list paragraph; tables, equations and captions stay separate. Each paragraph keeps its source block ids/sequences, bounding boxes, equation ids and figure/table references. **Offsets** are positions in the document's canonical text (the stored blocks joined with a blank line); for a paragraph merged from several blocks they give the source span, and `metadata.offsets_exact` says whether the text is exactly that slice.

### Passages

`passages` are generated per structural node, never across nodes. Defaults (all settings in `.env`): target **750**, minimum **500**, maximum **1000** tokens, **12%** overlap (never more than 15%).

- Boundaries fall between paragraphs; a node shorter than the target is one passage; a short last passage is merged into the previous one if the maximum allows.
- An equation stays with the paragraph before it and the one after it; a table stays whole (a table above `PASSAGE_TABLE_SPLIT_FACTOR` × maximum is split between rows); a figure caption stays with the preceding content and a table caption with its table; text ending in a colon stays with the list/table/equation it introduces. Such units may exceed the maximum (`metadata.oversized` says why).
- A paragraph over the maximum is split at sentence boundaries (abbreviations such as "e.g." or "Fig." do not end a sentence); a single sentence over the maximum is split at words. Each piece records which part of the paragraph it holds (`passage_paragraphs.text_start/text_end`).
- **Overlap** repeats the last sentences (or whole last paragraphs) of the previous passage of the *same* node, never an equation or table. It is marked so it cannot be extracted twice: `metadata.overlap_chars` is the length of the repeated prefix (skip `text[:overlap_chars]`), and `passage_paragraphs.is_overlap` flags paragraphs that are present only as overlap.
- **Tokens** are counted with `PASSAGE_TOKENIZER`: `approx` (characters ÷ `PASSAGE_CHARS_PER_TOKEN`, default 4), `words`, `whitespace`, or `tiktoken` (needs `pip install tiktoken`; not a dependency). The configuration used is stored on the analysis record.

### Pages and traceability

Every passage records `book_id`, `document_id`, `structural_node_id`, `sequence` (inside the node) and `document_sequence`, its text, `token_count`, `page_start`/`page_end`, offsets and `metadata` with the paragraph ids, page numbers, bounding boxes (`page`, `bbox`, `origin`), equation ids and figure/table references, so the original PDF page and its image regions can be fetched later. Pages are copied from the Day 3 blocks, never derived from text length. `page_status` is `exact`, `partial` (some blocks without a page) or `unknown`: EPUB and DOCX have no pages, so their pages are `NULL`/`unknown` rather than invented. Page numbers are PDF page indices (starting at 1), not the numbers printed on the page.

### Validation

`app/structure/validation.py` returns structured issues (`error`, `warning`, `info`, a stable `code`, the entity concerned) that are saved on the analysis record and shown in the Warnings tab.

- **Hierarchy**: valid ids and parents, no cycles, levels equal parent level + 1, document order, duplicate nodes, children inside their parent's blocks/offsets, numbering order/gaps, skipped levels, empty nodes.
- **Pages**: within the document's page count, start ≤ end, children within parents, missing page references on paginated documents.
- **Passages**: valid book/document/node, ordering and sequence, token counts recomputed, passage pages consistent with their paragraphs, overlap taken from the previous passage and within policy, no empty passages.
- **Content**: every extracted block ends up exactly once (as a heading or in a paragraph), paragraph text equals its source blocks, a node's passages together contain exactly its text (nothing omitted or duplicated), headings are not repeated as body text, ambiguous or low-confidence classifications produce warnings. Unclassified content is always kept.

### Running it again

Detection replaces a document's nodes, paragraphs and passages in one transaction, so repeating it never leaves duplicates and a failure keeps the previous result (the failure is recorded). **Regenerate passages** only rebuilds passages from the stored paragraphs. If a document is processed again after its structure was detected, the tab tells you the structure is out of date.

### Structure settings

`STRUCTURE_HEADING_ACCEPT_THRESHOLD`, `STRUCTURE_HEADING_REJECT_THRESHOLD`, `STRUCTURE_FILL_EMPTY_BOOK_METADATA`, `STRUCTURE_METADATA_MIN_CONFIDENCE`, `STRUCTURE_LLM_ENABLED`, `STRUCTURE_LLM_BASE_URL`, `STRUCTURE_LLM_MODEL`, `STRUCTURE_LLM_API_KEY`, `STRUCTURE_LLM_TIMEOUT_SECONDS`, `STRUCTURE_LLM_MAX_CANDIDATES`, `STRUCTURE_LLM_BATCH_SIZE`, `PASSAGE_TARGET_TOKENS`, `PASSAGE_MIN_TOKENS`, `PASSAGE_MAX_TOKENS`, `PASSAGE_OVERLAP_RATIO`, `PASSAGE_TOKENIZER`, `PASSAGE_CHARS_PER_TOKEN`, `TIKTOKEN_ENCODING`, `PASSAGE_TABLE_SPLIT_FACTOR` (see `.env.example`). Metadata you entered is never overwritten: detected values are shown beside it, and with `STRUCTURE_FILL_EMPTY_BOOK_METADATA=true` only *empty* subtitle/author fields are filled, at or above the confidence threshold and never if that would duplicate an existing book (title + author).

The LLM works with any OpenAI-compatible `/chat/completions` endpoint (hosted, or a local server such as Ollama or vLLM): set `STRUCTURE_LLM_ENABLED=true`, `STRUCTURE_LLM_MODEL` and, for hosted services, `STRUCTURE_LLM_BASE_URL` and `STRUCTURE_LLM_API_KEY`. Without them, no network call is ever made.

## Knowledge model (Phase 2)

Phase 2 turns passages into reusable knowledge. Day 1 provides the data model and validation, Day 2 the LLM extraction layer (`app/extraction`) and Day 3 the pipeline that runs it over stored passages, with the *Extraction* screen and a command line (see below). There is no knowledge browser, search or relationship extraction yet.

**Categories** (`app/knowledge/types.py`): concept, principle, method, framework, evidence, example, equation.

**Typed content.** Each category has its own validated Pydantic model in `app/knowledge/content.py` (for example a concept has a name, definition, explanation, characteristics, related concepts and examples; a method has purpose, preconditions, ordered steps, inputs, outputs, expected results and limitations). Models forbid unknown fields, trim text, reject blank required text and never use `None` for lists. `knowledge_type` identifies the category when content is read from JSON.

**Common envelope.** `KnowledgeUnitCreate` in `app/knowledge/schemas.py` wraps any content with what all units share: a UUID, book and document, one or more source passages (each `primary` or `supporting`, optionally with the quoted words), chapter and section, page range and page status, confidence (0-1, optional), explicit uncertainty, validation status, review status and JSON metadata. `KnowledgeRelationshipCreate` and `ExtractionRunCreate`/`ExtractionRunRead` follow the same pattern. `app/knowledge/records.py` converts between these schemas and the database records.

**Storage.**

| Table | Notes |
|---|---|
| `knowledge_units` | `details` (JSONB) holds the full typed content; `title`, `content` and `summary` are projections kept as columns for listing and search. `schema_version` versions the content format. The page range is stored on the unit itself. |
| `knowledge_unit_passages` | Many-to-many to `passages`, with position, role and quote. At most one `primary` per unit. |
| `knowledge_relationships` | `source <type> target`, with explanation, confidence and run. Both ends must belong to the same book (enforced by a composite foreign key). No self links; one row per source, target and type. |
| `extraction_runs` | Scope (book, optionally document and chapter/section), status (`pending`, `processing`, `completed`, `needs_review`, `failed`, `cancelled`), passages processed, units extracted, failure count, free-text model name and prompt version, times and error. |

**Relationship types** (`app/knowledge/relationships.py`): `supports` (evidence → concept/principle/method/framework), `illustrates` (example → ...), `implements` (method → framework/principle), `describes` (equation → ...), `part_of`, `depends_on`, `contradicts`, `related_to`. The allowed type combinations are checked with `check_relationship_types`; the database stores any combination of units in the same book.

**Equations are never pretended to be complete.** An equation keeps its `original_expression` as extracted and an optional `latex`; at least one is required. `extraction_status` (`complete`, `partial`, `ambiguous`, `failed`) has no default, and `reconstruction_status` records whether a clean form was rebuilt. An incomplete equation is marked uncertain automatically and cannot be validated as `valid`.

**Relation to Phase 1.** Knowledge survives regeneration of Phase 1 data. Deleting a document or book deletes its knowledge. Regenerating structure clears a unit's chapter/section references, and regenerating passages removes its passage links, but the unit stays and keeps its page range; `KnowledgeUnitRead.sources` is then empty (a unit whose sources were lost can be found by the missing links). New units must cite at least one passage. Apply the tables with `alembic upgrade head`; downgrading removes them (and any extracted knowledge) without touching Phase 1 data.

## Knowledge extraction (Phase 2, Day 2)

`app/extraction` sends one passage to an LLM and returns validated knowledge. It is independent of the database and of any single provider: it never reads or writes Phase 1 tables; the pipeline in the next section connects it to the stored passages and the screen.

```python
from app.config import get_settings
from app.extraction.service import ExtractionService
from app.extraction.settings import ExtractionSettings, load_model_registry
from app.extraction.results import ExtractionRequest, PassageInput

settings = get_settings()
service = ExtractionService(load_model_registry(settings), ExtractionSettings.from_settings(settings))
result = service.extract(ExtractionRequest(
    knowledge_type="concept",
    passage=PassageInput(id=passage.id, text=passage.text, page_start=passage.page_start, page_end=passage.page_end),
))
result.status        # succeeded | needs_review | failed
result.items         # validated content, sources, pages, confidence, uncertainty
result.execution     # model, prompt version, parameters, attempts, tokens, cost, timing
unit = result.items[0].to_unit_create(book_id, document_id)   # the Day 1 schema, ready to store
```

### Configuring the LLM provider

1. **Define models** in `config/llm_models.json` (path: `EXTRACTION_MODELS_FILE`). One entry per model: `id`, `provider` (`ollama`, `openai`, `anthropic`, `stormy` or `openai_compatible`), `model`, `deployment` (`local` or `hosted`), `base_url` (local and compatible servers), `context_window`, `temperature`, `max_output_tokens`, `structured_output` (`json_schema`, `json_mode` or `none`), `supports_temperature`, `extra_params`, `unsupported_params`, `timeout_seconds`, optional prices for cost estimates, and `active`. The file ships with a local Ollama model, an OpenAI model and an inactive Anthropic model as examples.
2. **Select the model** with `EXTRACTION_MODEL_ID=<id>`. Use `EXTRACTION_MODEL_OVERRIDES={"equation":"<other id>"}` to give a category its own model. Nothing is sent anywhere until a model is selected.
3. **Provide the API key through the environment only.** A hosted entry names the variable (`"api_key_env": "OPENAI_API_KEY"`); put the value in your shell or in `.env` (which is git-ignored). Keys are never written to configuration files, the database, logs or results, and a configuration whose `api_key_env` is not a variable name (for example a pasted key) is rejected.
4. Optional: `EXTRACTION_MAX_RETRIES` (default 2), `EXTRACTION_RETRY_BACKOFF_SECONDS`, `EXTRACTION_MAX_BACKOFF_SECONDS`, `EXTRACTION_RAW_RESPONSE_MAX_CHARS`, `EXTRACTION_PROMPT_VERSIONS={"concept":"1.0.0"}`.

**Switching and testing models.** The shipped file lists the Stormy models (Claude Sonnet/Haiku/Opus 5.5, Llama 3.3 70B, GPT-6 Luna, Pixtral Large, Palmyra X5; the image model is inactive). Put the PAT in `.env` as `STORMY_API_TOKEN=...` (git-ignored), then:

```bash
python -m app.extraction.check                              # list models and whether each is ready
python -m app.extraction.check --model stormy-claude-haiku-5.5            # test one connection
python -m app.extraction.check --model stormy-claude-haiku-5.5 --extract  # plus a sample extraction
python -m app.extraction.check --all                        # test every ready model
```

To switch the model used by extraction, change `EXTRACTION_MODEL_ID` (for example `stormy-claude-sonnet-5.5`), or in code use `registry.select("<id>")`, which returns a registry using another model. The context windows in the file are assumptions; adjust them to the real limits.

`service.check_ready(category)` validates the selected model (exists, active, credentials present) before a run; `extract` reports an invalid configuration as a failed result rather than raising.

**Model capabilities are configuration, not assumption.** `json_schema` and `json_mode` use the provider's structured output; `none` relies on the prompt. Either way the response is parsed and validated here. `anthropic` and `stormy` have no JSON mode here, so they must use `none`. The `stormy` provider posts to `https://stormy-api.dolby.net/api/v1/messages/` with the PAT from the variable named in `api_key_env` (use `STORMY_API_TOKEN`); that API takes no system prompt or generation parameters, so the system prompt is prepended to the input and `temperature`/`max_output_tokens` are not sent. Models that reject a temperature set `supports_temperature: false`; parameters a model does not accept are listed in `unsupported_params` and not sent. If the prompt does not fit `context_window` minus `max_output_tokens`, context passages are dropped (furthest first) and recorded; if the target passage alone does not fit, the extraction fails with `context_window_exceeded`. A new provider is one class in `app/extraction/providers.py` plus an entry in `PROVIDER_FACTORIES`.

### Running extraction on a book (Day 3 pipeline)

Apply the migrations first (`alembic upgrade head`). The pipeline is `KnowledgeExtractionService` in `app/services/knowledge_extraction_service.py`; the screen, the command line and your own code all use it.

**What a run does.** It validates the model, retrieves the passages of the scope in reading order, prepares each passage's context for the model, extracts every requested category with the versioned prompts, validates the answers, converts them to knowledge units, skips duplicates, stores everything, and records the model, prompt and attempts of every call plus the run's progress. One passage can yield several units and several categories.

| Step | Where |
|---|---|
| Passage retrieval (book, document, chapter, section; reading order; neighbours; outline) | `app/repositories/passage_repository.py` |
| Context preparation (book, chapter, section titles, pages, neighbours, fit to the model's window) | `app/services/context_preparation.py`, `app/extraction/context.py` |
| Orchestration, status and results | `app/services/knowledge_extraction_service.py` |
| Persistence, duplicate detection, run queries | `app/repositories/extraction_repository.py`, `app/knowledge/fingerprint.py` |
| Background execution for the screen | `app/services/extraction_job_runner.py` |

**Using it from code.**

```python
service = create_extraction_service(session)               # models file, default model and .env credentials
service.available_models()                                  # id, provider, deployment, active, ready, reason
service.extract_book(book_id, ["concept", "method"])        # also extract_chapter(node_id, ...), extract_section(node_id, ...)
run_id = service.create_run(scope, categories, model_id="stormy-claude-haiku-5.5")   # validate and record, then
status = service.execute_run(run_id, progress=callback)     # run it (the screen does this on a worker thread)
service.get_run_status(run_id)                              # RunStatus: status, progress, counts, model, error
service.list_runs(book_id); service.cancel_run(run_id)
service.get_book_knowledge(book_id, knowledge_type="concept", run_id=run_id)
```

**Models.** A run uses one model configuration: the `model_id` you pass, otherwise the application default (`EXTRACTION_MODEL_ID`); with neither there is a clear configuration error. The configuration must exist, be active and have its credentials before anything is recorded. Local and hosted models go through the same layer. Per-category overrides (`EXTRACTION_MODEL_OVERRIDES`) apply only when no model is chosen for the run; no automatic routing exists.

**Context.** Each passage is sent whole with its book, chapter and section, pages and (by default) one neighbouring passage on each side (`EXTRACTION_CONTEXT_PASSAGES`). If the prompt would not fit the model's context window (minus the reserved answer and `EXTRACTION_CONTEXT_SAFETY_MARGIN`), neighbouring passages are dropped first, furthest first; the reduction is stored on the execution (`context`) and flags it for review. A passage that cannot fit on its own is never truncated: that extraction fails with `context_window_exceeded` and the run continues.

**Statuses.** A run is `pending`, `processing`, then `completed` (nothing to look at), `needs_review` (some extractions are uncertain, flagged or failed), `failed` (everything failed, or the run stopped after `EXTRACTION_MAX_CONSECUTIVE_FAILURES` failures in a row, default 5, 0 = never) or `cancelled`. A failed passage never ends the run by itself. Per passage and category, the `extraction_executions` table holds `succeeded`, `needs_review` or `failed`. The run stores the total, processed, failed (executions), skipped and duplicate counts, attempts, retries, validation failures, tokens, estimated cost, the model configuration snapshot and the error. A failed, cancelled or interrupted run can be resumed (`execute_run` again) and repeats only what is missing. One extraction per book runs at a time.

**Idempotency and duplicates.** Passage and category pairs that already have a usable result are skipped unless `force` is set. Whatever is extracted, a unit is not stored again when the same primary source passage and category already have it: identical normalised content (`content_fingerprint`), near-identical wording (85% of words) or the same title with similar wording (50%). Another passage is never treated as a duplicate: the same concept in two passages is two units, with `same_title_as` in the later unit's metadata so a later step can relate them. Duplicates skipped are counted and noted on the execution.

**The screen.** The *Extraction* screen lets you pick a book and document, a model (only usable ones; unavailable ones are listed with the reason), a scope (whole document, a chapter or a section), the categories and options, shows the plan (passages, model calls, already extracted), starts the run on a background worker and shows the model in use, progress, counts, errors, cancel and resume, earlier runs and the first extracted units.

**Reviewing the results (Knowledge screen).** The *Knowledge* screen (`app/ui/knowledge.py`, backed by `app/services/knowledge_service.py`) is where you inspect what the model extracted. For a document it shows counts (units, need attention, approved, rejected, unreviewed), filters by category, validation, review status, extraction run, free-text search and "only uncertain or unvalidated", and a paged list in reading order. Opening a unit shows its full typed content (a concept's definition and examples, a principle's statement and conditions, an equation's expression and LaTeX, ...), the model's own doubts, its place (chapter, section, pages), each source passage with the quoted words (checked against the passage text) and the passage itself, the same title found elsewhere, and *How it was extracted*: model, prompt version, attempts, tokens, why it was flagged and the validation warnings. Approve, Reject, Needs changes or Reset set `review_status` and store an optional note in the unit's metadata. From code: `KnowledgeService(session).list_units(KnowledgeFilter(...))`, `.detail(unit_id)`, `.review(unit_id, "approved", note)`.

**Command line.**

```bash
python -m tools.extract_knowledge                                  # books, passage and unit counts, models
python -m tools.extract_knowledge --book "Latent" --outline        # chapters and sections with ids and passage counts
python -m tools.extract_knowledge --book "Latent" --model stormy-claude-haiku-5.5 --types concept,principle --limit 3
python -m tools.extract_knowledge --chapter <node id> --types all --yes
python -m tools.extract_knowledge --section <node id> --types equation --dry-run    # call the model, store nothing
python -m tools.extract_knowledge --book "Latent" --runs           # recent runs; --status <run id> for one
```

Other options: `--document`, `--offset`/`--limit` (passages), `--context N`, `--force`, `--yes`. It prints the plan and asks before more than 10 calls.

**What is stored.** `extraction_runs` (one per run), `extraction_executions` (one per passage and category: model configuration snapshot, provider, model id and version, prompt name and version, generation parameters, timing, attempts, retries, validation failures, tokens, estimated cost, context handling, quality indicators, issues, review reasons, last raw response), `extraction_attempts` (one per model call, with its own model, prompt, parameters, outcome, raw response and issues) and `knowledge_units` (typed content, source passage links with quotes, chapter, section and pages, confidence, validation status, `extraction_run_id`, `extraction_execution_id`). Units arrive `unreviewed`. Deleting a run deletes its executions and attempts but keeps the knowledge; regenerating passages keeps executions and units. Phase 1 data is only read.

### Prompts and versions

Texts are in `app/extraction/prompt_texts.py` (one template per category; `1.0.0` is the first release, `1.1.0` adds the rule to extract only from the target passage and use neighbouring passages only for understanding; `1.2.0`, the current one, also tells the model which book it is reading; pin an older one with `EXTRACTION_PROMPT_VERSIONS`); lookup and rendering are in `app/extraction/prompts.py`. Every result records `<category>_extraction@<version>`. To change a prompt, add a new version rather than editing a released one.

### Validation, retries and review

| Stage | Question | Consequence |
|---|---|---|
| Structural (`parsing.py`) | Valid JSON matching the Day 1 schema? | error: correction attempt |
| Source (`validation.py`) | Do cited passage ids and pages exist? Pages are never invented; if the model gives none they are taken from the cited passages. | error: correction attempt |
| Semantic | Are quotes and equation expressions present in the cited passage? Pluggable via `SemanticValidator`. | warning: flagged for review |
| Quality | Completeness, citation accuracy, duplicates, unsupported content. No overall reliability score. | indicators only |

Invalid responses get a correction prompt listing the problems; failed provider calls (timeout, rate limit, 5xx) are resent after a back-off that honours `Retry-After`. One budget (`max_retries`) bounds both, and authentication or rejected-request errors are never retried. An exhausted budget returns a `failed` result flagged `needs_review`, with the last raw response and every attempt kept. A structurally valid result can still be `needs_review` (uncertain, inferred, incomplete equation, unsupported quote, dropped context). Incomplete equations are kept, never rejected.

### Execution metadata

`result.execution` records the model configuration id and a snapshot of it (no secrets), provider, the model identifier the provider reported, model version, prompt name and version, generation parameters, start/end times, duration, attempts, retries, validation failures, token usage and estimated cost, plus one `AttemptRecord` per call. `execution.run_fields()` returns arguments for `ExtractionRunCreate`. Because the snapshot is stored with the result, later configuration changes never alter finished records.

### Provenance, relationships, equations and review flags (Phase 2, Day 4)

Apply the migrations (`alembic upgrade head`, revision `1eb6b9dd22a0`). Existing units are backfilled: each gets a source snapshot and each earlier attempt its run and passage.

**Source provenance** (`app/services/provenance_service.py`). Every unit keeps its source passages with a reference type, chapter/section node, pages, the quoted words and their character range in the passage, and a verification status. `source_status` on the unit is `verified`, `invalid` or `unverified`. Verification checks that the passage exists and belongs to the book, that quotes appear in the passage text, that pages lie inside the passage's pages and that the quote range fits. Problems raise `review_flags`, never silent deletions. A `source_snapshot` (passage text excerpt, pages, node titles) is stored when the unit is saved, so the evidence stays readable if Phase 1 passages are regenerated. Re-verify from the *Knowledge* screen (**Verify sources**), `python -m tools.extract_knowledge --book "..." --verify-sources`, or `ProvenanceService(session).verify_unit(id)` / `.verify_book(id)`. Phase 1 data is only read.

**Relationships** (`app/services/relationship_service.py`, `app/services/relationship_extraction.py`). Types: `depends_on`, `supports`, `explains`, `derived_from`, `example_of`, `applies_to`, `part_of`, `related_to`, `contradicts` and the others in `app/knowledge/types.py`. Each relationship records its `origin` (`extracted` from the book, `ai_suggested`, `manual`, `system`), a `review_status`, an explanation, the source passages and an `evidence_quote`. An `extracted` relationship needs a passage and a quote; an `ai_suggested` one is flagged for review. Self links, unknown units and units of other books are rejected; duplicates (both directions for symmetric types) are not stored twice. Tick **Also find relationships** on the *Extraction* screen or pass `--relationships`: after a passage's units are stored, one extra model call (`relationship_extraction@1.0.0`) links them to each other and to up to `EXTRACTION_RELATIONSHIP_CONTEXT_UNITS` earlier units of the book. The plan shows these calls separately. A failed relationship step marks the passage `needs_review` rather than failing it.

**Equations** (`app/services/equation_service.py`). What the source says is kept apart from what the model wrote. An `equation_sources` row copies the Phase 1 original text, source LaTeX, page, crop reference and surrounding text, never model output. Its `capture_status` is `complete`, `text_only`, `incomplete`, `ambiguous`, `image_only` or `needs_review`; math validation is `not_validated` (no reconstruction or code generation exists yet). An equation Phase 1 found but the model skipped becomes a `source_derived` unit so it is not lost; image-only equations are kept with the reason. `equation_reconstructions` is placeholder structure for a later phase; nothing writes to it.

**Attempts, validation and review flags.** Every model call is an `extraction_attempts` row with its run, passage, status, a snapshot of the model configuration and a `validation_outcome`: `schema_valid`, `schema_invalid`, `source_invalid`, `unsupported_content`, `semantic_concerns`, `duplicate` or `needs_review`, the most serious applying. A `schema_valid` answer can still be wrong, so this is a lead for review, not a score. `review_flags` (`app/services/review_flag_service.py`) record what needs a person: uncertain or inferred extractions, low confidence (below `EXTRACTION_LOW_CONFIDENCE_THRESHOLD`), unsupported quotes, source problems, possible duplicates, reduced context, unresolved errors and repeated validation failures. A flag has a severity, belongs to one unit, attempt or relationship, is raised once per cause, and is resolved by you (**Resolve** on the *Knowledge* screen) or automatically when re-verification shows the cause is gone. This is not a review workflow: no assignment or approval chains.

**Resumable processing.** Each run keeps one record per passage (`extraction_passage_records`): `pending`, `processing`, `completed`, `needs_review`, `failed` or `skipped`, with attempts, units, duplicates and the error. The run's status is derived from these. After a failure, cancel, restart or crash, **Resume** on the *Extraction* screen (or `--resume <run id>`) continues with the failed, not reached and interrupted passages only; completed ones are not repeated. A run that still looks active is not taken over unless it is older than the active-run timeout or you confirm. `service.resume_run(run_id, takeover=False, redo_completed=False)`; `redo_completed=True` also repeats the finished passages. The per-passage list is on the *Extraction* screen (**Passages**) and `--passages <run id>`; `--status <run id>` also says whether the run can be resumed.

| Setting | Default | Meaning |
|---|---|---|
| `EXTRACTION_LOW_CONFIDENCE_THRESHOLD` | `0.5` | confidence below this raises a `low_confidence` flag |
| `EXTRACTION_RELATIONSHIP_CONTEXT_UNITS` | `10` | earlier units offered as relationship candidates (0-50) |

```bash
python -m tools.extract_knowledge --book "Latent" --types concept,principle --relationships --limit 3
python -m tools.extract_knowledge --passages <run id>          # status of every passage
python -m tools.extract_knowledge --resume <run id>            # continue a failed or interrupted run
python -m tools.extract_knowledge --book "Latent" --verify-sources
```

## Evaluating extraction (Phase 2, Day 5)

Schema-valid output is not necessarily correct, so Day 5 adds a repeatable way to measure extraction against **manually reviewed reference results** that are kept apart from any model output. Reports and findings are in [docs/phase2](docs/phase2/): [evaluation report](docs/phase2/evaluation-report.md), [model comparison](docs/phase2/model-comparison.md), [test and integrity results](docs/phase2/test-and-integrity-results.md), [performance](docs/phase2/performance.md) and the [completion report](docs/phase2/completion-report.md).

**Datasets** (`evaluation/datasets/*.json`, validated by `app/evaluation/dataset.py`):

| Dataset | Kind | What it is |
|---|---|---|
| `study_guide_v1` | inline | Twelve short passages written for the evaluation (every category, relationships, a complete, an incomplete and an image-only equation, a rejected claim, a passage with nothing to extract). They are loaded as a fresh book each run, so they can be run any number of times with any model. |
| `latent_ebm_v1` | stored book | Reference results for six consecutive passages of the processed book in the development database. It evaluates extractions that already exist (it cannot be re-run without disturbing the stored knowledge). |

Each reference unit lists acceptable names, the key facts it must carry (alternatives per fact), phrases it must not contain, and for equations the symbols, acceptable capture statuses and whether the source is damaged enough that the system must flag it. `review.status` says whether a person has signed the reference off (`draft` until then; both datasets are drafts prepared by the AI assistant and still need a human reviewer).

```bash
python -m tools.evaluate_extraction --list
python -m tools.evaluate_extraction --dataset study_guide_v1 --validate            # check a dataset, run nothing
python -m tools.evaluate_extraction --dataset latent_ebm_v1 --stored               # measure the extractions already stored (read only)
python -m tools.evaluate_extraction --dataset study_guide_v1 --model stormy-claude-haiku-5.5 --types all --relationships
python -m tools.evaluate_extraction --dataset study_guide_v1 --model MODEL_A --model MODEL_B    # compare models on identical passages
python -m tools.evaluate_extraction --dataset study_guide_v1 --model MODEL_A --repeat 3         # repeatability
```

A live run sends the dataset's passages to the selected model (a hosted one needs `--yes`), executes the **normal pipeline** (`KnowledgeExtractionService`), measures the stored result and writes `evaluation/results/<dataset>-<time>.md` and `.json` with the model configuration, generation parameters, prompt versions, run ids and timing of every run. Each run uses a fresh evaluation book (`[Evaluation] ...`, removed afterwards unless `--keep`, so it can be inspected on the *Knowledge* screen), because the pipeline never stores the same unit twice for a passage.

**What is measured and how it is counted** (every metric carries its numerator, denominator and definition in the report; a metric that cannot be measured is `n/a`, never zero):

| Dimension | Metrics |
|---|---|
| Completeness | recall of required units (overall, per category, in passages with several units), precision against the reference, passages that should stay empty, possible merges |
| Semantic accuracy | key-fact coverage (a lexical proxy, not proof of meaning), forbidden phrases, units flagged unsupported, relationship recall, type accuracy and precision |
| Source accuracy | right primary passage, pages inside the passage, units whose sources verified, quotes found, units whose primary source is the passage that was extracted |
| Structural validity | share of answers that matched their schema, extractions that succeeded on the first call, category accuracy |
| Duplication | redundant units, duplicates the pipeline refused to store |
| Equations | found, original expression kept, Phase 1 source kept, image and page kept, capture status as expected, flagged when the source is damaged |
| Operations | model calls, retry rate, failure rate, needs-review rate, time per passage, tokens, estimated cost (only when the provider reports them / prices are configured) |

Matching is lexical (names, key words, symbols), so the report lists every matched pair, every missed required unit and every unmatched extraction for a reviewer to read. Precision counts unmatched units only in passages whose reference is exhaustive.

**The Knowledge screen** now opens with an *Overview*: units by category, equations by capture status, relationships by type, open review flags, the outcome of every model call, and the extractions that failed (page, category, reason, model and prompt). It opens by itself when something failed.

**Other quality tools:** `python -m tools.benchmark_extraction` measures the pipeline's own time, queries and memory per passage with a scripted model on a separate `_benchmark` database; `python -m tools.check_migrations_on_copy` copies the development database, rolls the copy back one revision and upgrades it again, and verifies that Phase 1 data is unchanged and nothing was lost.

## Architecture

```
Streamlit screens (app/ui)  ->  services  ->  repositories  ->  PostgreSQL
                                   |  \
                                   |   parsers / OCR / structure algorithms (pure computation, no database)
                                   storage backend (original files, equation images)
```

- **Screens** (`app/ui/`) only call services and display results; they never run a pipeline stage themselves. `navigation.py` switches screens and carries the selection; `selection.py` is the shared book/document selector.
- **Services** (`app/services/`): `document_service` (upload, delete), `pipeline_service` (the orchestrator), `document_processing_service` (text stages), `structure_service` (structure, paragraphs, passages), `integrity_service` (validation), `library_service` (library read model), `job_runner` (background thread), `storage_service`.
- **Pure computation**: `app/parsing/` (parsers, OCR, equations) and `app/structure/` (heading detection, hierarchy, paragraphs, segmentation, validation, optional LLM). They know nothing about the database or the UI, so they are unit-testable alone.
- **Repositories** (`app/repositories/`) hold the queries. Sessions are created with `database.connection.session_scope` and every service commits explicitly.

### Database entities

| Table | Role |
|---|---|
| `books` | A logical book: title, author, subtitle, publisher, publication year, ISBN, language, description. Unique on title + author (case-insensitive). |
| `documents` | One uploaded file of a book (a format or an **edition**): filename, format, size, SHA-256 (unique), storage key, `processing_status`. |
| `processing_runs` | One processing attempt of a document: status, stage, progress, times, retry count, `stage_results`, text/equation outcomes. |
| `processing_errors` | A failure: run, document, stage, attempt, type, message, details. |
| `extracted_blocks` | The extracted text in order, with page, bounding box and source location. |
| `equations` | Extracted equations (LaTeX) linked to their blocks. |
| `structural_nodes` | The hierarchy of one **document** (front matter, parts, chapters, sections...), self-referencing, with pages, offsets and confidence. |
| `paragraphs` | Source paragraphs of a node. |
| `passages`, `passage_paragraphs` | Derived passages and the paragraphs each was built from (ordered, many-to-many, with overlap flags). |
| `structure_analyses` | One per document: structure and passage status, summaries and validation findings. |
| `extraction_runs` | Phase 2: one knowledge-extraction execution over a book, document or section: status, counts, model and prompt version, times, error. |
| `knowledge_units`, `knowledge_unit_passages` | Phase 2: extracted knowledge of any category and the passages supporting it (one or many). |
| `knowledge_relationships`, `knowledge_relationship_passages` | Phase 2: typed links between knowledge units of one book, with the passages that show them. |

### Book versus document

A **book** is the logical work; a **document** is one file of it. Several documents (a PDF and an EPUB, or two editions) can belong to one book, and each is processed independently: its own run history, blocks, structural nodes, paragraphs and passages, with page references that refer to *that file*. Structure and passages carry both `book_id` and `document_id`; screens always scope to the selected document, and a failure in one document never changes another's status. Deleting a document removes everything derived from it (cascading foreign keys) and its files; a book with no documents left is removed.

### Provenance

Original files are never modified. Extracted text keeps its source location; paragraphs and passages keep block ids, pages, bounding boxes and offsets into the document's canonical text. Later steps can therefore tell original source text, extracted text, OCR text (`extraction_method`/OCR summary on the run) and anything reconstructed apart; Phase 1 stores no reconstructed content.

## Developer guide

**Add a document parser.** Implement `DocumentParser._extract` in `app/parsing/` (return `Extraction` with `ContentBlock`s carrying page numbers only if the format has real pages), register the format in `parser_factory.py` and in the upload validation (`app/services/file_validation.py`), and add tests with a generated sample file. The pipeline needs nothing else.

**Add a processing stage.** Add the value to `ProcessingStage` (`database/models/processing.py`), add it to `RESUMABLE_STAGES` if a failed run can resume at it, and to the check constraint through an Alembic migration (`alembic revision`, then edit the constraint). Enter it in `ProcessingPipeline` with `self._enter(run_id, stage)`, record its result with `_finish_stage`, raise `PipelineStageError` for failures that need a specific message, and add its label to `STAGE_FAILURE_MESSAGES` and `app/ui/formatting.py`.

**Extend structure detection.** Heading evidence lives in `app/structure/heading_detection.py` (`_score`), numbering and keywords in `numbering.py`, placement of front/back matter in `hierarchy.py`. Add a test with a synthetic document in `tests/structure_helpers.py`; node types are plain strings, so a new type needs no migration.

**Change passage segmentation.** Sizes and overlap are settings (`PASSAGE_*`); the rules are in `app/structure/segmentation.py`. Passages can be rebuilt without re-detecting structure (**Regenerate passages**).

**Create a migration.** After changing `database/models/`, run `alembic revision --autogenerate -m "describe change"`, review the file (constraints are not always detected), run `alembic upgrade head`, then `alembic check` and the test suite (`test_migrations.py` fails if models and migrations differ).

**Run the tests.** See [Tests](#tests). `pytest -m "not ocr"` skips tests that need Tesseract.

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
- `tests/test_end_to_end.py` is the acceptance test: it uploads a generated multi-page PDF, processes it through the real Docling pipeline in the background, checks structure and passages, verifies every passage's page range against the original PDF text, drives the screens (library → structure → passages → next/previous) and re-reads everything through a new database engine, as after a restart. `test_pipeline.py` covers stages, failure at each stage, resume, idempotency and multiple editions; `test_integrity.py` breaks saved data on purpose; `test_pipeline_ocr.py` covers OCR (including real Tesseract); `test_ui_flows.py` covers the screens.
- Structure and passage tests build synthetic books in code (`tests/structure_helpers.py`): numbered and unnumbered chapters, parts, front/back matter, multi-level numbering, equations, tables, captions, multi-page paragraphs and chapters, ambiguous headings. The LLM is always faked. Tests marked `sample_book` also run the real PDF pipeline on the first 70 pages of the sample book in `storage/original` (git-ignored, so they are skipped when it is absent or Docling's models are missing).
- Phase 2 tests never call a model: `tests/extraction_db_helpers.py` supplies a scripted provider (`FunctionProvider`) that answers from a function, and every pipeline, recovery, provenance, equation, relationship and evaluation test runs on the test database with it. Day 5 added `test_evaluation_metrics.py` (the metrics as pure functions), `test_evaluation_runner.py` (the evaluation through the real pipeline with scripted models, repeatability, comparison, failures, the command line), `test_day5_integration.py` (a DOCX book processed by the real Phase 1 pipeline, then extracted, inspected, interrupted and resumed), `test_day5_integrity.py` (the delete rule of every Phase 2 foreign key), `test_day5_equation_provenance.py` and `test_day5_tools.py`. A test fixture refuses any use of the application database, so a test can never read or change your data by accident.
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
| `relation "structural_nodes" does not exist`, or `column ... retry_count / publisher does not exist` | A migration hasn't been applied. Run `alembic upgrade head`. |

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
| A document is *Failed during validation* | The data was saved but a consistency check failed. The Processing screen lists the findings (for example a passage linked to another section). Retry re-runs only validation; if it fails again, report the finding code with the log. |
| A document stays *Processing* | The app was restarted mid-run. After `PROCESSING_STALE_MINUTES` it can be retried; if its text was already saved, the retry resumes from the failed stage. |
| Structure tab: "must be processed before its structure can be detected" | Process the document first (Processing tab). Structure detection reads the stored text. |
| Structure tab: `ambiguous_heading`, `low_confidence_node` | Expected for headings with little typographic evidence (plain-text input, unusual layouts). The text is kept; check the node in the Warnings tab, or enable the optional LLM. |
| Structure tab: `toc_mismatch` | Few table-of-contents entries match detected headings: the structure may be incomplete or the contents list a different edition. |
| Structure tab: `PASSAGE_TOKENIZER=tiktoken needs…` | Install `tiktoken` or choose `approx`, `words` or `whitespace`. |
| Equations marked `uncertain` | Expected for model output with nothing to confirm it (scans, no reported confidence). Check them against the region image. |
| `unsupported_omml` | A Word equation uses a structure the converter can't handle. It is listed as invalid with its original OMML kept; the rest of the document is unaffected. |

## Project layout

```
app/
  main.py          Streamlit entry point
  config.py        Settings (environment variables)
  schemas.py       Pydantic models (book metadata, read models)
  ui/              Streamlit screens: library.py, upload.py, structure.py (explorer), passages.py (viewer),
                   processing.py, equations.py; navigation.py, selection.py, formatting.py
  services/        Business logic: book_service, document_service, storage_service,
                   file_validation, document_processing_service, pipeline_service (orchestrator),
                   integrity_service, library_service, job_runner, structure_service
  structure/       Day 4, pure computation (no database): numbering.py, toc.py, typography.py,
                   heading_detection.py, front_matter.py, hierarchy.py, paragraphs.py,
                   segmentation.py, tokenizer.py, validation.py, llm.py (optional),
                   detector.py (pipeline), tree.py (outline helpers), models.py
  parsing/         Parsers and OCR, independent of the UI and the database:
                   models.py (output models), base.py, parser_factory.py,
                   pdf_parser.py, epub_parser.py, docx_parser.py, ocr_service.py,
                   scan_detection.py, text_normalizer.py, omml.py (Word equations)
  parsing/equation/  Equation extraction: equation_service.py (orchestration), pdf_equations.py,
                   equation_extractor.py (EPUB/DOCX), equation_ocr.py (math OCR engines),
                   latex_normalizer.py, equation_validator.py, mathml_converter.py,
                   equation_detector.py, math_symbols.py, equation_models.py
  knowledge/       Phase 2 (no database access except records.py): types.py, content.py (typed content per
                   category), schemas.py (common envelope, relationships, runs), relationships.py, records.py
  extraction/      Phase 2 LLM layer (no database): providers, model configuration, versioned prompts, parsing, validation
  evaluation/      Phase 2 Day 5: dataset.py (reference results), metrics.py (pure), observation.py,
                   runner.py (loads a dataset, runs the pipeline, observes the result), report.py
  repositories/    Database queries
database/
  connection.py    Engine, sessions, connection check
  models/          SQLAlchemy models (Book, Document, ProcessingRun, ProcessingError, ExtractedBlock, Equation,
                   StructuralNode, Paragraph, Passage, PassageParagraph, StructureAnalysis,
                   KnowledgeUnit, KnowledgeUnitPassage, KnowledgeRelationship, ExtractionRun)
  migrations/      Alembic migrations (configured by alembic.ini)
  init/            Scripts run when the Docker database is first created
evaluation/        Reviewed reference datasets (datasets/) and generated reports (results/)
docs/phase2/       Evaluation report, model comparison, test and integrity results, performance, completion report
ingestion/         Reserved for later phases
storage/           Default location of uploaded files (git-ignored)
tests/             pytest suite
tools/             Quality measurement: evaluate_equations.py, compare_math_engines.py, extract_knowledge.py,
                   evaluate_extraction.py, benchmark_extraction.py, check_migrations_on_copy.py
```

The layers are kept separate: the UI calls services, services call parsers, repositories and the storage backend, and only repositories talk to the database. Parsers return structured results (`ParsingResult`) and never touch the database or the UI. Storage sits behind the `StorageBackend` interface in `storage_service.py`, so cloud storage can be added later.

## Known limitations

- A completed document cannot be re-processed from scratch yet (for example after changing OCR settings); only failed documents can be retried. Structure and passages can be rebuilt from its stored text.
- Processing runs in one background thread of the app process (no task queue): one document at a time, and restarting the app interrupts a run (it can be retried and resumed).
- No cover images or automatic publisher/year extraction: publisher, year and ISBN come from the upload form.
- Phase 2 extracts knowledge per passage. Knowledge graphs, chat/RAG retrieval, equation reconstruction, equation-to-Python generation, embeddings and semantic search are not implemented, and review is a simple status and flag, not a workflow. Known Phase 2 limitations (a unit can cite a neighbouring passage as its primary source, units are checked lexically rather than semantically, no live multi-model comparison has been run yet, the reference datasets await a human reviewer) are listed in [docs/phase2/completion-report.md](docs/phase2/completion-report.md#known-limitations-and-recommendations).
- The sample book used by the `sample_book` tests is not part of the repository; those tests are skipped when `storage/original` has no suitable PDF (the end-to-end test uses a generated book instead).
- Structure detection is heuristic. It works from the headings, sizes and positions Docling reports; books whose headings are not marked up (plain text, scans with poor layout analysis) produce more ambiguous nodes, and the optional LLM only classifies candidates the heuristics already found. A two-line title page that the parser merges into one block is split only when the author's name appears elsewhere in the opening pages. Headings the parser mislabels (formulas, pseudocode) can still appear as nodes, usually flagged ambiguous. Page headers repeated on three or more nearby pages are ignored as headings.
- Structure and passages are rebuilt for the whole document at once; they cannot be edited by hand yet. Overlap is taken from prose paragraphs only. Passage token counts are approximate unless `tiktoken` is used.
- The Structure tab shows page numbers but not page images yet; the stored page numbers and bounding boxes are what a later step needs to fetch them.
- Page headers and footers detected by Docling are left out of PDF text. Text inside images that are not full-page scans is not OCR'd.
- DOCX and EPUB have no page numbers; blocks are located by element and chapter references instead.
- One document is processed at a time, and the work runs inside the app process: restarting the app interrupts a run (it can then be retried).
- Equations: PDF inline formulas are not detected (only display equations); Docling's layout model misses some formula regions (4 of 20 in the synthetic test above); equation images in PDFs are recognized only inside regions the layout model finds; a model can still misread symbols (for example `x` as `X`), which is why unconfirmed model output is marked `uncertain`. Handwriting and degraded scans were not evaluated. Only the equation types listed in the table were tested; chemistry, units, and multi-line derivations are stored as ordinary equations without further structure.
- On scanned PDFs, an equation that layout analysis labels a picture (rather than a formula) is still read by Tesseract as garbled text in the page's text blocks; the equation itself is recovered separately in the Equations tab. The text blocks for such regions are not cleaned up.
- Docling can place a paragraph slightly out of order around formulas, so the "surrounding text" of an equation (taken from neighbouring blocks) can occasionally be off by a block.
- Re-extracting equations re-reads the document, so it takes as long as the first extraction.
