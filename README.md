# Book2Brain

Turn books into a searchable knowledge base. Documents are uploaded, extracted and stored in PostgreSQL with pgvector.

## Prerequisites

- Git
- Python 3.12
- Docker with Docker Compose
- Tesseract OCR (macOS: `brew install tesseract`)

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
| `DATABASE_URL` | SQLAlchemy connection URL used by the app |

If you change `POSTGRES_PORT`, change the port in `DATABASE_URL` too. Special characters in the password must be URL-encoded in `DATABASE_URL`.

### 4. Start the database

```bash
docker compose up -d
docker compose ps
```

Wait until the `db` service shows `healthy`. The pgvector extension is enabled automatically the first time the database starts. Data is kept in the `pgdata` Docker volume.

### 5. Launch the application

```bash
streamlit run app/main.py
```

Open http://localhost:8501. The page shows the title, version, database connection status, and the upload and processing status placeholders.

## Common tasks

| Task | Command |
|---|---|
| Stop the database | `docker compose stop` |
| Delete the database and its data | `docker compose down -v` |
| Run tests | `pytest` |

If you change the database password in `.env` after the first start, run `docker compose down -v` and start the database again, because the password is only applied when the volume is created.

## Project layout

```
app/         Streamlit application
ingestion/   Document upload and ingestion
database/    Connection, models and init scripts
models/      Domain models
storage/     Uploaded and extracted files (git-ignored)
tests/       Tests
config/      Configuration
```
