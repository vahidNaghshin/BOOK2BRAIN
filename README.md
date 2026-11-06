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
