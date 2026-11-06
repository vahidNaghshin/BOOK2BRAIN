# Book2Brain

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d
```

## Run

```bash
streamlit run app/main.py
```
