[![License](https://img.shields.io/github/license/rafsaf/minimal-fastapi-postgres-template)](https://github.com/rafsaf/minimal-fastapi-postgres-template/blob/main/LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue)](https://docs.python.org/3/whatsnew/3.13.html)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://github.com/rafsaf/minimal-fastapi-postgres-template/actions/workflows/tests.yml/badge.svg)](https://github.com/rafsaf/minimal-fastapi-postgres-template/actions/workflows/tests.yml)


# RAG: PDF CVs
- [Quickstart](#quickstart)

<br>


## Quickstart

### 1. Install dependecies with [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
cd your_project_name

### copy env vars (edit file)
cp .env.example .env 

### uv install
uv sync
```

### 2. Setup database and migrations

```bash
### Setup database
docker-compose up -d

### Run Alembic migrations
uv run alembic upgrade head
```



### 3. Run Scripts
```bash
### Generate pdfs (CVs) using LangChain with images based on LLM requests in the folder configured
uv run scripts/generate_pdfs.py


### The script looks for files in the folder configured and stores the parsed documents in Postgres and the configured PGVector store.
uv run scripts/ingest_pdfs.py
```



### 4. Run app

```bash
uv run uvicorn app.main:app --reload
```


### 5. Running tests

Note, it will create databases for session and run tests in many processes by default (using pytest-xdist) to speed up execution, based on how many CPU are available in environment.

For more details about initial database setup, see logic `app/tests/conftest.py` file, `fixture_setup_new_test_database` function.

Moreover, there is coverage pytest plugin with required code coverage level 100%.

```bash
# see all pytest configuration flags in pyproject.toml
uv run pytest
```

<br>

