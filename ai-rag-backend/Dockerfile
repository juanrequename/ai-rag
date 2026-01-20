FROM python:3.13.5-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1
WORKDIR /build

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (without dev dependencies)
RUN uv sync --frozen --no-install-project --no-dev

# Install uvicorn with standard extras for production
RUN uv pip install uvicorn[standard] --python .venv/bin/python

# Copy the rest of app
COPY app app
COPY alembic alembic
COPY alembic.ini .
COPY init.sh .

# Expose port
EXPOSE 8000

# Make the init script executable
RUN chmod +x ./init.sh

# Add venv to path
ENV PATH="/build/.venv/bin:$PATH"

# Set ENTRYPOINT to always run init.sh
ENTRYPOINT ["./init.sh"]

# Set CMD to uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--loop", "uvloop"]
