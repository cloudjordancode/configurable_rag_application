# Environment Management & Configuration

**Module:** 8 — RAG Intro & Docker Deployment
**Estimated time:** 25 minutes

## Objective

Refactor your RAG application to read all configuration from environment variables, using a centralized `Settings` class and proper `.env` file management.

## What You'll Build

- **`config.py`** — A `Settings` class that reads from environment variables with sensible defaults
- **`.env`** — Local overrides for development (not committed to git)
- **`.env.example`** — Documented example file safe to commit

**Settings to include (at minimum):**

| Variable               | Default                  | Purpose                 |
| ---------------------- | ------------------------ | ----------------------- |
| `OLLAMA_URL`           | `http://localhost:11434` | Ollama endpoint         |
| `MODEL_NAME`           | `llama3.2:1b`            | Model to use            |
| `CHROMA_PATH`          | `./rag_db`               | ChromaDB storage path   |
| `MAX_RESULTS`          | `3`                      | Default retrieval count |
| `CONFIDENCE_THRESHOLD` | `1.0`                    | Distance filter cutoff  |
| `DEBUG`                | `false`                  | Enable debug output     |

## Reference Code

The starter file (`config.py`) provides a scaffold with TODOs.

## Running

```bash
python config.py          # prints current settings
```

Update `docker-compose.yml` to use `env_file: .env`, then verify: change `MODEL_NAME` in `.env`, restart the stack, and confirm the change takes effect.

## Deliverable

A working `config.py`, a `.env` file, and a `.env.example`. The app should pick up config changes from `.env` without code changes.
