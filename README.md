# task-api

CRUD-сервис для управления задачами. FastAPI + Docker + автодеплой.

## Локальный запуск

```bash
conda create -y -n task-api python=3.11
conda activate task-api
pip install -r requirements.txt
uvicorn app.main:app --reload