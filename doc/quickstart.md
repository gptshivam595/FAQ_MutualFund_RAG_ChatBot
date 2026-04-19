# Quickstart

## Backend

1. `cd backend`
2. Create and activate a virtual environment
3. `pip install -r requirements.txt`
4. Optionally set `ALLOWED_ORIGINS`
5. Run `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

## Frontend

1. `cd frontend`
2. `npm install`
3. Create `.env` with `VITE_API_BASE_URL=http://localhost:8000`
4. Run `npm run dev`

## Endpoints

- `GET /api/health`
- `POST /api/ask`
