# Deployment

## Frontend on Vercel

- Set project root directory to `frontend`
- Build command: `npm install && npm run build`
- Output directory: `dist`
- Set `VITE_API_BASE_URL=https://your-backend.onrender.com`

## Backend on Render

- Set project root directory to `backend`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health check path is `/api/health`

## Required environment variables

### Backend

- `ALLOWED_ORIGINS`

### Frontend

- `VITE_API_BASE_URL`
