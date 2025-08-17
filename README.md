# Nuerascan AI - Run Guide

- Start server:
  - If uvicorn available: `uvicorn server:app --host 0.0.0.0 --port 8000`
  - Or with Python: `python -m uvicorn server:app --host 0.0.0.0 --port 8000`

- Open browser at: http://localhost:8000/

- Frontend files served from `/static`:
  - `index.html` → `/`
  - `style.css`, `app.js`

- Known: system Python in this environment blocks `venv`; Mediapipe might be unavailable. The backend falls back to heuristics and OpenCV cascade if DeepFace/Mediapipe are missing.
