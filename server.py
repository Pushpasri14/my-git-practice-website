import os
import uuid
import threading
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import fnmatch

from backend_analysis import run_analysis, shared_data, data_lock, LOG_FOLDER, SCREENSHOT_FOLDER

app = FastAPI(title="Video Multimodal Analysis API")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Serve workspace static files and outputs
static_root = "/workspace"
if os.path.isdir(static_root):
	app.mount("/static", StaticFiles(directory=static_root), name="static")
if os.path.isdir(os.path.join(static_root, LOG_FOLDER)):
	app.mount("/logs", StaticFiles(directory=os.path.join(static_root, LOG_FOLDER)), name="logs")
if os.path.isdir(os.path.join(static_root, SCREENSHOT_FOLDER)):
	app.mount("/screenshots", StaticFiles(directory=os.path.join(static_root, SCREENSHOT_FOLDER)), name="screenshots")

UPLOAD_DIR = "/workspace/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Session storage (single user focus)
sessions: Dict[str, Dict[str, Any]] = {}


def analysis_thread(session_id: str, user_name: str, video_path: str):
	def on_update(state: Dict[str, Any]):
		with threading.Lock():
			sessions[session_id]['state'] = state
	try:
		outputs = run_analysis(user_name=user_name, video_path=video_path, on_update=on_update)
		sessions[session_id]['outputs'] = outputs
		sessions[session_id]['status'] = 'completed'
	except Exception as e:
		sessions[session_id]['status'] = 'error'
		sessions[session_id]['error'] = str(e)


@app.post("/api/analyze")
def api_analyze(file: UploadFile = File(...), user: str = Form("User")):
	if file.content_type and not file.content_type.startswith('video'):
		raise HTTPException(status_code=400, detail="Please upload a video file")
	session_id = str(uuid.uuid4())
	filename = f"{session_id}_{file.filename}"
	save_path = os.path.join(UPLOAD_DIR, filename)
	with open(save_path, 'wb') as f:
		f.write(file.file.read())
	sessions[session_id] = {
		'status': 'running',
		'user': user,
		'video': save_path,
		'state': {},
		'outputs': {},
	}
	thr = threading.Thread(target=analysis_thread, args=(session_id, user, save_path), daemon=True)
	thr.start()
	return { 'session_id': session_id }


@app.get("/api/status/{session_id}")
def api_status(session_id: str):
	s = sessions.get(session_id)
	if not s:
		raise HTTPException(status_code=404, detail="session not found")
	resp = { 'status': s['status'], 'user': s['user'], 'video': s['video'], 'state': s.get('state', {}), 'outputs': s.get('outputs', {}) }
	return JSONResponse(resp)


@app.get("/api/last_state")
def api_last_state():
	with data_lock:
		state = {k: v for k, v in shared_data.items() if k not in ['hand_landmarks']}
	return state


@app.get("/api/file")
def api_file(path: str, download: bool = False):
	# Serve a file only if it resides under the project directory
	if not _is_under_base(path):
		raise HTTPException(status_code=400, detail="invalid path")
	if not os.path.exists(path):
		raise HTTPException(status_code=404, detail="file not found")
	filename = os.path.basename(path)
	headers = {"Content-Disposition": f"attachment; filename={filename}"} if download else None
	return FileResponse(path, headers=headers)


@app.get("/api/list_screenshots")
def api_list_screenshots(path: str, pattern: str = "eye_turn_*.jpg", limit: int = 24):
	if not path.startswith("/workspace/"):
		raise HTTPException(status_code=400, detail="invalid path")
	if not os.path.isdir(path):
		raise HTTPException(status_code=404, detail="folder not found")
	entries = []
	for name in os.listdir(path):
		if fnmatch.fnmatch(name, pattern):
			entries.append({
				'name': name,
				'path': os.path.join(path, name)
			})
	entries.sort(key=lambda x: x['name'], reverse=True)
	entries = entries[:max(1, min(100, limit))]
	# return URLs via the /api/file route
	return {
		'files': [f"/api/file?path={os.path.join(path, e['name'])}" for e in entries]
	}


@app.get("/")
def root():
	index_path = os.path.join(static_root, "index.html")
	if os.path.exists(index_path):
		return FileResponse(index_path)
	return JSONResponse({"message": "Frontend not found"})


# To run: uvicorn server:app --host 0.0.0.0 --port 8000