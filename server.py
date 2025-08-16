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

# Serve static files and outputs
base_dir = os.path.dirname(os.path.abspath(__file__))
static_root = base_dir
if os.path.isdir(static_root):
	app.mount("/static", StaticFiles(directory=static_root), name="static")
from backend_analysis import LOG_DIR as LOG_DIR_RESOLVED
from backend_analysis import SCREENSHOT_DIR as SCREENSHOT_DIR_RESOLVED

logs_dir = LOG_DIR_RESOLVED
screens_dir = SCREENSHOT_DIR_RESOLVED
if os.path.isdir(logs_dir):
	app.mount("/logs", StaticFiles(directory=logs_dir), name="logs")
if os.path.isdir(screens_dir):
	app.mount("/screenshots", StaticFiles(directory=screens_dir), name="screenshots")

UPLOAD_DIR = os.path.join(base_dir, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Session storage (single user focus)
sessions: Dict[str, Dict[str, Any]] = {}


def _is_under_base(path: str) -> bool:
	try:
		abs_p = os.path.abspath(path)
		abs_base = os.path.abspath(base_dir)
		abs_p_norm = os.path.normcase(os.path.realpath(abs_p))
		abs_base_norm = os.path.normcase(os.path.realpath(abs_base))
		return abs_p_norm == abs_base_norm or abs_p_norm.startswith(abs_base_norm + os.sep)
	except Exception:
		return False


def _sanitize_user(user: str) -> str:
	if not user:
		return "User"
	safe = ''.join(c for c in user if c.isalnum() or c in (' ', '-', '_')).rstrip()
	return safe.replace(' ', '_')


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


@app.get("/api/list_screenshots_by_session/{session_id}")
def api_list_screenshots_by_session(session_id: str, pattern: str = "eye_turn_*.jpg", limit: int = 24):
	s = sessions.get(session_id)
	if not s:
		raise HTTPException(status_code=404, detail="session not found")
	user = s.get('user', 'User')
	safe = _sanitize_user(user)
	folder = os.path.join(SCREENSHOT_DIR_RESOLVED, safe)
	if not os.path.isdir(folder):
		return { 'files': [] }
	names = []
	for name in os.listdir(folder):
		if fnmatch.fnmatch(name, pattern):
			names.append(name)
	names.sort(reverse=True)
	names = names[:max(1, min(100, limit))]
	return { 'files': [f"/api/file?path={os.path.join(folder, n)}" for n in names] }


@app.get("/api/session_paths/{session_id}")
def api_session_paths(session_id: str):
	s = sessions.get(session_id)
	if not s:
		raise HTTPException(status_code=404, detail="session not found")
	user = s.get('user', 'User')
	safe = _sanitize_user(user)
	screenshots_folder = os.path.join(base_dir, SCREENSHOT_FOLDER, safe)
	return { 'screenshots_folder': screenshots_folder }


@app.get("/")
def root():
	index_path = os.path.join(static_root, "index.html")
	if os.path.exists(index_path):
		return FileResponse(index_path)
	return JSONResponse({"message": "Frontend not found"})


# To run: uvicorn server:app --host 0.0.0.0 --port 8000