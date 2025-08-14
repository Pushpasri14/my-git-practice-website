from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from uuid import uuid4
import random

app = FastAPI(title="NeuraScan Backend")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["*"],
	allow_headers=["*"],
)

UPLOAD_DIR = Path("/workspace/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def seconds_to_mmss(total_seconds: int) -> str:
	minutes = total_seconds // 60
	seconds = total_seconds % 60
	return f"{minutes}:{seconds:02d}"


@app.get("/api/health")
async def health() -> dict:
	return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
	# Save uploaded file
	extension = Path(file.filename).suffix.lower()
	job_id = str(uuid4())
	destination_path = UPLOAD_DIR / f"{job_id}{extension}"
	with destination_path.open("wb") as out_file:
		while True:
			chunk = await file.read(1024 * 1024)
			if not chunk:
				break
			out_file.write(chunk)

	# Simulated analysis (replace with your real analyzer when ready)
	rng = random.Random(job_id)
	duration_seconds = rng.randint(75, 320)
	emotions = ["happy", "sad", "angry", "neutral", "surprised"]
	emotion_scores = [{"name": e, "percent": rng.randint(5, 45)} for e in emotions]
	dominant_emotion = max(emotion_scores, key=lambda x: x["percent"])['name']

	motions = []
	for _ in range(4):
		timestamp_seconds = rng.randint(5, max(6, duration_seconds - 5))
		motions.append({
			"timestamp": seconds_to_mmss(timestamp_seconds),
			"direction": rng.choice(["left", "right"]),
			"emotion": rng.choice(emotions),
			"duration": round(rng.uniform(2.0, 4.0), 1),
			"accuracy": round(rng.uniform(90.0, 99.5), 1),
		})

	gestures = []
	gesture_types = ["wave", "thumbs up", "peace sign", "point", "clap"]
	for _ in range(4):
		timestamp_seconds = rng.randint(5, max(6, duration_seconds - 5))
		gestures.append({
			"timestamp": seconds_to_mmss(timestamp_seconds),
			"gesture": rng.choice(gesture_types),
			"confidence": round(rng.uniform(85.0, 98.0), 1),
			"duration": round(rng.uniform(1.0, 3.0), 1),
			"activity": rng.choice(["Active", "Moderate"]),
		})

	moments = []
	for _ in range(5):
		timestamp_seconds = rng.randint(5, max(6, duration_seconds - 5))
		moments.append({
			"timestamp": seconds_to_mmss(timestamp_seconds),
			"event": rng.choice(["Significant facial expression change", "Major head/body movement"]),
		})

	response = {
		"success": True,
		"jobId": job_id,
		"fileName": file.filename,
		"dominantEmotion": dominant_emotion,
		"stats": {
			"duration": seconds_to_mmss(duration_seconds),
			"motionEvents": len(motions),
			"gestureCount": len(gestures),
		},
		"emotions": emotion_scores,
		"motions": motions,
		"gestures": gestures,
		"moments": moments,
	}
	return response