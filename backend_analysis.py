import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import threading
import time
from datetime import datetime
from typing import Callable, Optional, Dict, Any

try:
	from deepface import DeepFace
	_deepface_available = True
except Exception:
	DeepFace = None
	_deepface_available = False
try:
	import mediapipe as mp
	_mediapipe_available = True
except Exception:
	mp = None
	_mediapipe_available = False

# --- Global variables for communication between threads ---
shared_data = {
	'dominant_emotion': "No Face",
	'emotions': {},
	'face_region': None,
	'audio_volume': "No audio detected",
	'hand_status': "Inactive",
	'hand_landmarks': [],
	'eye_turn_count': 0,
	'left_eye_turns': 0,
	'right_eye_turns': 0,
	'current_gaze_direction': 'center',
	'previous_gaze_direction': 'center',
	'gaze_start_time': 0,
	'gaze_hold_duration': 2.0,
	'gaze_counted': False,
	'gaze_history': [],
	'left_iris_x': 0.5,
	'right_iris_x': 0.5,
	'combined_ratio': 0.5,
	'screenshot_taken': False,
	'video_timestamp': 0.0,
	'is_running': True,
	'emotion_history': [],
	'emotion_smoothing_window': 15,
	'emotion_confidence_threshold': 0.3,
	'stable_emotion': "No Face",
	'emotion_start_time': {},
	'emotion_duration_threshold': 1.0,
	'confirmed_emotions': {},
	'ema_emotions': {},
	'ema_alpha': 0.35,
	'mouth_width_ratio': 0.0,
	'mouth_height_ratio': 0.0,
	'smile_confidence': 0.0,
	# aggregate counts for summary
	'emotion_counts': {}
}

data_lock = threading.Lock()

# MediaPipe constants
LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

# Storage configuration
LOG_INTERVAL = 3
LOG_FOLDER = "multimodal_logs"
SCREENSHOT_FOLDER = "eye_turn_screenshots"
LOG_FILE = None
current_frame = None

VIDEO_WRITER = None
VIDEO_CODEC = 'mp4v'
ANALYZED_VIDEO_PATH = None
USER_SCREENSHOT_FOLDER = None
SESSION_TIMESTAMP = None

# Anchor directories to project base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, LOG_FOLDER)
SCREENSHOT_DIR = os.path.join(BASE_DIR, SCREENSHOT_FOLDER)

# Fallback face detector (OpenCV Haar cascade)
_CASCADE = None

def _load_cascade():
	global _CASCADE
	if _CASCADE is None:
		try:
			cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
			_CASCADE = cv2.CascadeClassifier(cascade_path)
		except Exception:
			_CASCADE = None
	return _CASCADE

# Utility

def clamp(x, lo, hi):
	return max(lo, min(hi, x))

def create_directories():
	os.makedirs(LOG_DIR, exist_ok=True)
	os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def format_video_time(seconds):
	minutes = int(seconds // 60)
	secs = int(seconds % 60)
	return f"{minutes:02d}:{secs:02d}"

def format_detailed_video_time(seconds):
	minutes = int(seconds // 60)
	secs = seconds % 60
	return f"{minutes:02d}:{secs:06.3f}"


def preprocess_face_roi(roi_bgr):
	try:
		if roi_bgr is None or roi_bgr.size == 0:
			return roi_bgr
		ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
		y, cr, cb = cv2.split(ycrcb)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		y_eq = clahe.apply(y)
		ycrcb_eq = cv2.merge((y_eq, cr, cb))
		bgr_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
		return bgr_eq
	except Exception:
		return roi_bgr


def smooth_emotions_ema(current_emotions, ema_state, alpha=0.35):
	if not current_emotions:
		return ema_state, None
	if not ema_state:
		ema_state = {k: float(v) for k, v in current_emotions.items()}
	else:
		for k, v in current_emotions.items():
			prev = ema_state.get(k, float(v))
			ema_state[k] = (1 - alpha) * prev + alpha * float(v)
		for k in list(ema_state.keys()):
			if k not in current_emotions:
				ema_state[k] = (1 - alpha) * ema_state[k]
	return ema_state, max(ema_state, key=ema_state.get)


def track_emotion_duration(emotion, current_time, emotion_start_times, duration_threshold=1.0):
	try:
		if emotion == "No Face" or emotion == "neutral":
			return False
		if emotion not in emotion_start_times:
			emotion_start_times[emotion] = current_time
			return False
		duration = current_time - emotion_start_times[emotion]
		return duration >= duration_threshold
	except Exception:
		return False


def reset_emotion_tracking(emotion_start_times, except_emotion=None):
	emotions_to_reset = [e for e in emotion_start_times.keys() if e != except_emotion]
	for emotion in emotions_to_reset:
		del emotion_start_times[emotion]


def setup_logging(user_name):
	global LOG_FILE, USER_SCREENSHOT_FOLDER, SESSION_TIMESTAMP, ANALYZED_VIDEO_PATH
	create_directories()
	safe_name = "".join(c for c in user_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
	user_screenshot_folder = os.path.join(SCREENSHOT_DIR, safe_name)
	os.makedirs(user_screenshot_folder, exist_ok=True)
	USER_SCREENSHOT_FOLDER = user_screenshot_folder
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	SESSION_TIMESTAMP = timestamp
	log_filename = f"{safe_name}_video_analysis_{timestamp}.txt"
	LOG_FILE = os.path.join(LOG_DIR, log_filename)
	ANALYZED_VIDEO_PATH = os.path.join(user_screenshot_folder, f"{safe_name}_ANALYZED_{timestamp}.mp4")
	with open(LOG_FILE, 'w', encoding='utf-8') as f:
		f.write("="*80 + "\n")
		f.write("VIDEO MULTIMODAL ANALYSIS LOG WITH VIDEO TIMESTAMPS\n")
		f.write(f"User: {user_name}\n")
		f.write(f"Session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
		f.write(f"Log location: {os.path.abspath(LOG_FILE)}\n")
		f.write(f"Screenshots folder: {os.path.abspath(user_screenshot_folder)}\n")
		f.write(f"Analyzed video (overlay) will be saved: {os.path.abspath(ANALYZED_VIDEO_PATH)}\n")
		f.write("AI Summary will be generated at session end\n")
		f.write("="*80 + "\n\n")
	return safe_name


def save_screenshot(frame, user_name, gaze_direction, video_timestamp):
	try:
		system_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
		video_time_str = format_video_time(video_timestamp)
		video_time_detailed = format_detailed_video_time(video_timestamp)
		safe_name = "".join(c for c in user_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
		user_screenshot_folder = os.path.join(SCREENSHOT_DIR, safe_name)
		screenshot_filename = f"eye_turn_{gaze_direction}_video{video_time_str.replace(':', 'm')}_sys{system_timestamp}.jpg"
		screenshot_path = os.path.join(user_screenshot_folder, screenshot_filename)
		frame_with_overlay = frame.copy()
		font = cv2.FONT_HERSHEY_SIMPLEX
		video_text = f"Video Time: {video_time_str}"
		text_size = cv2.getTextSize(video_text, font, 0.8, 2)[0]
		cv2.rectangle(frame_with_overlay, (frame.shape[1] - text_size[0] - 20, 10), (frame.shape[1] - 5, 50), (0, 0, 0), -1)
		cv2.putText(frame_with_overlay, video_text, (frame.shape[1] - text_size[0] - 10, 35), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
		gaze_text = f"Gaze: {gaze_direction.upper()}"
		gaze_size = cv2.getTextSize(gaze_text, font, 0.8, 2)[0]
		cv2.rectangle(frame_with_overlay, (frame.shape[1] - gaze_size[0] - 20, frame.shape[0] - 50), (frame.shape[1] - 5, frame.shape[0] - 10), (0, 0, 0), -1)
		cv2.putText(frame_with_overlay, gaze_text, (frame.shape[1] - gaze_size[0] - 10, frame.shape[0] - 25), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
		cv2.imwrite(screenshot_path, frame_with_overlay)
		if LOG_FILE:
			with open(LOG_FILE, 'a', encoding='utf-8') as f:
				f.write(f"📸 SCREENSHOT CAPTURED\n")
				f.write(f"   System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
				f.write(f"   Video Time: {video_time_detailed} ({video_time_str})\n")
				f.write(f"   Eye Turn Direction: {gaze_direction.upper()}\n")
				f.write(f"   Screenshot saved: {screenshot_filename}\n")
				f.write("-" * 40 + "\n\n")
		return screenshot_path
	except Exception:
		return None


def log_summary(user_name):
	global LOG_FILE
	if LOG_FILE is None:
		return
	try:
		with data_lock:
			timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
			video_time_str = format_video_time(shared_data['video_timestamp'])
			summary = {
				'timestamp': timestamp,
				'video_timestamp': shared_data['video_timestamp'],
				'video_time_str': video_time_str,
				'user_name': user_name,
				'dominant_emotion': shared_data['dominant_emotion'],
				'emotions': shared_data['emotions'].copy() if shared_data['emotions'] else {},
				'audio_volume': shared_data['audio_volume'],
				'hand_status': shared_data['hand_status'],
				'eye_turn_count': shared_data['eye_turn_count'],
				'current_gaze_direction': shared_data['current_gaze_direction'],
				'left_iris_ratio': shared_data['left_iris_x'],
				'right_iris_ratio': shared_data['right_iris_x'],
				'combined_ratio': shared_data['combined_ratio'],
				'smile_confidence': shared_data.get('smile_confidence', 0.0)
			}
		with open(LOG_FILE, 'a', encoding='utf-8') as f:
			f.write(f"[{summary['timestamp']}] VIDEO ANALYSIS SUMMARY - {summary['user_name']}\n")
			f.write(f"Video Time: {summary['video_time_str']} ({summary['video_timestamp']:.3f}s)\n")
			f.write("-" * 60 + "\n")
			f.write(f"Primary Emotion: {summary['dominant_emotion']}\n")
			if summary['emotions']:
				f.write("Emotion Breakdown:\n")
				for emotion, score in summary['emotions'].items():
					f.write(f"  - {emotion.capitalize()}: {score:.1f}%\n")
			else:
				f.write("Emotion Breakdown: No face detected\n")
			f.write(f"Audio Status: Video playback\n")
			f.write(f"Hand Activity: {summary['hand_status']}\n")
			f.write(f"Eye Turn Count: {summary['eye_turn_count']}\n")
			f.write(f"Current Gaze Direction: {summary['current_gaze_direction'].upper()}\n")
			f.write(f"Gaze Ratios - Left: {summary['left_iris_ratio']:.3f}, Right: {summary['right_iris_ratio']:.3f}, Combined: {summary['combined_ratio']:.3f}\n")
			f.write(f"Smile Confidence: {summary['smile_confidence']:.2f}\n")
			f.write("-" * 60 + "\n\n")
	except Exception:
		pass


def logging_thread(user_name):
	time.sleep(LOG_INTERVAL)
	while shared_data['is_running']:
		try:
			log_summary(user_name)
			time.sleep(LOG_INTERVAL)
		except Exception:
			time.sleep(LOG_INTERVAL)


def audio_listener():
	with data_lock:
		shared_data['audio_volume'] = "Video playback (no live audio)"
	while shared_data['is_running']:
		time.sleep(1)


def get_iris_center(landmarks, iris_indices, frame_width, frame_height):
	try:
		iris_points = []
		for idx in iris_indices:
			if idx < len(landmarks.landmark):
				x = landmarks.landmark[idx].x * frame_width
				y = landmarks.landmark[idx].y * frame_height
				iris_points.append([x, y])
		if len(iris_points) < 4:
			return None
		iris_points = np.array(iris_points)
		center = np.mean(iris_points, axis=0)
		return center
	except Exception:
		return None


def get_eye_center(landmarks, eye_indices, frame_width, frame_height):
	try:
		eye_points = []
		for idx in eye_indices:
			if idx < len(landmarks.landmark):
				x = landmarks.landmark[idx].x * frame_width
				y = landmarks.landmark[idx].y * frame_height
				eye_points.append([x, y])
		if len(eye_points) < 4:
			return None
		eye_points = np.array(eye_points)
		center = np.mean(eye_points, axis=0)
		return center
	except Exception:
		return None


def calculate_gaze_ratio(iris_center, eye_center):
	try:
		if iris_center is None or eye_center is None:
			return 0.5
		iris_x = iris_center[0]
		eye_x = eye_center[0]
		eye_width = 25
		displacement = (iris_x - eye_x) / eye_width
		ratio = 0.5 + displacement
		ratio = max(0.0, min(1.0, ratio))
		return ratio
	except Exception:
		return 0.5


def compute_smile_metrics(face_landmarks, frame_width, frame_height, left_eye_center=None, right_eye_center=None):
	try:
		if face_landmarks is None:
			return 0.0, 0.0, 0.0
		def lm(idx):
			pt = face_landmarks.landmark[idx]
			return np.array([pt.x * frame_width, pt.y * frame_height], dtype=np.float32)
		left_corner = lm(MOUTH_LEFT)
		right_corner = lm(MOUTH_RIGHT)
		top_lip = lm(MOUTH_TOP)
		bottom_lip = lm(MOUTH_BOTTOM)
		mouth_width = float(np.linalg.norm(right_corner - left_corner))
		mouth_height = float(np.linalg.norm(bottom_lip - top_lip))
		if left_eye_center is not None and right_eye_center is not None:
			eye_dist = float(np.linalg.norm(np.array(right_eye_center) - np.array(left_eye_center)))
		else:
			eye_dist = max(1.0, mouth_width)
		width_ratio = clamp(mouth_width / max(eye_dist, 1e-3), 0.0, 3.0)
		height_ratio = clamp(mouth_height / max(eye_dist, 1e-3), 0.0, 2.0)
		width_component = clamp((width_ratio - 1.05) / 0.40, 0.0, 1.0)
		height_component = clamp((height_ratio - 0.22) / 0.35, 0.0, 1.0)
		smile_confidence = 0.6 * width_component + 0.4 * height_component
		return width_ratio, height_ratio, smile_confidence
	except Exception:
		return 0.0, 0.0, 0.0


def get_face_roi_from_detection(frame, detection):
	try:
		h, w = frame.shape[:2]
		rel_box = detection.location_data.relative_bounding_box
		x = rel_box.xmin
		y = rel_box.ymin
		bw = rel_box.width
		bh = rel_box.height
		margin = 0.18
		x0 = int(clamp((x - margin) * w, 0, w - 1))
		y0 = int(clamp((y - margin) * h, 0, h - 1))
		x1 = int(clamp((x + bw + margin) * w, 1, w))
		y1 = int(clamp((y + bh + margin) * h, 1, h))
		if x1 <= x0 or y1 <= y0:
			return None, None
		roi = frame[y0:y1, x0:x1]
		return roi, {'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0}
	except Exception:
		return None, None


def get_face_roi_from_rect(frame, rect, margin_ratio: float = 0.18):
	try:
		h, w = frame.shape[:2]
		x, y, rw, rh = rect
		mx = int(rw * margin_ratio)
		my = int(rh * margin_ratio)
		x0 = clamp(x - mx, 0, w - 1)
		y0 = clamp(y - my, 0, h - 1)
		x1 = clamp(x + rw + mx, 1, w)
		y1 = clamp(y + rh + my, 1, h)
		x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
		if x1 <= x0 or y1 <= y0:
			return None, None
		roi = frame[y0:y1, x0:x1]
		return roi, {'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0}
	except Exception:
		return None, None


def detect_face_cascade(frame):
	try:
		casc = _load_cascade()
		if casc is None:
			return None
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = casc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
		if faces is None or len(faces) == 0:
			return None
		# pick largest face
		faces = sorted(list(faces), key=lambda r: r[2] * r[3], reverse=True)
		return faces[0]
	except Exception:
		return None


def video_process(user_name: str, video_path: str, on_update: Optional[Callable[[Dict[str, Any]], None]] = None, headless: bool = True):
	global current_frame, VIDEO_WRITER, ANALYZED_VIDEO_PATH
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		return
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
	try:
		fps = float(fps)
	except Exception:
		fps = 25.0
	writer_initialized = False

	if _mediapipe_available:
		mp_hands = mp.solutions.hands
		hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
		mp_drawing = mp.solutions.drawing_utils
		mp_face_mesh = mp.solutions.face_mesh
		face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
		mp_face_detection = mp.solutions.face_detection
		face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.6, model_selection=0)
	else:
		hands = None
		mp_drawing = None
		face_mesh = None
		face_detector = None

	frame_count = 0
	last_update_ts = time.time()
	# periodic screenshot fallback
	with data_lock:
		if 'last_screenshot_time' not in shared_data:
			shared_data['last_screenshot_time'] = 0.0
		if 'screenshot_interval' not in shared_data:
			shared_data['screenshot_interval'] = 5.0
	try:
		while shared_data['is_running']:
			ret, frame = cap.read()
			if not ret:
				break
			current_frame = frame.copy()
			frame_count += 1
			frame_height, frame_width = frame.shape[:2]
			if not writer_initialized and ANALYZED_VIDEO_PATH is not None:
				try:
					fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
					VIDEO_WRITER = cv2.VideoWriter(ANALYZED_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
					writer_initialized = True
				except Exception:
					writer_initialized = False
			current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
			current_time_sec = current_frame_num / fps if fps > 0 else 0
			with data_lock:
				shared_data['video_timestamp'] = current_time_sec
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			# Hands
			if hands is not None:
				hand_results = hands.process(frame_rgb)
			else:
				hand_results = type('obj', (), {'multi_hand_landmarks': None})()
			with data_lock:
				if getattr(hand_results, 'multi_hand_landmarks', None):
					shared_data['hand_status'] = "Active"
					shared_data['hand_landmarks'] = hand_results.multi_hand_landmarks
				else:
					shared_data['hand_status'] = "Inactive"
					shared_data['hand_landmarks'] = []
			# Face mesh
			if face_mesh is not None:
				face_results = face_mesh.process(frame_rgb)
			else:
				face_results = type('obj', (), {'multi_face_landmarks': None})()
			left_eye_center = None
			right_eye_center = None
			with data_lock:
				if getattr(face_results, 'multi_face_landmarks', None):
					face_landmarks = face_results.multi_face_landmarks[0]
					try:
						left_iris_center = get_iris_center(face_landmarks, LEFT_IRIS, frame_width, frame_height)
						right_iris_center = get_iris_center(face_landmarks, RIGHT_IRIS, frame_width, frame_height)
						left_eye_center = get_eye_center(face_landmarks, LEFT_EYE, frame_width, frame_height)
						right_eye_center = get_eye_center(face_landmarks, RIGHT_EYE, frame_width, frame_height)
						left_gaze_ratio = calculate_gaze_ratio(left_iris_center, left_eye_center)
						right_gaze_ratio = calculate_gaze_ratio(right_iris_center, right_eye_center)
						shared_data['left_iris_x'] = left_gaze_ratio
						shared_data['right_iris_x'] = right_gaze_ratio
						combined_gaze = (left_gaze_ratio + right_gaze_ratio) / 2
						shared_data['combined_ratio'] = combined_gaze
						gaze_history = shared_data.get('gaze_history', [])
						gaze_history.append(combined_gaze)
						if len(gaze_history) > 8:
							gaze_history = gaze_history[-8:]
						shared_data['gaze_history'] = gaze_history
						smoothed_gaze = np.mean(gaze_history)
						new_gaze_direction = "center"
						if smoothed_gaze < 0.40:
							new_gaze_direction = "left"
						elif smoothed_gaze > 0.65:
							new_gaze_direction = "right"
						current_time = time.time()
						if new_gaze_direction != shared_data['current_gaze_direction']:
							shared_data['previous_gaze_direction'] = shared_data['current_gaze_direction']
							shared_data['current_gaze_direction'] = new_gaze_direction
							shared_data['gaze_start_time'] = current_time
							shared_data['gaze_counted'] = False
							shared_data['screenshot_taken'] = False
						if new_gaze_direction in ["left", "right"] and not shared_data['gaze_counted']:
							time_held = current_time - shared_data['gaze_start_time']
							if time_held >= shared_data['gaze_hold_duration']:
								shared_data['eye_turn_count'] += 1
								if new_gaze_direction == 'left':
									shared_data['left_eye_turns'] = shared_data.get('left_eye_turns', 0) + 1
								elif new_gaze_direction == 'right':
									shared_data['right_eye_turns'] = shared_data.get('right_eye_turns', 0) + 1
								shared_data['gaze_counted'] = True
								if not shared_data['screenshot_taken'] and current_frame is not None:
									save_screenshot(current_frame, user_name, new_gaze_direction, shared_data['video_timestamp'])
									shared_data['screenshot_taken'] = True
						mw_ratio, mh_ratio, smile_conf = compute_smile_metrics(face_landmarks, frame_width, frame_height, left_eye_center, right_eye_center)
						shared_data['mouth_width_ratio'] = mw_ratio
						shared_data['mouth_height_ratio'] = mh_ratio
						shared_data['smile_confidence'] = smile_conf
					except Exception:
						shared_data['current_gaze_direction'] = "center"
				else:
					shared_data['current_gaze_direction'] = "center"
			if not _mediapipe_available:
				with data_lock:
					shared_data['smile_confidence'] = 0.0
					shared_data['left_iris_x'] = 0.5
					shared_data['right_iris_x'] = 0.5
					shared_data['combined_ratio'] = 0.5
					shared_data['current_gaze_direction'] = "center"
			# Face detection and emotion
			emotion_updated = False
			try:
				det_results = None
				if face_detector is not None:
					det_results = face_detector.process(frame_rgb)
				if det_results and det_results.detections:
					det = max(det_results.detections, key=lambda d: d.score[0] if d.score else 0.0)
					roi, region = get_face_roi_from_detection(frame, det)
					if roi is not None:
						roi_pre = preprocess_face_roi(roi)
						result = None
						try:
							if _deepface_available:
								result = DeepFace.analyze(img_path=roi_pre, actions=['emotion'], enforce_detection=False, detector_backend='skip', align=True)
							else:
								result = None
						except Exception:
							if _deepface_available:
								result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False, detector_backend='mediapipe', align=True)
							else:
								result = None
						if isinstance(result, list) and len(result) > 0:
							data = result[0]
						elif isinstance(result, dict):
							data = result
						else:
							data = None
						current_time = time.time()
						with data_lock:
							if data and 'emotion' in data:
								raw_emotions = {k: float(v) for k, v in data['emotion'].items()}
								smile_c = shared_data.get('smile_confidence', 0.0)
								if smile_c >= 0.55:
									boost = 25.0 * (smile_c - 0.55) / 0.45
									raw_emotions['happy'] = max(raw_emotions.get('happy', 0.0), min(90.0, raw_emotions.get('happy', 0.0) + 20.0 + boost))
									for neg in ['angry', 'disgust', 'fear']:
										if neg in raw_emotions:
											raw_emotions[neg] = raw_emotions[neg] * (0.65 if smile_c > 0.7 else 0.8)
								ema_state, max_key = smooth_emotions_ema(raw_emotions, shared_data.get('ema_emotions', {}), alpha=shared_data.get('ema_alpha', 0.35))
								shared_data['ema_emotions'] = ema_state
								if ema_state:
									max_emotion = max(ema_state, key=ema_state.get)
									second = sorted(ema_state.values(), reverse=True)[1] if len(ema_state) > 1 else 0.0
									confidence_gap = ema_state[max_emotion] - second
									is_stable = track_emotion_duration(max_emotion, current_time, shared_data['emotion_start_time'], shared_data['emotion_duration_threshold'])
									reset_emotion_tracking(shared_data['emotion_start_time'], max_emotion)
									shared_data['emotions'] = ema_state
									shared_data['face_region'] = region
									if is_stable and confidence_gap >= 6.0:
										shared_data['stable_emotion'] = max_emotion
										shared_data['dominant_emotion'] = max_emotion
										shared_data['confirmed_emotions'][max_emotion] = current_time
									else:
										shared_data['dominant_emotion'] = f"{max_emotion} (detecting...)"
								emotion_updated = True
							else:
								smile_c = shared_data.get('smile_confidence', 0.0)
								heuristic = {
									'happy': max(0.0, min(100.0, 20.0 + 70.0 * smile_c)),
									'neutral': max(0.0, 60.0 - 30.0 * smile_c),
									'surprise': 10.0 * smile_c,
									'angry': 5.0 * (1.0 - smile_c),
									'sad': 5.0 * (1.0 - smile_c),
									'fear': 5.0 * (1.0 - smile_c),
									'disgust': 3.0 * (1.0 - smile_c)
								}
								ema_state, _ = smooth_emotions_ema(heuristic, shared_data.get('ema_emotions', {}), alpha=shared_data.get('ema_alpha', 0.35))
								shared_data['ema_emotions'] = ema_state
								if ema_state:
									max_emotion = max(ema_state, key=ema_state.get)
									shared_data['emotions'] = ema_state
									shared_data['face_region'] = region
									shared_data['dominant_emotion'] = f"{max_emotion} (heuristic)"
								else:
									shared_data['dominant_emotion'] = "No Face"
									shared_data['stable_emotion'] = "No Face"
									shared_data['emotions'] = {}
									shared_data['face_region'] = None
				else:
					# OpenCV Haar fallback
					rect = detect_face_cascade(frame)
					if rect is not None:
						roi, region = get_face_roi_from_rect(frame, rect)
						if roi is not None:
							roi_pre = preprocess_face_roi(roi)
							result = None
							try:
								if _deepface_available:
									result = DeepFace.analyze(img_path=roi_pre, actions=['emotion'], enforce_detection=False, detector_backend='skip', align=True)
								else:
									result = None
							except Exception:
								result = None
							if isinstance(result, list) and len(result) > 0:
								data = result[0]
							elif isinstance(result, dict):
								data = result
							else:
								data = None
							with data_lock:
								if data and 'emotion' in data:
									raw_emotions = {k: float(v) for k, v in data['emotion'].items()}
									shared_data['ema_emotions'], _ = smooth_emotions_ema(raw_emotions, shared_data.get('ema_emotions', {}), alpha=shared_data.get('ema_alpha', 0.35))
									max_emotion = max(shared_data['ema_emotions'], key=shared_data['ema_emotions'].get)
									shared_data['emotions'] = shared_data['ema_emotions']
									shared_data['dominant_emotion'] = f"{max_emotion}"
									shared_data['face_region'] = region
									emotion_updated = True
								else:
									# Heuristic emotions when DeepFace not available
									smile_c = shared_data.get('smile_confidence', 0.0)
									heuristic = {
										'neutral': max(0.0, 80.0 - 30.0 * smile_c),
										'happy': max(0.0, 15.0 + 70.0 * smile_c),
										'angry': 2.0 * (1.0 - smile_c),
										'sad': 2.0 * (1.0 - smile_c),
										'fear': 1.0 * (1.0 - smile_c),
										'disgust': 0.5 * (1.0 - smile_c),
										'surprise': 1.5 * smile_c,
									}
									shared_data['ema_emotions'], _ = smooth_emotions_ema(heuristic, shared_data.get('ema_emotions', {}), alpha=shared_data.get('ema_alpha', 0.35))
									max_emotion = max(shared_data['ema_emotions'], key=shared_data['ema_emotions'].get)
									shared_data['emotions'] = shared_data['ema_emotions']
									shared_data['dominant_emotion'] = f"{max_emotion} (heuristic)"
									shared_data['face_region'] = region
						else:
							with data_lock:
								shared_data['dominant_emotion'] = "No Face"
								shared_data['emotions'] = {}
								shared_data['face_region'] = None
			except Exception:
				with data_lock:
					shared_data['dominant_emotion'] = "No Face"
					shared_data['emotions'] = {}
					shared_data['face_region'] = None

			# Periodic screenshot fallback when iris tracking is unavailable
			if not _mediapipe_available:
				with data_lock:
					now = time.time()
					if (now - shared_data.get('last_screenshot_time', 0.0)) >= shared_data.get('screenshot_interval', 30.0):
						if current_frame is not None:
							save_screenshot(current_frame, user_name, shared_data['current_gaze_direction'], shared_data['video_timestamp'])
							shared_data['last_screenshot_time'] = now
			# Draw overlays to frame for video writer only
			with data_lock:
				# update simple emotion counts for summary
				try:
					label = shared_data.get('dominant_emotion') or ''
					label = label.split(' ')[0].strip().lower()
					if label and label not in ['no', 'neutral']:
						cnts = shared_data.get('emotion_counts', {})
						cnts[label] = cnts.get(label, 0) + 1
						shared_data['emotion_counts'] = cnts
				except Exception:
					pass
				# overlays
				if shared_data['face_region']:
					face_region = shared_data['face_region']
					x, y, w_, h_ = face_region['x'], face_region['y'], face_region['w'], face_region['h']
					cv2.rectangle(frame, (x, y), (x + w_, y + h_), (255, 0, 0), 2)
				# HUD
				font = cv2.FONT_HERSHEY_SIMPLEX
				white = (255, 255, 255)
				cyan = (255, 255, 0)
				green = (0, 255, 0)
				yellow = (0, 255, 255)
				red = (0, 0, 255)
				text_y = 35
				# video time
				cv2.putText(frame, f"Time: {format_video_time(shared_data['video_timestamp'])}", (15, text_y), font, 0.8, cyan, 2, cv2.LINE_AA)
				text_y += 35
				# dominant emotion
				emo = shared_data.get('dominant_emotion', 'No Face')
				emo_color = white if 'detecting' not in emo else yellow
				cv2.putText(frame, f"Emotion: {emo}", (15, text_y), font, 1.0, emo_color, 2, cv2.LINE_AA)
				text_y += 35
				# top emotions (text bars)
				top = []
				if shared_data.get('emotions'):
					top = sorted(shared_data['emotions'].items(), key=lambda kv: kv[1], reverse=True)[:4]
				for name, score in top:
					bar_w = int(score * 2)
					cv2.rectangle(frame, (15, text_y - 10), (15 + bar_w, text_y), (50, 200, 50), -1)
					cv2.putText(frame, f"{name.capitalize()}: {score:.1f}%", (230, text_y), font, 0.7, white, 2, cv2.LINE_AA)
					text_y += 28
				# smile
				cv2.putText(frame, f"Smile: {shared_data.get('smile_confidence', 0.0):.2f}", (15, text_y), font, 0.8, (180,220,255), 2, cv2.LINE_AA)
				text_y += 30
				# hand + eye turns
				cv2.putText(frame, f"Hand: {shared_data.get('hand_status','Inactive')}", (15, text_y), font, 0.9, white, 2, cv2.LINE_AA)
				text_y += 32
				cv2.putText(frame, f"Eye Turns: {shared_data.get('eye_turn_count',0)} (L:{shared_data.get('left_eye_turns',0)} R:{shared_data.get('right_eye_turns',0)})", (15, text_y), font, 0.9, green if shared_data.get('eye_turn_count',0)>0 else white, 2, cv2.LINE_AA)
				text_y += 32
				# gaze
				gaze = shared_data.get('current_gaze_direction','center')
				gaze_color = cyan
				if shared_data.get('gaze_start_time',0)>0 and gaze in ['left','right']:
					hold = time.time() - shared_data['gaze_start_time']
					if hold >= shared_data['gaze_hold_duration']:
						gaze_color = green
					elif hold >= shared_data['gaze_hold_duration']*0.5:
						gaze_color = yellow
					else:
						gaze_color = red
					cv2.putText(frame, f"Gaze: {gaze.upper()} ({hold:.1f}s)", (15, text_y), font, 0.9, gaze_color, 2, cv2.LINE_AA)
				else:
					cv2.putText(frame, f"Gaze: {gaze.upper()}", (15, text_y), font, 0.9, gaze_color, 2, cv2.LINE_AA)
				text_y += 32
			# Write frame to video
			if writer_initialized and VIDEO_WRITER is not None:
				try:
					VIDEO_WRITER.write(frame)
				except Exception:
					pass
			# Throttle and publish update
			if on_update is not None and (time.time() - last_update_ts) >= 0.5:
				last_update_ts = time.time()
				with data_lock:
					state = {k: v for k, v in shared_data.items() if k not in ['hand_landmarks']}
				on_update(state)
			# Optional: pace loop a bit to avoid 100% CPU
			time.sleep(0.001)
	finally:
		cap.release()
		try:
			# append summary slate at end of analyzed video
			if writer_initialized and VIDEO_WRITER is not None:
				# build summary frame over last frame size or default HD
				try:
					height, width = frame.shape[:2]
				except Exception:
					width, height = 1280, 720
				summary = []
				with data_lock:
					eye_turns = shared_data.get('eye_turn_count', 0)
					l_turns = shared_data.get('left_eye_turns', 0)
					r_turns = shared_data.get('right_eye_turns', 0)
					cnts = shared_data.get('emotion_counts', {})
					top2 = sorted(cnts.items(), key=lambda kv: kv[1], reverse=True)[:2]
					top_txt = ", ".join([k.capitalize() for k,_ in top2]) if top2 else "-"
				summary.append(f"Analysis complete for {user_name}")
				summary.append(f"Total eye turns: {eye_turns} (L:{l_turns} R:{r_turns})")
				summary.append(f"Top emotions: {top_txt}")
				# create slate
				slate = np.zeros((height, width, 3), dtype=np.uint8)
				cv2.rectangle(slate, (40, 40), (width-40, height-40), (30, 30, 30), -1)
				font = cv2.FONT_HERSHEY_SIMPLEX
				y = 140
				cv2.putText(slate, "SESSION SUMMARY", (80, y), font, 1.6, (255,255,255), 3, cv2.LINE_AA)
				y += 70
				for line in summary:
					cv2.putText(slate, line, (80, y), font, 1.2, (200,230,255), 2, cv2.LINE_AA)
					y += 60
				# hold slate ~3s
				for _ in range(int(fps*3)):
					VIDEO_WRITER.write(slate)
				VIDEO_WRITER.release()
		except Exception:
			pass
		with data_lock:
			shared_data['is_running'] = False


def cleanup_and_finalize(user_name: str):
	global LOG_FILE, ANALYZED_VIDEO_PATH
	with data_lock:
		shared_data['is_running'] = False
	time.sleep(0.5)
	if LOG_FILE and os.path.exists(LOG_FILE):
		try:
			safe_name = "".join(c for c in user_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
			user_screenshot_folder = os.path.join(SCREENSHOT_DIR, safe_name)
			with open(LOG_FILE, 'a', encoding='utf-8') as f:
				f.write("\n" + "="*80 + "\n")
				f.write(f"Video analysis session ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
				f.write(f"User: {user_name}\n")
				f.write(f"Total eye turns detected: {shared_data.get('eye_turn_count', 0)}\n")
				f.write(f"Screenshots saved in: {user_screenshot_folder}\n")
				if ANALYZED_VIDEO_PATH:
					f.write(f"Analyzed video saved: {os.path.abspath(ANALYZED_VIDEO_PATH)}\n")
				f.write("="*80 + "\n")
		except Exception:
			pass


def run_analysis(user_name: str, video_path: str, on_update: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
	"""Run the analysis headlessly and report progress via callback. Returns paths of outputs."""
	with data_lock:
		shared_data['is_running'] = True
		shared_data['eye_turn_count'] = 0
		shared_data['left_eye_turns'] = 0
		shared_data['right_eye_turns'] = 0
		shared_data['gaze_history'] = []
		shared_data['emotion_start_time'] = {}
		shared_data['ema_emotions'] = {}
	safe_name = setup_logging(user_name)
	# Start helper threads
	thr_audio = threading.Thread(target=audio_listener, daemon=True)
	thr_audio.start()
	thr_log = threading.Thread(target=logging_thread, args=(user_name,), daemon=True)
	thr_log.start()
	# Process video
	video_process(user_name, video_path, on_update=on_update, headless=True)
	cleanup_and_finalize(user_name)
	return {
		'log_file': os.path.abspath(LOG_FILE) if LOG_FILE else None,
		'analyzed_video': os.path.abspath(ANALYZED_VIDEO_PATH) if ANALYZED_VIDEO_PATH else None,
		'screenshots_folder': os.path.abspath(USER_SCREENSHOT_FOLDER) if USER_SCREENSHOT_FOLDER else None,
	}