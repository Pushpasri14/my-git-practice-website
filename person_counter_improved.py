import cv2
import sys
import time
from collections import defaultdict
import numpy as np

try:
	from ultralytics import YOLO
except Exception as e:
	print("Ultralytics not installed or import failed:", e)
	print("Install with: pip install ultralytics --upgrade")
	raise

try:
	from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception as e:
	print("deep_sort_realtime not installed or import failed:", e)
	print("Install with: pip install deep-sort-realtime")
	raise


class PersonCounter:
	"""Simplified high-accuracy person counter with YOLO + DeepSort.

	- Uses a single strong YOLO model pass
	- Applies NMS and class filtering via Ultralytics API
	- Feeds detections to DeepSort in the expected format
	- Minimal pre-filters to avoid suppressing true positives
	- Tracks confirmed people, logs stats, renders overlays
	"""

	def __init__(self):
		self.confirmed_ids = set()
		self.active_ids = set()
		self.rejected_ids = set()
		self.frames_seen = defaultdict(int)
		self.track_area_history = defaultdict(list)
		self.max_simultaneous_confirmed = 0
		self.max_simultaneous_frame = 0

		# Detection config (tuned moderate)
		self.conf_threshold = 0.25
		self.iou_threshold = 0.45
		self.max_det = 300

		# Size constraints to suppress obvious noise
		self.min_width = 8
		self.min_height = 12
		self.min_area = 100

		# Track confirmation
		self.min_track_length = 3

	def _bbox_width_height_area(self, xyxy):
		x1, y1, x2, y2 = xyxy
		w = max(0.0, x2 - x1)
		h = max(0.0, y2 - y1)
		return w, h, w * h

	def process_frame(self, frame, yolo_model, tracker, frame_idx):
		"""Run detection, tracking, and bookkeeping on a single frame."""
		h, w = frame.shape[:2]

		# 1) Run YOLO once per frame for person class only (class 0)
		results = yolo_model(
			frame,
			conf=self.conf_threshold,
			iou=self.iou_threshold,
			classes=[0],
			max_det=self.max_det,
			verbose=False,
		)

		# 2) Convert detections to DeepSort expected format
		# deep_sort_realtime expects: list of [ [x1,y1,x2,y2], confidence, class ]
		# Optionally a dict: {"feature":..., "detector":...}
		detections = []
		if results:
			for r in results:
				boxes = getattr(r, 'boxes', None)
				if boxes is None:
					continue
				xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
				confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
				for i in range(len(xyxy)):
					x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
					conf = float(confs[i])

					# Minimal size filter to reduce noise
					bw, bh, ba = self._bbox_width_height_area((x1, y1, x2, y2))
					if bw < self.min_width or bh < self.min_height or ba < self.min_area:
						continue

					detections.append(([x1, y1, x2, y2], conf, 'person'))

		# 3) Update tracker
		tracks = tracker.update_tracks(detections, frame=frame)

		# 4) Bookkeeping and stats
		self.active_ids.clear()
		for t in tracks:
			if not t.is_confirmed():
				continue
			track_id = t.track_id
			self.active_ids.add(track_id)
			ltrb = t.to_ltrb()
			bw, bh, ba = self._bbox_width_height_area(ltrb)
			self.track_area_history[track_id].append(ba)
			self.frames_seen[track_id] += 1

			if self.frames_seen[track_id] >= self.min_track_length:
				self.confirmed_ids.add(track_id)

		current_confirmed_active = len(self.confirmed_ids & self.active_ids)
		if current_confirmed_active > self.max_simultaneous_confirmed:
			self.max_simultaneous_confirmed = current_confirmed_active
			self.max_simultaneous_frame = frame_idx

		return tracks, detections

	def draw_overlays(self, frame, tracks, detections, frame_idx, total_frames):
		# Draw detections (thin cyan)
		for (x1, y1, x2, y2), conf, _ in detections:
			cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)
			cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

		# Draw tracks (green for confirmed, orange for warming)
		for t in tracks:
			if not t.is_confirmed():
				continue
			track_id = t.track_id
			x1, y1, x2, y2 = [int(v) for v in t.to_ltrb()]
			is_confirmed = track_id in self.confirmed_ids
			color = (0, 200, 0) if is_confirmed else (0, 165, 255)
			thickness = 3 if is_confirmed else 2
			cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
			cv2.putText(frame, f"ID {track_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

		# Stats header
		active_confirmed = len(self.confirmed_ids & self.active_ids)
		cv2.putText(frame, f"CONFIRMED: {active_confirmed}  MAX: {self.max_simultaneous_confirmed}", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
		cv2.putText(frame, f"ACTIVE TRACKS: {len(self.active_ids)}  DETS: {len(detections)}", (24, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
		cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (24, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
	video_path = 'new7.mp4'
	output_path = 'person_counter_improved.avi'

	# Prefer a stronger, recent model if available; fallback to n if needed
	# Examples: 'yolov8s.pt', 'yolov8n.pt', 'yolo11s.pt', 'yolo11n.pt'
	model_candidates = ['yolov8s.pt', 'yolo11s.pt', 'yolov8n.pt', 'yolo11n.pt']
	model_path = None
	for cand in model_candidates:
		try:
			YOLO(cand)  # quick probe
			model_path = cand
			break
		except Exception:
			continue
	if model_path is None:
		print('No local YOLO model weights found from candidates. Downloading yolov8n...')
		model_path = 'yolov8n.pt'

	model = YOLO(model_path)

	# Tracker tuned for stability
	tracker = DeepSort(
		max_age=30,
		n_init=3,
		max_cosine_distance=0.6,
		nn_budget=150,
	)

	counter = PersonCounter()

	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print(f"Failed to open video: {video_path}")
		sys.exit(1)

	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

	print(f"Model: {model_path} | Video: {video_path} | FPS: {fps:.1f}")
	print("Starting processing...")
	start = time.time()

	frame_idx = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		frame_idx += 1

		tracks, detections = counter.process_frame(frame, model, tracker, frame_idx)
		counter.draw_overlays(frame, tracks, detections, frame_idx, total_frames)

		out.write(frame)
		cv2.imshow('Person Counter (Improved)', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	out.release()
	cv2.destroyAllWindows()

	elapsed = time.time() - start
	print("\nProcessing done.")
	print(f"Max simultaneous confirmed: {counter.max_simultaneous_confirmed} at frame {counter.max_simultaneous_frame}")
	print(f"Total confirmed ids: {len(counter.confirmed_ids)}")
	print(f"Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
	main()