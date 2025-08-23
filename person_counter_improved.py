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

# SAHI optional import
SAHI_AVAILABLE = True
try:
	from sahi.models.ultralytics import UltralyticsDetectionModel
	from sahi.predict import get_sliced_prediction
except Exception as e:
	print("SAHI not installed or import failed:", e)
	print("Install with: pip install sahi")
	SAHI_AVAILABLE = False


class PersonCounter:
	"""High-accuracy person counter with YOLO + DeepSort + SAHI (for top/overhead views).

	- One YOLO pass per frame (higher imgsz) for regular-size persons
	- SAHI multi-scale slicing for small/overhead persons
	- IoU-based deduplication between YOLO and SAHI detections
	- Proper DeepSort detection tuple format
	- Tuned minimal size filters to include small top-view humans
	"""

	def __init__(self):
		self.confirmed_ids = set()
		self.active_ids = set()
		self.frames_seen = defaultdict(int)
		self.track_area_history = defaultdict(list)
		self.max_simultaneous_confirmed = 0
		self.max_simultaneous_frame = 0

		# Detection config
		self.conf_threshold = 0.25  # direct YOLO
		self.iou_threshold = 0.45   # NMS for YOLO
		self.max_det = 400
		self.imgsz = 960           # larger for better small-object recall

		# SAHI config (small + medium slices)
		self.sahi_confs = [
			{"slice_height": 256, "slice_width": 256, "overlap": 0.25, "conf": 0.15},
			{"slice_height": 512, "slice_width": 512, "overlap": 0.30, "conf": 0.18},
		]

		# Size constraints to suppress obvious noise but keep small top-view humans
		self.min_width = 6
		self.min_height = 8
		self.min_area = 64

		# Track confirmation
		self.min_track_length = 3

		# Deduplication
		self.dup_iou_threshold = 0.5

	@staticmethod
	def _bbox_width_height_area(xyxy):
		x1, y1, x2, y2 = xyxy
		w = max(0.0, x2 - x1)
		h = max(0.0, y2 - y1)
		return w, h, w * h

	@staticmethod
	def _iou(b1, b2):
		x1, y1, x2, y2 = b1
		x1b, y1b, x2b, y2b = b2
		xi1 = max(x1, x1b)
		yi1 = max(y1, y1b)
		xi2 = min(x2, x2b)
		yi2 = min(y2, y2b)
		if xi2 <= xi1 or yi2 <= yi1:
			return 0.0
		inter = (xi2 - xi1) * (yi2 - yi1)
		area1 = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
		area2 = max(0.0, (x2b - x1b)) * max(0.0, (y2b - y1b))
		union = area1 + area2 - inter
		return inter / union if union > 0 else 0.0

	def _deduplicate(self, dets):
		"""Deduplicate detections by IoU keeping higher-confidence ones.
		Input: list of (bbox[x1,y1,x2,y2], conf, cls)
		"""
		unique = []
		for bbox, conf, cls in dets:
			replace_idx = -1
			best_iou = 0.0
			for j, (ubox, uconf, ucls) in enumerate(unique):
				iou = self._iou(bbox, ubox)
				if iou > self.dup_iou_threshold and iou > best_iou:
					best_iou = iou
					replace_idx = j
			if replace_idx >= 0:
				if conf > unique[replace_idx][1]:
					unique[replace_idx] = (bbox, conf, cls)
			else:
				unique.append((bbox, conf, cls))
		return unique

	def _run_yolo(self, frame, yolo_model):
		results = yolo_model(
			frame,
			conf=self.conf_threshold,
			iou=self.iou_threshold,
			classes=[0],
			max_det=self.max_det,
			imgsz=self.imgsz,
			verbose=False,
		)
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
					bw, bh, ba = self._bbox_width_height_area((x1, y1, x2, y2))
					if bw < self.min_width or bh < self.min_height or ba < self.min_area:
						continue
					detections.append(([x1, y1, x2, y2], conf, 'person'))
		return detections

	def _run_sahi(self, frame, sahi_model):
		detections = []
		if not SAHI_AVAILABLE or sahi_model is None:
			return detections
		for cfg in self.sahi_confs:
			try:
				res = get_sliced_prediction(
					frame,
					sahi_model,
					slice_height=cfg["slice_height"],
					slice_width=cfg["slice_width"],
					overlap_height_ratio=cfg["overlap"],
					overlap_width_ratio=cfg["overlap"],
				)
				for obj in res.object_prediction_list:
					# Ultralytics model: person class id is typically 0
					if getattr(obj.category, 'id', 0) != 0:
						continue
					x1, y1, x2, y2 = obj.bbox.to_xyxy()
					conf = float(obj.score.value)
					bw, bh, ba = self._bbox_width_height_area((x1, y1, x2, y2))
					if bw < self.min_width or bh < self.min_height or ba < self.min_area:
						continue
					detections.append(([float(x1), float(y1), float(x2), float(y2)], conf, 'person'))
			except Exception:
				# If SAHI fails for a slice size, skip it
				continue
		return detections

	def process_frame(self, frame, yolo_model, tracker, frame_idx, sahi_model=None):
		"""Run detection (YOLO + optional SAHI), deduplicate, track, and update stats."""
		# 1) Direct YOLO detections
		yolo_dets = self._run_yolo(frame, yolo_model)

		# 2) SAHI detections for small/overhead
		sahi_dets = self._run_sahi(frame, sahi_model) if sahi_model is not None else []

		# 3) Merge and deduplicate
		all_dets = self._deduplicate(yolo_dets + sahi_dets)

		# 4) Update tracker
		tracks = tracker.update_tracks(all_dets, frame=frame)

		# 5) Bookkeeping and stats
		self.active_ids.clear()
		for t in tracks:
			if not t.is_confirmed():
				continue
			track_id = t.track_id
			self.active_ids.add(track_id)
			ltrb = t.to_ltrb()
			_, _, ba = self._bbox_width_height_area(ltrb)
			self.track_area_history[track_id].append(ba)
			self.frames_seen[track_id] += 1
			if self.frames_seen[track_id] >= self.min_track_length:
				self.confirmed_ids.add(track_id)

		current_confirmed_active = len(self.confirmed_ids & self.active_ids)
		if current_confirmed_active > self.max_simultaneous_confirmed:
			self.max_simultaneous_confirmed = current_confirmed_active
			self.max_simultaneous_frame = frame_idx

		return tracks, all_dets, yolo_dets, sahi_dets

	def draw_overlays(self, frame, tracks, all_dets, yolo_dets, sahi_dets, frame_idx, total_frames):
		# Draw YOLO detections (thin cyan)
		for (x1, y1, x2, y2), conf, _ in yolo_dets:
			cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)
			cv2.putText(frame, f"Y {conf:.2f}", (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

		# Draw SAHI detections (thin magenta)
		for (x1, y1, x2, y2), conf, _ in sahi_dets:
			cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 1)
			cv2.putText(frame, f"S {conf:.2f}", (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

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
		cv2.putText(frame, f"ACTIVE: {len(self.active_ids)}  DETS(Y/S/All): {len(yolo_dets)}/{len(sahi_dets)}/{len(all_dets)}", (24, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
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

	# SAHI detection model (if available)
	sahi_model = None
	if SAHI_AVAILABLE:
		try:
			sahi_model = UltralyticsDetectionModel(model_path=model_path, confidence_threshold=0.15)
		except Exception as e:
			print('Failed to initialize SAHI model:', e)
			sahi_model = None

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

	print(f"Model: {model_path} | Video: {video_path} | FPS: {fps:.1f} | SAHI: {'on' if sahi_model is not None else 'off'}")
	print("Starting processing...")
	start = time.time()

	frame_idx = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		frame_idx += 1

		tracks, all_dets, yolo_dets, sahi_dets = counter.process_frame(frame, model, tracker, frame_idx, sahi_model)
		counter.draw_overlays(frame, tracks, all_dets, yolo_dets, sahi_dets, frame_idx, total_frames)

		out.write(frame)
		cv2.imshow('Person Counter (Improved + SAHI)', frame)
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