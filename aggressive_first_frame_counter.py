import cv2
import signal
import sys
import numpy as np
from ultralytics import YOLO
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
from deep_sort_realtime.deepsort_tracker import DeepSort
import math
import time
from collections import defaultdict


class AggressivePersonCounter:
    def __init__(self):
        self.confirmed_persons = set()
        self.track_history = defaultdict(list)
        self.track_confidence_history = defaultdict(list)
        self.track_area_history = defaultdict(list)
        self.track_frames_seen = defaultdict(int)
        self.rejected_tracks = set()
        self.active_tracks = set()
        self.max_simultaneous_active = 0
        self.max_simultaneous_frame = 0
        self.max_active_tracks = 0
        self.max_active_tracks_frame = 0

        # *** VERY AGGRESSIVE PARAMETERS FOR CROWDED SCENES ***
        self.min_confidence = 0.15         # Very low confidence
        self.min_track_length = 3          # Very short track requirement
        self.min_avg_confidence = 0.2      # Very low average confidence
        self.max_area_variation = 2.0      # Allow huge variation
        self.min_person_area = 200         # Very small minimum area
        self.max_person_area = 100000      # Very large max area
        self.aspect_ratio_range = (0.2, 8.0)  # Extremely wide range
        self.position_stability_threshold = 200  # Allow lots of movement

        # Additional parameters for crowded scenes
        self.min_width = 8                 # Very small minimum width
        self.min_height = 10               # Very small minimum height
        self.enable_area_based_confidence = True

    def calculate_distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def get_center(self, bbox):
        l, t, r, b = bbox
        return ((l + r) / 2, (t + b) / 2)

    def calculate_bbox_metrics(self, bbox):
        l, t, r, b = bbox
        width = r - l
        height = b - t
        area = width * height
        aspect_ratio = height / width if width > 0 else 0
        return width, height, area, aspect_ratio

    def estimate_confidence_from_area(self, area):
        """Estimate confidence based on person area for small detections"""
        if area < 500:
            return 0.3
        elif area < 1000:
            return 0.4
        elif area < 2000:
            return 0.5
        else:
            return 0.6

    def is_valid_person_detection(self, bbox, confidence):
        """VERY AGGRESSIVE validation for crowded scenes"""
        width, height, area, aspect_ratio = self.calculate_bbox_metrics(bbox)

        # Boost confidence for small areas (distant people)
        if self.enable_area_based_confidence and area < 1000:
            confidence = max(confidence, self.estimate_confidence_from_area(area))

        # Extremely relaxed criteria
        criteria = [
            confidence >= self.min_confidence,  # Very low confidence
            width >= self.min_width,            # Very small minimum width
            height >= self.min_height,          # Very small minimum height
            area >= self.min_person_area,       # Very small minimum area
            area <= self.max_person_area,       # Very large maximum area
            aspect_ratio >= self.aspect_ratio_range[0],  # Very wide range
            aspect_ratio <= self.aspect_ratio_range[1],
        ]

        return all(criteria)

    def analyze_track_quality(self, track_id):
        """Very lenient track quality analysis"""
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return False, "Insufficient data"

        history = self.track_history[track_id]
        confidences = self.track_confidence_history[track_id]
        areas = self.track_area_history[track_id]

        # Very lenient checks
        avg_confidence = np.mean(confidences)
        if avg_confidence < self.min_avg_confidence:
            return False, f"Low avg confidence: {avg_confidence:.2f}"

        # For small areas, be even more lenient
        avg_area = np.mean(areas)
        if avg_area < 800:  # Small person
            return True, "Small person - accepting"

        # Check area variation (very tolerant)
        if len(areas) > 1:
            area_variation = (max(areas) - min(areas)) / np.mean(areas)
            if area_variation > self.max_area_variation:
                return False, f"High area variation: {area_variation:.2f}"

        return True, "Valid person track"

    def update_tracks(self, tracks, frame_number):
        """Update tracks with very lenient analysis"""
        current_tracks = set()
        self.active_tracks = set()

        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id
                current_tracks.add(track_id)
                self.active_tracks.add(track_id)

                if track_id in self.rejected_tracks:
                    continue

                bbox = track.to_ltrb()
                center = self.get_center(bbox)
                width, height, area, aspect_ratio = self.calculate_bbox_metrics(bbox)

                self.track_history[track_id].append({
                    'position': center,
                    'bbox': bbox,
                    'frame': frame_number,
                    'area': area,
                    'aspect_ratio': aspect_ratio
                })

                # Enhanced confidence estimation
                if area < 800:
                    estimated_confidence = self.estimate_confidence_from_area(area)
                else:
                    estimated_confidence = min(0.8, 0.3 + (area / 5000))

                self.track_confidence_history[track_id].append(estimated_confidence)
                self.track_area_history[track_id].append(area)

                # Keep moderate history
                max_history = 30
                if len(self.track_history[track_id]) > max_history:
                    self.track_history[track_id] = self.track_history[track_id][-max_history:]
                    self.track_confidence_history[track_id] = self.track_confidence_history[track_id][-max_history:]
                    self.track_area_history[track_id] = self.track_area_history[track_id][-max_history:]

                self.track_frames_seen[track_id] += 1

                # Very early confirmation
                if self.track_frames_seen[track_id] >= self.min_track_length:
                    is_valid, reason = self.analyze_track_quality(track_id)
                    if is_valid:
                        if track_id not in self.confirmed_persons:
                            self.confirmed_persons.add(track_id)
                            if track_id in self.active_tracks:
                                avg_area = np.mean(self.track_area_history[track_id])
                                print(f"\u2713 Confirmed person: ID {track_id} (Frames: {self.track_frames_seen[track_id]}, Area: {avg_area:.0f})")
                    else:
                        if track_id not in self.rejected_tracks:
                            self.rejected_tracks.add(track_id)
                            if track_id in self.active_tracks:
                                print(f"\u2717 Rejected track: ID {track_id} - {reason}")

        # Update statistics
        current_active_confirmed = len(self.confirmed_persons & self.active_tracks)

        if current_active_confirmed > self.max_simultaneous_active:
            self.max_simultaneous_active = current_active_confirmed
            self.max_simultaneous_frame = frame_number
            print(f"NEW MAXIMUM CONFIRMED: {self.max_simultaneous_active} simultaneous persons at frame {frame_number}")

        current_active_tracks_count = len(self.active_tracks)
        if current_active_tracks_count > self.max_active_tracks:
            self.max_active_tracks = current_active_tracks_count
            self.max_active_tracks_frame = frame_number
            print(f"NEW MAXIMUM ACTIVE TRACKS: {self.max_active_tracks} total tracks at frame {frame_number}")

        return current_tracks

    def get_track_status(self, track_id):
        if track_id in self.confirmed_persons:
            return "CONFIRMED", (0, 255, 0)
        elif track_id in self.rejected_tracks:
            return "REJECTED", (0, 0, 255)
        elif self.track_frames_seen[track_id] >= self.min_track_length:
            return "ANALYZING", (255, 165, 0)
        else:
            frames_needed = self.min_track_length - self.track_frames_seen[track_id]
            return f"TRACK({frames_needed})", (0, 165, 255)


# Global variables
person_counter = None
cap = None
out = None
frame_count = 0
first_frame_unique_count = None  # Freeze unique person count from the first frame


def signal_handler(sig, frame):
    if person_counter:
        print(f"\nFinal Results:")
        print(f"Maximum Active Tracks: {person_counter.max_active_tracks}")
        print(f"Maximum Confirmed Persons: {person_counter.max_simultaneous_active}")
    if cap:
        cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# Initialize aggressive counter
person_counter = AggressivePersonCounter()

print("Loading YOLO model for AGGRESSIVE crowded scene detection...")
model = YOLO("yolo11n.pt")

# VERY AGGRESSIVE SAHI settings
detection_model = UltralyticsDetectionModel(
    model_path="yolo11n.pt",
    confidence_threshold=0.1  # Extremely low threshold
)

# Very sensitive tracker settings
tracker = DeepSort(
    max_age=20,              # Shorter max age for busy scenes
    n_init=2,               # Only 2 frames needed to confirm
    max_cosine_distance=0.8, # Very lenient appearance matching
    nn_budget=100           # Lower budget for speed
)

# Video setup
cap = cv2.VideoCapture("new7.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter("aggressive_crowded_count.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

print(f"Processing with AGGRESSIVE settings for crowded scenes:")
print(f"- Min confidence: {person_counter.min_confidence}")
print(f"- Min area: {person_counter.min_person_area}")
print(f"- Min track length: {person_counter.min_track_length}")
print(f"- Detection threshold: 0.1")

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Multi-approach detection for maximum coverage
    all_detections = []

    # 1. SAHI with very small slices and high overlap
    try:
        result = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=200,  # Very small slices
            slice_width=200,
            overlap_height_ratio=0.4,  # High overlap
            overlap_width_ratio=0.4,
        )

        for obj in result.object_prediction_list:
            if obj.category.id == 0:  # person class
                x1, y1, x2, y2 = obj.bbox.to_xyxy()
                bbox = [x1, y1, x2, y2]
                confidence = obj.score.value
                all_detections.append((bbox, confidence, 'sahi'))
    except Exception:
        print(f"SAHI failed at frame {frame_count}")

    # 2. Direct YOLO with multiple scales
    for conf_thresh in [0.1, 0.15, 0.2]:
        try:
            results = model(frame, conf=conf_thresh, classes=[0], imgsz=640)
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        if len(box.xyxy) > 0:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            bbox = [x1, y1, x2, y2]
                            all_detections.append((bbox, confidence, f'yolo_{conf_thresh}'))
        except Exception:
            continue

    # 3. Remove duplicates and filter
    unique_detections = []
    for i, (bbox1, conf1, source1) in enumerate(all_detections):
        is_duplicate = False
        for j, (bbox2, conf2, source2) in enumerate(unique_detections):
            # Check overlap
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2

            overlap_x = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
            overlap_y = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
            overlap_area = overlap_x * overlap_y

            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

            if overlap_area > 0.3 * min(area1, area2):  # 30% overlap threshold
                is_duplicate = True
                # Keep the one with higher confidence
                if conf1 > conf2:
                    unique_detections[j] = (bbox1, conf1, source1)
                break

        if not is_duplicate:
            unique_detections.append((bbox1, conf1, source1))

    # Apply validation
    detections = []
    for bbox, conf, source in unique_detections:
        if person_counter.is_valid_person_detection(bbox, conf):
            detections.append((bbox, conf, 'person'))

    # Freeze unique count from first frame
    if frame_count == 1 and first_frame_unique_count is None:
        first_frame_unique_count = len(detections)

    print(f"Frame {frame_count}: Raw detections: {len(all_detections)}, Unique: {len(unique_detections)}, Valid: {len(detections)}")

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)
    current_tracks = person_counter.update_tracks(tracks, frame_count)

    # Enhanced visualization
    # Draw all unique detections in yellow (for debugging)
    for bbox, conf, source in unique_detections[:50]:  # Limit to first 50 for visibility
        l, t, r, b = map(int, bbox)
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 255), 1)  # Yellow
        area = (r - l) * (b - t)
        cv2.putText(frame, f"{conf:.2f}|{area:.0f}", (l, t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

    # Draw valid detections in cyan
    for bbox, conf, _ in detections:
        l, t, r, b = map(int, bbox)
        cv2.rectangle(frame, (l, t), (r, b), (255, 255, 0), 2)  # Cyan

    # Draw confirmed tracks
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            status, color = person_counter.get_track_status(track_id)

            # Draw thick bounding box for confirmed persons
            thickness = 4 if track_id in person_counter.confirmed_persons else 2
            cv2.rectangle(frame, (l, t), (r, b), color, thickness)

            # Add track info
            info_text = f"ID:{track_id}"
            cv2.putText(frame, info_text, (l, t - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, status, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add area for confirmed
            if track_id in person_counter.confirmed_persons and person_counter.track_area_history[track_id]:
                avg_area = np.mean(person_counter.track_area_history[track_id])
                cv2.putText(frame, f"A:{avg_area:.0f}", (l, b + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Status display
    if first_frame_unique_count is not None:
        cv2.putText(
            frame,
            f"UNIQUE PERSONS (first frame): {first_frame_unique_count}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )
    else:
        active_confirmed = len(person_counter.confirmed_persons & person_counter.active_tracks)
        cv2.putText(
            frame,
            f"CONFIRMED: {active_confirmed} | MAX: {person_counter.max_simultaneous_active}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )

    cv2.putText(
        frame,
        f"ACTIVE TRACKS: {len(person_counter.active_tracks)} | MAX: {person_counter.max_active_tracks}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"DETECTIONS: {len(detections)} | UNIQUE: {len(unique_detections)}",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Frame: {frame_count}/{total_frames} ({frame_count / total_frames * 100:.1f}%)",
        (30, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    out.write(frame)
    cv2.imshow("AGGRESSIVE Crowded Scene Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Final results
end_time = time.time()
processing_time = end_time - start_time

print(f"\n{'=' * 80}")
print(f"AGGRESSIVE CROWDED SCENE COUNTING RESULTS")
print(f"{'=' * 80}")
print(f"MAXIMUM ACTIVE TRACKS: {person_counter.max_active_tracks}")
print(f"MAXIMUM CONFIRMED PERSONS: {person_counter.max_simultaneous_active}")
print(f"Total confirmed persons: {len(person_counter.confirmed_persons)}")
print(f"UNIQUE PERSONS (first frame): {first_frame_unique_count if first_frame_unique_count is not None else 'N/A'}")
print(f"Processing time: {processing_time:.1f}s")
print(f"AGGRESSIVE MODE: Very low thresholds for maximum detection")