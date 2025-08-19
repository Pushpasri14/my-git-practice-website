import argparse
import os
import sys
import time
from typing import Optional, Dict, Set

import cv2
import numpy as np
from tqdm import tqdm
import supervision as sv

# --------- Optional imports (only used if backend="roboflow") ----------
try:
    from inference import get_model as rf_get_model  # Roboflow Inference SDK
except Exception:
    rf_get_model = None

# --------- Optional imports (only used if backend="yolo") --------------
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def build_yolo_backend(model_path: str, device: str = "cpu"):
    """Loads an Ultralytics YOLO model and returns an infer(image) -> sv.Detections callable."""
    if YOLO is None:
        raise RuntimeError("Ultralytics not installed. Run: pip install ultralytics")

    model = YOLO(model_path)

    def infer(image_bgr: np.ndarray, conf: float) -> sv.Detections:
        results = model.predict(image_bgr, conf=conf, verbose=False, device=device)[0]
        dets = sv.Detections.from_ultralytics(results)
        return dets

    return infer


def build_roboflow_backend(model_id: str, api_key: Optional[str]):
    """Loads a Roboflow Inference model and returns an infer(image) -> sv.Detections callable."""
    if rf_get_model is None:
        raise RuntimeError("Roboflow Inference SDK not installed. Run: pip install inference")

    if api_key is None:
        api_key = os.environ.get("ROBOFLOW_API_KEY", None)

    model = rf_get_model(model_id=model_id, api_key=api_key)

    def infer(image_bgr: np.ndarray, conf: float) -> sv.Detections:
        res = model.infer(image_bgr, confidence=conf)[0]
        dets = sv.Detections.from_inference(res)
        return dets

    return infer


def nms_cv2(xyxy: np.ndarray, scores: np.ndarray, score_thresh: float = 0.05, iou_thresh: float = 0.5):
    """Apply OpenCV NMS and return indices to keep."""
    if len(xyxy) == 0:
        return []
    boxes_xywh = []
    for x1, y1, x2, y2 in xyxy:
        boxes_xywh.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
    idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), score_thresh, iou_thresh)
    if len(idxs) == 0:
        return []
    if isinstance(idxs, (list, tuple)):
        return [int(i) for i in idxs]
    return [int(i) for i in idxs.flatten()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count unique persons in entire video using robust tracking (ByteTrack) and per-frame NMS."
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", default="unique_persons_output.mp4", help="Path to save annotated video")
    parser.add_argument("--backend", choices=["yolo", "roboflow"], default="yolo",
                        help="Detection backend to use")

    # YOLO args
    parser.add_argument("--yolo-model", default="yolov8n.pt",
                        help="Ultralytics model path/name")

    # Roboflow args
    parser.add_argument("--rf-model-id", default=None, help="Roboflow model id")
    parser.add_argument("--rf-api-key", default=None, help="Roboflow API key")

    # Detection params
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for person detection")

    # Processing params
    parser.add_argument("--device", default="cpu", help="Device hint for backend (Ultralytics)")
    parser.add_argument("--max-frames", type=int, default=-1, help="Process only first N frames (-1 for all)")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    parser.add_argument("--save-report", default="unique_persons_report.txt", help="File to save detailed report")

    # NMS params (post-processing safeguard)
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold")

    return parser.parse_args()


def main():
    args = parse_args()

    # Build backend
    if args.backend == "yolo":
        infer_fn = build_yolo_backend(args.yolo_model, device=args.device)
        backend_name = f"YOLO ({args.yolo_model})"
    elif args.backend == "roboflow":
        if args.rf_model_id is None:
            raise ValueError("--rf-model-id is required when backend=roboflow")
        infer_fn = build_roboflow_backend(args.rf_model_id, api_key=args.rf_api_key)
        backend_name = f"Roboflow ({args.rf_model_id})"
    else:
        raise ValueError("Unsupported backend")

    # Initialize ByteTrack for stable identities
    byte_tracker = sv.ByteTrack()

    # Maintain stats per track id
    seen_track_ids: Set[int] = set()
    track_stats: Dict[int, Dict] = {}

    # Annotators
    box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color.GREEN)
    label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=1, color=sv.Color.WHITE)

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Could not open writer for: {args.out}")

    print(f"\n*** UNIQUE PERSON COUNTER (TRACKER-BASED) ***")
    print(f"{'='*50}")
    print(f"Backend: {backend_name}")
    print(f"Video: {args.video}")
    print(f"Total frames: {total_frames}")
    print(f"Detection confidence: {args.conf}")
    print(f"NMS IoU: {args.nms_iou}")
    print(f"Tracker: ByteTrack via supervision")
    print(f"{'='*50}\n")

    max_frames = args.max_frames if args.max_frames > 0 else total_frames
    processed = 0

    pbar = tqdm(total=max_frames, desc="Analyzing video for unique persons", unit="frame")
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed += 1
            if args.max_frames > 0 and processed > args.max_frames:
                break

            # Direct model inference on full frame (avoid slicing duplicates)
            detections: sv.Detections = infer_fn(frame, conf=args.conf)

            # Filter to person class id == 0 if class ids exist
            person_indices = []
            if len(detections) > 0 and detections.class_id is not None:
                for i in range(len(detections)):
                    if int(detections.class_id[i]) == 0:  # 0 = person
                        person_indices.append(i)
            else:
                # If class ids are missing, keep all
                person_indices = list(range(len(detections)))

            detections = detections[person_indices] if person_indices else sv.Detections.empty()

            # Extra NMS safeguard (Ultralytics already does NMS, but keep to fight occasional dupes)
            if len(detections) > 0 and detections.confidence is not None:
                keep = nms_cv2(detections.xyxy, detections.confidence, score_thresh=0.0, iou_thresh=args.nms_iou)
                detections = detections[keep] if keep else sv.Detections.empty()

            # Update tracker and accumulate unique IDs
            tracked = byte_tracker.update_with_detections(detections)

            # Update stats
            for i in range(len(tracked)):
                tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None
                if tid is None:
                    continue
                seen_track_ids.add(tid)
                if tid not in track_stats:
                    track_stats[tid] = {
                        "frames_seen": set(),
                        "total_detections": 0,
                        "first_seen_frame": processed,
                        "last_seen_frame": processed,
                        "total_confidence": 0.0,
                    }
                track_stats[tid]["frames_seen"].add(processed)
                track_stats[tid]["total_detections"] += 1
                track_stats[tid]["last_seen_frame"] = processed
                conf_i = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
                track_stats[tid]["total_confidence"] += conf_i

            # Prepare labels for annotation
            annotated = frame.copy()
            if len(tracked) > 0:
                labels = []
                for i in range(len(tracked)):
                    tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                    conf = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
                    labels.append(f"ID {tid} {conf:.2f}")

                annotated = box_annotator.annotate(scene=annotated, detections=tracked)
                annotated = label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)

            # Overlay status
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, 10), (420, 110), (0, 0, 0), -1)
            cv2.rectangle(overlay, (10, 10), (420, 110), (0, 255, 255), 2)
            cv2.putText(overlay, "TRACKING UNIQUE PERSONS", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, f"Unique persons so far: {len(seen_track_ids)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(overlay, f"Frame: {processed}/{max_frames}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)

            out.write(annotated)

            if args.show:
                cv2.imshow("Unique Person Counter (ByteTrack)", annotated)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    break

            pbar.set_postfix({"Unique": len(seen_track_ids)})
            pbar.update(1)

    finally:
        cap.release()
        out.release()
        if args.show:
            cv2.destroyAllWindows()
        pbar.close()

    # Final analysis
    dt = time.time() - t0
    unique_count = len(seen_track_ids)

    # Save detailed report
    with open(args.save_report, 'w', encoding='utf-8') as f:
        f.write("UNIQUE PERSON COUNT (TRACKER-BASED)\n")
        f.write("="*50 + "\n\n")
        f.write(f"Video: {args.video}\n")
        f.write(f"Total frames processed: {processed}\n")
        f.write(f"Processing time: {dt:.1f} seconds\n")
        f.write(f"Average FPS: {processed/max(1e-6, dt):.2f}\n\n")
        f.write("FINAL RESULTS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"*** UNIQUE PERSONS IN ENTIRE VIDEO: {unique_count} ***\n\n")
        f.write("CONFIRMED UNIQUE TRACKS DETAILS:\n")
        f.write("-" * 30 + "\n")
        if track_stats:
            for tid, details in sorted(track_stats.items()):
                avg_conf = details["total_confidence"] / max(1, details["total_detections"])
                f.write(f"Track ID {tid}:\n")
                f.write(f"  - Total detections: {details['total_detections']}\n")
                f.write(f"  - Frames appeared in: {len(details['frames_seen'])}\n")
                f.write(f"  - Average confidence: {avg_conf:.3f}\n")
                f.write(f"  - First seen: frame {details['first_seen_frame']}\n")
                f.write(f"  - Last seen: frame {details['last_seen_frame']}\n\n")
        else:
            f.write("No persons tracked.\n\n")

    # Print final results
    print(f"\n" + "="*70)
    print(f"*** FINAL RESULT: {unique_count} UNIQUE PERSONS IN ENTIRE VIDEO ***")
    print(f"="*70)
    if track_stats:
        print(f"\n*** UNIQUE TRACKS SUMMARY ***")
        for tid, details in sorted(track_stats.items()):
            avg_conf = details["total_confidence"] / max(1, details["total_detections"])
            print(f"ID {tid}: {len(details['frames_seen'])} frames, {details['total_detections']} detections, {avg_conf:.2f} avg conf")
    else:
        print("No tracks found.")

    print(f"\n*** PROCESSING STATS ***")
    print(f"Frames processed: {processed}")
    print(f"Processing time: {dt:.1f}s")
    print(f"Average FPS: {processed/max(1e-6, dt):.2f}")

    print(f"\n*** OUTPUT FILES ***")
    print(f"Annotated video: {args.out}")
    print(f"Detailed report: {args.save_report}")
    print(f"\n*** Analysis complete! ***")


if __name__ == "__main__":
    main()

