import argparse
import os
import time
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm
import supervision as sv

# Optional import: Ultralytics YOLO (for detection)
try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None


def build_yolo_backend(model_path: str, device: str = "cpu") -> Tuple[Callable[[np.ndarray, float], sv.Detections], List[str]]:
    """
    Load an Ultralytics YOLO model and return:
      - infer(image_bgr, conf) -> sv.Detections
      - class_names list (index -> name)
    """
    if YOLO is None:
        raise RuntimeError("Ultralytics not installed. Run: pip install ultralytics")

    model = YOLO(model_path)

    # Try to hint device via predict call. Ultralytics will manage model placement.
    # Collect class names (COCO models: 0 == person)
    # Ultralytics keeps names on the model
    names_map = getattr(model, "names", None)
    if isinstance(names_map, dict):
        class_names = [names_map[k] for k in sorted(names_map.keys())]
    elif isinstance(names_map, list):
        class_names = names_map
    else:
        class_names = []

    def infer(image_bgr: np.ndarray, conf: float) -> sv.Detections:
        results = model.predict(source=image_bgr, conf=conf, device=device, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections

    return infer, class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and count unique persons in a video using SAHI-style slicing and tracking."
    )
    parser.add_argument("--video", required=True, help="Path to input video (e.g., input.mp4) or camera index (e.g., 0)")
    parser.add_argument("--out", default="output_unique_persons.mp4", help="Output annotated video path")

    # Model/Backend
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="Ultralytics YOLO model (e.g., yolov8n.pt)")
    parser.add_argument("--device", default="cpu", help='Device hint (e.g., "cpu" or "cuda")')

    # Slicing (SAHI-style)
    parser.add_argument("--slice-w", type=int, default=640, help="Slice width")
    parser.add_argument("--slice-h", type=int, default=640, help="Slice height")
    parser.add_argument("--overlap-w", type=float, default=0.20, help="Overlap width ratio (0..1)")
    parser.add_argument("--overlap-h", type=float, default=0.20, help="Overlap height ratio (0..1)")

    # Thresholds
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")

    # Runtime
    parser.add_argument("--max-frames", type=int, default=-1, help="Process only first N frames (-1 for all)")
    parser.add_argument("--show", action="store_true", help="Show live preview window")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build YOLO backend (detector) and get class names
    infer_fn, class_names = build_yolo_backend(args.yolo_model, device=args.device)

    # Create SAHI-style slicer
    slicer = sv.InferenceSlicer(
        callback=lambda img: infer_fn(img, conf=args.conf),
        slice_wh=(args.slice_w, args.slice_h),
        overlap_ratio_wh=(args.overlap_w, args.overlap_h),
    )

    # Prepare annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_annotator = sv.TraceAnnotator(thickness=2)

    # Tracker for assigning stable IDs across frames
    tracker = sv.ByteTrack()

    # Open input video or webcam
    video_source: Union[str, int]
    if isinstance(args.video, str) and args.video.isdigit():
        video_source = int(args.video)
    else:
        video_source = args.video
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Could not open writer for: {args.out}")

    # Person class index (COCO: 0). If not sure, try to resolve from class_names
    person_class_ids: List[int] = []
    if class_names:
        person_class_ids = [i for i, n in enumerate(class_names) if str(n).lower() == "person"]
    # Fallback to 0 if names unknown
    if not person_class_ids:
        person_class_ids = [0]

    print("\nSAHI Unique Person Counter")
    print(f"Model: {args.yolo_model}")
    print(f"Device: {args.device}")
    print(f"Slicing: slice=({args.slice_w}x{args.slice_h}), overlap=({args.overlap_w},{args.overlap_h})")
    print(f"Reading: {args.video}")
    print(f"Saving:  {args.out}\n")

    max_frames = args.max_frames if args.max_frames > 0 else total_frames
    processed = 0
    seen_track_ids: set = set()

    pbar = tqdm(total=max_frames if max_frames > 0 else total_frames, desc="Processing", unit="frame")
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed += 1
            if args.max_frames > 0 and processed > args.max_frames:
                break

            # SAHI-style detection
            detections: sv.Detections = slicer(frame)

            # Filter to person class only
            if detections.class_id is not None and len(detections) > 0:
                mask = np.isin(detections.class_id, np.array(person_class_ids, dtype=int))
                detections = detections[mask]

            # Track to get stable IDs
            tracked: sv.Detections = tracker.update_with_detections(detections)

            # Update set of seen IDs
            if tracked.tracker_id is not None:
                for tid in tracked.tracker_id:
                    if tid is not None:
                        seen_track_ids.add(int(tid))

            # Build labels: ID and confidence
            labels: List[str] = []
            for i in range(len(tracked)):
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                labels.append(f"ID {tid} {conf:.2f}")

            # Annotate
            annotated = frame.copy()
            annotated = trace_annotator.annotate(annotated, tracked)
            annotated = box_annotator.annotate(scene=annotated, detections=tracked)
            annotated = label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)

            # Draw running unique count
            unique_count = len(seen_track_ids)
            cv2.putText(
                annotated,
                f"Unique persons: {unique_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            out.write(annotated)

            if args.show:
                cv2.imshow("SAHI Unique Persons", annotated)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    break

            pbar.update(1)

    finally:
        cap.release()
        out.release()
        if args.show:
            cv2.destroyAllWindows()
        pbar.close()

    dt = time.time() - t0
    print(f"Done. Frames: {processed} | Time: {dt:.1f}s | FPS (e2e): {processed/max(1e-6, dt):.2f}")
    print(f"Unique persons counted: {len(seen_track_ids)}")
    print(f"Output saved to: {args.out}")


if __name__ == "__main__":
    main()

