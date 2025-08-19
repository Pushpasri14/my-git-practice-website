import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import supervision as sv

try:
    from inference import get_model as rf_get_model  # Roboflow Inference SDK
except Exception:
    rf_get_model = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def nms_cv2(xyxy: np.ndarray, scores: np.ndarray, score_thresh: float = 0.05, iou_thresh: float = 0.5):
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


def extract_torso_feature(image_bgr: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    h_img, w_img = image_bgr.shape[:2]
    x1 = max(0, min(w_img - 1, x1))
    x2 = max(0, min(w_img - 1, x2))
    y1 = max(0, min(h_img - 1, y1))
    y2 = max(0, min(h_img - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros(48, dtype=np.float32)

    height = y2 - y1
    torso_top = y1 + int(0.35 * height)
    torso_bottom = y1 + int(0.85 * height)
    torso_bottom = max(torso_top + 4, min(y2, torso_bottom))
    crop = image_bgr[torso_top:torso_bottom, x1:x2]
    if crop.size == 0:
        return np.zeros(48, dtype=np.float32)

    try:
        crop = cv2.resize(crop, (64, 64))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        cv2.normalize(hist_h, hist_h)
        cv2.normalize(hist_s, hist_s)
        cv2.normalize(hist_v, hist_v)
        feat = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()]).astype(np.float32)
        norm = np.linalg.norm(feat) + 1e-8
        return feat / norm
    except Exception:
        return np.zeros(48, dtype=np.float32)


def feature_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    if f1.size == 0 or f2.size == 0:
        return 0.0
    if np.linalg.norm(f1) == 0 or np.linalg.norm(f2) == 0:
        return 0.0
    return float(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unique person counter using Roboflow RF-DETR-nano + SAHI + ByteTrack + identity merging."
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", default="rf_detr_sahi_unique_out.mp4", help="Annotated video output path")
    parser.add_argument("--rf-model-id", default="rf-detr-nano/1", help="Roboflow model id (workspace/model:version)")
    parser.add_argument("--rf-api-key", default=None, help="Roboflow API key (or set ROBOFLOW_API_KEY env)")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--device", default="cpu", help="Device for Ultralytics fallback (cpu or cuda:0)")
    parser.add_argument("--fallback-yolo-model", default="rtdetr-n.pt", help="Ultralytics model for fallback")

    # SAHI slicing defaults tuned for person scale
    parser.add_argument("--slice-w", type=int, default=640, help="Slice width")
    parser.add_argument("--slice-h", type=int, default=640, help="Slice height")
    parser.add_argument("--overlap-w", type=float, default=0.20, help="Overlap width ratio")
    parser.add_argument("--overlap-h", type=float, default=0.20, help="Overlap height ratio")

    parser.add_argument("--nms-iou", type=float, default=0.6, help="Extra NMS IoU threshold")
    parser.add_argument("--max-frames", type=int, default=-1, help="Only process first N frames (-1 for all)")
    parser.add_argument("--show", action="store_true", help="Show visualization window")
    parser.add_argument("--save-report", default="rf_detr_sahi_unique_report.txt", help="Detailed report path")

    # Filtering and identity merge
    parser.add_argument("--min-box-area-frac", type=float, default=0.001, help="Min bbox area fraction of frame")
    parser.add_argument("--min-track-frames", type=int, default=12, help="Min frames a track must exist")
    parser.add_argument("--min-track-conf", type=float, default=0.55, help="Min average confidence per track")
    parser.add_argument("--merge-sim-thresh", type=float, default=0.75, help="Cosine similarity to merge tracks")
    parser.add_argument("--merge-max-gap", type=int, default=60, help="Max frame gap to impose spatial continuity")
    parser.add_argument("--allow-overlap-frames", type=int, default=0, help="Allowed overlapping frames when merging")
    parser.add_argument("--max-spatial-jump-ratio", type=float, default=1.2, help="Max normalized jump between tracks")

    return parser.parse_args()


def build_roboflow_backend(model_id: str, api_key: Optional[str]):
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


def build_ultralytics_backend(model_path: str, device: str = "cpu"):
    if YOLO is None:
        raise RuntimeError("Ultralytics not installed. Run: pip install ultralytics")
    model = YOLO(model_path)

    def infer(image_bgr: np.ndarray, conf: float) -> sv.Detections:
        results = model.predict(image_bgr, conf=conf, verbose=False, device=device)[0]
        dets = sv.Detections.from_ultralytics(results)
        return dets

    return infer


def main():
    args = parse_args()
    backend_name = ""

    # Build Roboflow backend, with fallback to Ultralytics if unauthorized/missing
    infer_fn = None
    try:
        infer_fn = build_roboflow_backend(args.rf_model_id, api_key=args.rf_api_key)
        backend_name = f"Roboflow ({args.rf_model_id})"
    except Exception as e:
        print("[WARN] Falling back to Ultralytics due to Roboflow init error:", str(e))
        infer_fn = build_ultralytics_backend(args.fallback_yolo_model, device=args.device)
        backend_name = f"Ultralytics ({args.fallback_yolo_model})"

    # SAHI slicer around chosen detector
    slicer = sv.InferenceSlicer(
        callback=lambda img: infer_fn(img, conf=args.conf),
        slice_wh=(args.slice_w, args.slice_h),
        overlap_ratio_wh=(args.overlap_w, args.overlap_h),
    )

    # Tracker and stats
    byte_tracker = sv.ByteTrack()
    track_stats: Dict[int, Dict] = {}

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Could not open writer for: {args.out}")

    print("\n*** RF-DETR-nano + SAHI UNIQUE PERSON COUNTER ***")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Total frames: {total_frames}")
    print(f"Backend: {backend_name}")
    print(f"Confidence: {args.conf}")
    print(f"SAHI: slice=({args.slice_w}x{args.slice_h}), overlap=({args.overlap_w},{args.overlap_h})")
    print("Tracker: ByteTrack + identity merging")
    print("=" * 60)

    pbar_total = args.max_frames if args.max_frames > 0 else total_frames
    pbar = tqdm(total=pbar_total, desc="Processing", unit="frame")

    box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color.GREEN)
    label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=1, color=sv.Color.WHITE)

    processed = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            processed += 1
            if args.max_frames > 0 and processed > args.max_frames:
                break

            # Sliced inference (merged across tiles by InferenceSlicer)
            detections: sv.Detections = slicer(frame)

            # Filter to person class (0) if available
            person_idx: List[int] = []
            if len(detections) > 0 and detections.class_id is not None:
                for i in range(len(detections)):
                    if int(detections.class_id[i]) == 0:
                        person_idx.append(i)
            else:
                person_idx = list(range(len(detections)))
            detections = detections[person_idx] if person_idx else sv.Detections.empty()

            # Filter tiny boxes by area
            if len(detections) > 0:
                frame_area = float(width * height)
                min_area = args.min_box_area_frac * frame_area
                keep = []
                for i in range(len(detections)):
                    x1, y1, x2, y2 = detections.xyxy[i]
                    area = float(max(0.0, (x2 - x1))) * float(max(0.0, (y2 - y1)))
                    if area >= min_area:
                        keep.append(i)
                detections = detections[keep] if keep else sv.Detections.empty()

            # Extra NMS safeguard
            if len(detections) > 0 and detections.confidence is not None:
                keep = nms_cv2(detections.xyxy, detections.confidence, score_thresh=0.0, iou_thresh=args.nms_iou)
                detections = detections[keep] if keep else sv.Detections.empty()

            # Track update
            tracked = byte_tracker.update_with_detections(detections)

            # Update stats
            for i in range(len(tracked)):
                tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None
                if tid is None:
                    continue
                if tid not in track_stats:
                    track_stats[tid] = {
                        "frames_seen": set(),
                        "total_detections": 0,
                        "first_seen_frame": processed,
                        "last_seen_frame": processed,
                        "total_confidence": 0.0,
                        "feature": np.zeros(48, dtype=np.float32),
                        "feature_updates": 0,
                        "first_bbox": None,
                        "last_bbox": None,
                    }
                s = track_stats[tid]
                s["frames_seen"].add(processed)
                s["total_detections"] += 1
                s["last_seen_frame"] = processed
                conf_i = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
                s["total_confidence"] += conf_i
                bbox = tracked.xyxy[i]
                if s["first_bbox"] is None:
                    s["first_bbox"] = bbox.copy()
                s["last_bbox"] = bbox.copy()
                feat = extract_torso_feature(frame, bbox)
                if s["feature_updates"] == 0:
                    s["feature"] = feat
                else:
                    alpha = 0.1
                    s["feature"] = (1 - alpha) * s["feature"] + alpha * feat
                s["feature_updates"] += 1

            # Annotate
            annotated = frame.copy()
            if len(tracked) > 0:
                labels = []
                for i in range(len(tracked)):
                    tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                    conf = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
                    labels.append(f"ID {tid} {conf:.2f}")
                annotated = box_annotator.annotate(scene=annotated, detections=tracked)
                annotated = label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)

            out.write(annotated)

            if args.show:
                cv2.imshow("RF-DETR-nano + SAHI Unique Counter", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            pbar.update(1)

    finally:
        cap.release()
        out.release()
        if args.show:
            cv2.destroyAllWindows()
        pbar.close()

    # Build identities from tracks
    def bbox_center_and_diag(bxyxy: np.ndarray) -> Tuple[float, float, float]:
        x1, y1, x2, y2 = map(float, bxyxy)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return cx, cy, max(1e-6, diag)

    qualified: List[Tuple[int, Dict]] = []
    for tid, s in track_stats.items():
        avg_conf = s["total_confidence"] / max(1, s["total_detections"])
        if len(s["frames_seen"]) >= args.min_track_frames and avg_conf >= args.min_track_conf:
            qualified.append((tid, s))

    identities: List[Dict] = []
    for tid, s in sorted(qualified, key=lambda kv: (-kv[1]["total_detections"], kv[1]["first_seen_frame"])):
        assigned = False
        feat = s["feature"]
        first_f = s["first_seen_frame"]
        last_f = s["last_seen_frame"]
        f_bbox = s["first_bbox"] if s["first_bbox"] is not None else np.array([0, 0, 0, 0])
        l_bbox = s["last_bbox"] if s["last_bbox"] is not None else np.array([0, 0, 0, 0])
        for ident in identities:
            overlap = len(s["frames_seen"].intersection(ident["frames_seen"]))
            if overlap > args.allow_overlap_frames:
                continue
            sim = feature_similarity(feat, ident["feature"])
            if sim < args.merge_sim_thresh:
                continue
            gap = 0
            if first_f >= ident["last_frame"]:
                gap = first_f - ident["last_frame"]
                if gap <= args.merge_max_gap and ident["last_bbox"] is not None and s["first_bbox"] is not None:
                    cx1, cy1, d1 = bbox_center_and_diag(ident["last_bbox"]) 
                    cx2, cy2, d2 = bbox_center_and_diag(f_bbox)
                    dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                    if dist / d1 > args.max_spatial_jump_ratio:
                        continue
            elif ident["first_frame"] >= last_f:
                gap = ident["first_frame"] - last_f
                if gap <= args.merge_max_gap and s["last_bbox"] is not None and ident["first_bbox"] is not None:
                    cx1, cy1, d1 = bbox_center_and_diag(l_bbox)
                    cx2, cy2, d2 = bbox_center_and_diag(ident["first_bbox"])
                    dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                    if dist / d1 > args.max_spatial_jump_ratio:
                        continue
            ident["member_track_ids"].add(tid)
            ident["frames_seen"].update(s["frames_seen"])
            ident["total_detections"] += s["total_detections"]
            ident["total_confidence"] += s["total_confidence"]
            ident["feature"] = 0.5 * ident["feature"] + 0.5 * feat
            ident["first_frame"] = min(ident["first_frame"], first_f)
            ident["last_frame"] = max(ident["last_frame"], last_f)
            if ident["first_frame"] == first_f and s["first_bbox"] is not None:
                ident["first_bbox"] = s["first_bbox"]
            if ident["last_frame"] == last_f and s["last_bbox"] is not None:
                ident["last_bbox"] = s["last_bbox"]
            assigned = True
            break
        if not assigned:
            identities.append({
                "member_track_ids": {tid},
                "frames_seen": set(s["frames_seen"]),
                "total_detections": s["total_detections"],
                "total_confidence": s["total_confidence"],
                "feature": s["feature"].copy(),
                "first_frame": first_f,
                "last_frame": last_f,
                "first_bbox": f_bbox,
                "last_bbox": l_bbox,
            })

    unique_count = len(identities)

    dt = time.time() - t0
    with open(args.save_report, "w", encoding="utf-8") as f:
        f.write("RF-DETR-nano + SAHI UNIQUE PERSON COUNT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Video: {args.video}\n")
        f.write(f"Frames processed: {processed}\n")
        f.write(f"Processing time: {dt:.1f}s\n")
        f.write(f"Average FPS: {processed/max(1e-6, dt):.2f}\n\n")
        f.write(f"*** UNIQUE PERSONS: {unique_count} ***\n\n")
        if identities:
            for idx, ident in enumerate(identities, start=1):
                avg_conf = ident["total_confidence"] / max(1, ident["total_detections"])
                f.write(f"Identity {idx}: tracks={sorted(list(ident['member_track_ids']))}, frames={len(ident['frames_seen'])}, dets={ident['total_detections']}, avg_conf={avg_conf:.3f}\n")

    print("\n" + "=" * 70)
    print(f"FINAL UNIQUE PERSONS: {unique_count}")
    print("=" * 70)
    print(f"Annotated video: {args.out}")
    print(f"Report: {args.save_report}")


if __name__ == "__main__":
    main()

