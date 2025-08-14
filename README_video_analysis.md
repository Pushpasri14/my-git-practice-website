# Video Multimodal Analysis (Improved)

This script improves emotion accuracy by face-gating DeepFace with MediaPipe detection, ROI preprocessing (CLAHE), EMA smoothing, and a smile/laughter heuristic based on facial landmarks. It also records an analyzed video and logs periodic summaries with gaze/eye-turn counts.

## Setup

```bash
# If venv creation is not available on this OS image, you can install with:
python3 -m pip install --break-system-packages -r /workspace/requirements.txt
# Otherwise use a venv:
# python -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt
```

If you want the AI-generated summary at the end, set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY=sk-... # from https://openrouter.ai
```

## Run

```bash
python /workspace/video_multimodal_analysis_improved.py
```

- Press SPACE to pause/resume, 'q' to quit.
- Logs in `multimodal_logs/`, screenshots in `eye_turn_screenshots/<username>/`, overlaid analyzed video in the same screenshots folder.

## Tuning for better happiness/smile detection

In `video_multimodal_analysis_improved.py`:
- `shared_data['ema_alpha']` controls smoothing (higher = more responsive, lower = steadier).
- Smile heuristic thresholds are in `compute_smile_metrics`:
  - Width baseline ~1.05, slope 0.40
  - Height baseline ~0.22, slope 0.35
  - To detect smiles more strongly, lower baselines slightly or increase the boost near `smile_c >= 0.55`.

## Notes
- DeepFace and OpenAI are optional. The script will run without them and will use a smile-based heuristic for emotions and skip AI summary.
- If TensorFlow/DeepFace are available, you can uncomment them in `requirements.txt` to enable higher-quality emotion analysis.
- If DeepFace detector errors happen, the script falls back to `mediapipe` backend on full frame and/or the heuristic.
- For dark/noisy videos, CLAHE helps; you can disable it by returning the original ROI in `preprocess_face_roi`.