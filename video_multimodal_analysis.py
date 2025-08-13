import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages

import cv2
import numpy as np
import pyaudio
import threading
import sys
import time
from datetime import datetime
from deepface import DeepFace
import mediapipe as mp
from tkinter import filedialog
import tkinter as tk
from openai import OpenAI
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Global variables for communication between threads ---
shared_data = {
    'dominant_emotion': "No Face",
    'emotions': {},
    'face_region': None,
    'audio_volume': "No audio detected",
    'hand_status': "Inactive",
    'hand_landmarks': [],
    'eye_turn_count': 0,
    
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
    'is_running': True,  # Flag to control threads
    
    # Enhanced emotion tracking
    'emotion_history': [],
    'emotion_smoothing_window': 15,
    'emotion_confidence_threshold': 0.3,
    'stable_emotion': "No Face",
    'emotion_start_time': {},
    'emotion_duration_threshold': 1.0,
    'confirmed_emotions': {}
}
data_lock = threading.Lock()

# MediaPipe face mesh landmarks for iris tracking
LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Logging and storage configuration
LOG_INTERVAL = 3  # seconds
LOG_FOLDER = "multimodal_logs"
SCREENSHOT_FOLDER = "eye_turn_screenshots"
LOG_FILE = None
current_frame = None

# Video output configuration (for saving analyzed frames with overlays)
VIDEO_WRITER = None
VIDEO_CODEC = 'mp4v'  # mp4v is broadly compatible; consider 'XVID' as alternative
ANALYZED_VIDEO_PATH = None
USER_SCREENSHOT_FOLDER = None
SESSION_TIMESTAMP = None

# LLM Configuration - upgraded model and improved prompting via OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = "openai/gpt-4o"  # Upgraded from gpt-3.5-turbo for higher-quality, more creative output
FALLBACK_LLM_MODEL = "anthropic/claude-3.5-sonnet"  # Optional fallback if the primary model is unavailable

# Fast analysis configuration (optional, does not change default behavior)
FAST_MAX_FRAMES = int(os.getenv("FAST_MAX_FRAMES", "200"))  # total frames sampled across the entire video
FAST_RESIZE_MAX_WIDTH = int(os.getenv("FAST_RESIZE_MAX_WIDTH", "640"))
FAST_GAZE_HOLD_SECONDS = float(os.getenv("FAST_GAZE_HOLD_SECONDS", "0.05"))  # shorter hold for fast counting
FAST_WORKERS = int(os.getenv("FAST_WORKERS", max(1, (os.cpu_count() or 2) - 1)))


def create_directories():
    """Create necessary directories if they don't exist"""
    try:
        # Create main folders
        os.makedirs(LOG_FOLDER, exist_ok=True)
        os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)
        
        # Get current working directory
        current_dir = os.getcwd()
        log_path = os.path.join(current_dir, LOG_FOLDER)
        screenshot_path = os.path.join(current_dir, SCREENSHOT_FOLDER)
        
        print(f"📁 Created directories:")
        print(f"   Logs: {log_path}")
        print(f"   Screenshots: {screenshot_path}")
        
        return True
    except Exception as e:
        print(f"❌ Error creating directories: {e}")
        return False


def generate_offline_summary(log_text: str, user_name: str) -> str:
    """Produce a concise 2–3 sentence summary without using an external LLM.
    Heuristic: analyze primary emotions frequency, gaze direction balance, eye turns, and hand activity.
    """
    try:
        lines = log_text.splitlines()
        primary_emotions: list[str] = []
        gaze_dirs: list[str] = []
        hand_activity: list[str] = []
        emotion_counts: dict[str, int] = {}
        emotion_freq: dict[str, int] = {}
        total_eye_turns = 0

        # Extract key fields
        for ln in lines:
            if ln.startswith("Primary Emotion:"):
                em = ln.split(":", 1)[1].strip()
                em = em.replace(" (detecting...)", "").strip()
                if em and em.lower() != "no face":
                    primary_emotions.append(em)
                    emotion_freq[em] = emotion_freq.get(em, 0) + 1
            elif ln.startswith("Current Gaze Direction:"):
                gd = ln.split(":", 1)[1].strip().lower()
                if gd in {"left", "right", "center"}:
                    gaze_dirs.append(gd)
            elif ln.startswith("Hand Activity:"):
                ha = ln.split(":", 1)[1].strip().lower()
                if ha in {"active", "inactive"}:
                    hand_activity.append(ha)
            elif ln.lower().startswith("total eye turns detected:"):
                try:
                    total_eye_turns = int(re.findall(r"(\d+)", ln)[0])
                except Exception:
                    pass

        # If final total wasn't captured, try last seen Eye Turn Count
        if total_eye_turns == 0:
            for ln in reversed(lines):
                if ln.startswith("Eye Turn Count:"):
                    try:
                        total_eye_turns = int(re.findall(r"(\d+)", ln)[0])
                        break
                    except Exception:
                        continue

        # Determine top emotions
        top_emotion = None
        second_emotion = None
        if emotion_freq:
            sorted_em = sorted(emotion_freq.items(), key=lambda x: x[1], reverse=True)
            top_emotion = sorted_em[0][0]
            if len(sorted_em) > 1:
                second_emotion = sorted_em[1][0]

        distinct_emotions = len(emotion_freq)
        stability = "stable" if distinct_emotions <= 2 else "variable"

        # Gaze distribution
        left_count = sum(1 for g in gaze_dirs if g == "left")
        right_count = sum(1 for g in gaze_dirs if g == "right")
        center_count = sum(1 for g in gaze_dirs if g == "center")
        total_gaze = max(1, len(gaze_dirs))
        dom_gaze = "center" if center_count >= max(left_count, right_count, center_count) else ("left" if left_count >= right_count else "right")

        # Hand activity
        active_count = sum(1 for h in hand_activity if h == "active")
        inactive_count = sum(1 for h in hand_activity if h == "inactive")
        total_hand = max(1, len(hand_activity))
        hands_mostly = "mostly inactive" if inactive_count >= active_count else "mostly active"

        # Build 2–3 sentence summary
        emotion_clause = "neutral overall" if not top_emotion else (top_emotion if not second_emotion else f"{top_emotion}, with moments of {second_emotion}")
        eye_clause = f"{total_eye_turns} sustained eye turns" if total_eye_turns > 0 else "few sustained eye turns"
        gaze_clause = f"attention {dom_gaze}-leaning"

        sentence_1 = f"Emotional tone appeared {stability} and {emotion_clause}."
        sentence_2 = f"Gaze behavior showed {eye_clause} and {gaze_clause}, while hands were {hands_mostly}."
        sentence_3 = "Overall demeanor suggested steady engagement with occasional shifts in affect." if stability == "stable" else "Overall impression was engaged yet dynamic, with noticeable but balanced shifts."

        # Prefer 2 sentences unless variability is high
        summary = f"{sentence_1} {sentence_2}" if distinct_emotions <= 2 else f"{sentence_1} {sentence_2} {sentence_3}"
        return summary
    except Exception:
        # Ultra-safe minimal fallback
        return f"Emotional tone was generally steady. Visual attention remained consistent with moderate eye movement; overall behavior suggested engaged participation."


def generate_llm_summary(log_file_path, user_name):
    """Generate a 2-3 line summary of the analysis using a stronger LLM; fall back to offline heuristic if needed."""
    try:
        print(f"\n🤖 Generating AI summary for {user_name}...")
        
        # Check if log file exists and has content
        if not os.path.exists(log_file_path):
            print(f"❌ Log file not found: {log_file_path}")
            return None
        
        file_size = os.path.getsize(log_file_path)
        if file_size == 0:
            print(f"❌ Log file is empty: {log_file_path}")
            return None
        
        print(f"📄 Reading log file ({file_size} bytes)...")
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # Require a reasonable amount of content to summarize
        if len(log_content.strip()) < 100:
            print("❌ Log content too short for meaningful analysis")
            return None
        
        # Use the most recent portion of the log for relevance
        context_slice = log_content[-4000:]
        
        # If API key missing, skip remote call and do offline summary
        if not OPENROUTER_API_KEY:
            print("⚠️ OPENROUTER_API_KEY not set; generating offline summary.")
            summary = generate_offline_summary(context_slice, user_name)
            used_model = "offline-heuristic"
        else:
            # Initialize OpenAI client with OpenRouter
            print("🔗 Connecting to AI service...")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
            
            # Higher-quality, creative prompt with clear constraints
            system_msg = (
                "You are a senior behavioral analysis expert. Given structured logs of a video session, "
                "write a crisp, human-sounding executive summary (2–3 short sentences). "
                "Synthesize insights; do not restate raw numbers. Focus on primary emotions (patterns and shifts), "
                "attention/eye behavior (turn direction, steadiness, timing), and overall behavioral impression. "
                "Be professional, vivid, and constructive."
            )
            user_prompt = f"""Analyze this video behavioral analysis log and provide a brief 2–3 sentence professional summary.\n\nData for {user_name}:\n{context_slice}\n\nReturn only 2–3 short sentences, no lists or headings."""
            
            def call_model(model_name):
                return client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=180,
                    temperature=0.8,
                    top_p=0.95,
                    presence_penalty=0.3,
                    frequency_penalty=0.1,
                )
            
            print("🤖 Sending request to AI service...")
            try:
                completion = call_model(LLM_MODEL)
                used_model = LLM_MODEL
                summary = completion.choices[0].message.content.strip() if completion else ""
            except Exception as primary_err:
                print(f"⚠️ Primary model failed ({LLM_MODEL}): {primary_err}")
                try:
                    print(f"🔁 Trying fallback model: {FALLBACK_LLM_MODEL}")
                    completion = call_model(FALLBACK_LLM_MODEL)
                    used_model = FALLBACK_LLM_MODEL
                    summary = completion.choices[0].message.content.strip() if completion else ""
                except Exception as fallback_err:
                    print(f"❌ Fallback model failed: {fallback_err}")
                    print("🛈 Falling back to offline heuristic summary.")
                    summary = generate_offline_summary(context_slice, user_name)
                    used_model = "offline-heuristic"
        
        if not summary:
            print("❌ AI returned empty summary; using offline heuristic.")
            summary = generate_offline_summary(context_slice, user_name)
            used_model = "offline-heuristic"
        
        print("✅ Summary generated successfully!")
        
        # Save summary to separate file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in user_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        summary_filename = f"{safe_name}_AI_SUMMARY_{timestamp}.txt"
        summary_file = os.path.join(LOG_FOLDER, summary_filename)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write(f"AI BEHAVIORAL ANALYSIS SUMMARY - {user_name}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {used_model}\n")
            f.write(f"Log file: {os.path.basename(log_file_path)}\n")
            f.write("="*60 + "\n\n")
            f.write(summary)
            f.write(f"\n\n--- End of AI Summary ---")
        
        # Append to original log file
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("🤖 AI GENERATED BEHAVIORAL SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(summary)
            f.write(f"\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            f.write("\n" + "="*80 + "\n")
        
        print(f"✅ AI Summary saved to: {summary_filename}")
        print(f"\n🤖 AI SUMMARY FOR {user_name}:")
        print("-" * 50)
        print(summary)
        print("-" * 50)
        
        return summary, summary_file
        
    except Exception as e:
        print(f"❌ Error generating AI summary: {e}")
        print(f"   Error type: {type(e).__name__}")
        return None


def format_video_time(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_detailed_video_time(seconds):
    """Convert seconds to MM:SS.mmm format with milliseconds"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def smooth_emotions(current_emotions, emotion_history, window_size=15):
    """Smooth emotions over multiple frames to reduce noise"""
    try:
        if not current_emotions:
            return None, {}
        
        emotion_history.append(current_emotions.copy())
        
        if len(emotion_history) > window_size:
            emotion_history = emotion_history[-window_size:]
        
        if len(emotion_history) < 3:
            return None, current_emotions
        
        emotion_sums = {}
        total_weight = 0
        
        for i, emotions in enumerate(emotion_history):
            weight = (i + 1) ** 1.5
            total_weight += weight
            
            for emotion, score in emotions.items():
                if emotion not in emotion_sums:
                    emotion_sums[emotion] = 0
                emotion_sums[emotion] += score * weight
        
        smoothed_emotions = {}
        for emotion, weighted_sum in emotion_sums.items():
            smoothed_emotions[emotion] = weighted_sum / total_weight
        
        if smoothed_emotions:
            max_emotion = max(smoothed_emotions, key=smoothed_emotions.get)
            max_score = smoothed_emotions[max_emotion]
            
            confidence_threshold = 25.0
            second_highest = sorted(smoothed_emotions.values(), reverse=True)[1] if len(smoothed_emotions) > 1 else 0
            
            if max_score >= confidence_threshold and max_score - second_highest >= 10.0:
                return max_emotion, smoothed_emotions
            else:
                return "neutral", smoothed_emotions
        
        return None, {}
        
    except Exception as e:
        print(f"Emotion smoothing error: {e}")
        return None, current_emotions if current_emotions else {}


def track_emotion_duration(emotion, current_time, emotion_start_times, duration_threshold=1.0):
    """Track how long an emotion has been consistently detected"""
    try:
        if emotion == "No Face" or emotion == "neutral":
            return False
        
        if emotion not in emotion_start_times:
            emotion_start_times[emotion] = current_time
            return False
        
        duration = current_time - emotion_start_times[emotion]
        return duration >= duration_threshold
        
    except Exception as e:
        return False


def reset_emotion_tracking(emotion_start_times, except_emotion=None):
    """Reset emotion tracking times for all emotions except the specified one"""
    emotions_to_reset = [e for e in emotion_start_times.keys() if e != except_emotion]
    for emotion in emotions_to_reset:
        del emotion_start_times[emotion]


def select_video_file():
    """Select video file with multiple options"""
    print("\n🎬 VIDEO FILE SELECTION")
    print("="*50)
    print("1️⃣  Type video file path manually")
    print("2️⃣  Use file browser (GUI)")
    
    while True:
        try:
            choice = input("\nChoose option (1 or 2): ").strip()
            
            if choice == "1":
                print("\n📝 MANUAL PATH INPUT")
                print("Examples:")
                print("  Windows: C:\\Users\\YourName\\Videos\\video.mp4")
                print("  Mac/Linux: /home/username/videos/video.mp4")
                print("  Current folder: video.mp4")
                
                video_path = input("\n📁 Enter full path to your video file: ").strip()
                video_path = video_path.strip('"').strip("'")
                
                if video_path and os.path.exists(video_path):
                    return video_path
                else:
                    print(f"❌ File not found: {video_path}")
                    continue
            
            elif choice == "2":
                print("\n🖱 Opening file browser...")
                
                try:
                    root = tk.Tk()
                    root.lift()
                    root.attributes('-topmost', True)
                    root.withdraw()
                    
                    video_path = filedialog.askopenfilename(
                        title="Select Video File",
                        filetypes=[
                            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                            ("MP4 files", "*.mp4"),
                            ("AVI files", "*.avi"),
                            ("MOV files", "*.mov"),
                            ("All files", "*.*")
                        ]
                    )
                    
                    root.destroy()
                    
                    if video_path:
                        return video_path
                    else:
                        print("❌ No file selected.")
                        continue
                        
                except Exception as e:
                    print(f"❌ File browser error: {e}")
                    continue
            
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
                continue
                
        except KeyboardInterrupt:
            print("\n👋 Selection cancelled.")
            return None


def setup_logging(user_name):
    """Create log folder and initialize log file with user's name; prepare analyzed video output path"""
    global LOG_FILE, USER_SCREENSHOT_FOLDER, SESSION_TIMESTAMP, ANALYZED_VIDEO_PATH
    
    # Create directories
    if not create_directories():
        return None
    
    # Create user-specific screenshot folder
    safe_name = "".join(c for c in user_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_name = safe_name.replace(' ', '_')
    user_screenshot_folder = os.path.join(SCREENSHOT_FOLDER, safe_name)
    os.makedirs(user_screenshot_folder, exist_ok=True)
    USER_SCREENSHOT_FOLDER = user_screenshot_folder
    print(f"📁 User screenshot folder: {os.path.join(os.getcwd(), user_screenshot_folder)}")
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    SESSION_TIMESTAMP = timestamp
    log_filename = f"{safe_name}_video_analysis_{timestamp}.txt"
    LOG_FILE = os.path.join(LOG_FOLDER, log_filename)
    
    # Precompute analyzed video path (same folder as screenshots)
    ANALYZED_VIDEO_PATH = os.path.join(user_screenshot_folder, f"{safe_name}_ANALYZED_{timestamp}.mp4")
    
    # Write header to log file
    try:
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
        
        print(f"📁 Log file created: {os.path.abspath(LOG_FILE)}")
        return safe_name
        
    except Exception as e:
        print(f"❌ Error creating log file: {e}")
        return None


def save_screenshot(frame, user_name, gaze_direction, video_timestamp):
    """Save screenshot with timestamps"""
    try:
        system_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        video_time_str = format_video_time(video_timestamp)
        video_time_detailed = format_detailed_video_time(video_timestamp)
        
        safe_name = "".join(c for c in user_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        
        user_screenshot_folder = os.path.join(SCREENSHOT_FOLDER, safe_name)
        screenshot_filename = f"eye_turn_{gaze_direction}_video{video_time_str.replace(':', 'm')}_sys{system_timestamp}.jpg"
        screenshot_path = os.path.join(user_screenshot_folder, screenshot_filename)
        
        # Add timestamp overlay
        frame_with_overlay = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        video_text = f"Video Time: {video_time_str}"
        text_size = cv2.getTextSize(video_text, font, 0.8, 2)[0]
        cv2.rectangle(frame_with_overlay, 
                     (frame.shape[1] - text_size[0] - 20, 10), 
                     (frame.shape[1] - 5, 50), 
                     (0, 0, 0), -1)
        cv2.putText(frame_with_overlay, video_text, 
                   (frame.shape[1] - text_size[0] - 10, 35), 
                   font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        gaze_text = f"Gaze: {gaze_direction.upper()}"
        gaze_size = cv2.getTextSize(gaze_text, font, 0.8, 2)[0]
        cv2.rectangle(frame_with_overlay, 
                     (frame.shape[1] - gaze_size[0] - 20, frame.shape[0] - 50), 
                     (frame.shape[1] - 5, frame.shape[0] - 10), 
                     (0, 0, 0), -1)
        cv2.putText(frame_with_overlay, gaze_text, 
                   (frame.shape[1] - gaze_size[0] - 10, frame.shape[0] - 25), 
                   font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imwrite(screenshot_path, frame_with_overlay)
        
        # Log the screenshot
        if LOG_FILE:
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"📸 SCREENSHOT CAPTURED\n")
                f.write(f"   System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"   Video Time: {video_time_detailed} ({video_time_str})\n")
                f.write(f"   Eye Turn Direction: {gaze_direction.upper()}\n")
                f.write(f"   Screenshot saved: {screenshot_filename}\n")
                f.write("-" * 40 + "\n\n")
        
        print(f"📸 Screenshot saved: {screenshot_filename}")
        return screenshot_path
    except Exception as e:
        print(f"❌ Error saving screenshot: {e}")
        return None


def log_summary(user_name):
    """Log current analysis summary to file every 3 seconds"""
    global LOG_FILE
    
    if LOG_FILE is None:
        print("⚠️ No log file available")
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
                'combined_ratio': shared_data['combined_ratio']
            }
        
        # Write to log file with error handling
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
            f.write("-" * 60 + "\n\n")
        
        print(f"📝 [{timestamp}] {summary['user_name']} @ {summary['video_time_str']} - Logged successfully")
        
    except Exception as e:
        print(f"❌ Error writing to log file: {e}")


def logging_thread(user_name):
    """Thread function to log summary every 3 seconds"""
    print(f"📊 Logging thread started for {user_name}")
    
    # Initial delay
    time.sleep(LOG_INTERVAL)
    
    while shared_data['is_running']:
        try:
            log_summary(user_name)
            time.sleep(LOG_INTERVAL)
        except Exception as e:
            print(f"❌ Logging thread error: {e}")
            time.sleep(LOG_INTERVAL)


def audio_listener():
    """Audio listener disabled for video analysis"""
    with data_lock:
        shared_data['audio_volume'] = "Video playback (no live audio)"
    
    while shared_data['is_running']:
        time.sleep(1)


def get_iris_center(landmarks, iris_indices, frame_width, frame_height):
    """Calculate the center of the iris using MediaPipe iris landmarks"""
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
    except Exception as e:
        return None


def get_eye_center(landmarks, eye_indices, frame_width, frame_height):
    """Calculate the center of the eye using eye contour landmarks"""
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
    except Exception as e:
        return None


def calculate_gaze_ratio(iris_center, eye_center):
    """Calculate gaze ratio based on iris position relative to eye center"""
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
    except Exception as e:
        return 0.5


def video_display(user_name, video_path):
    """Main video display function with analyzed video recording"""
    global current_frame, VIDEO_WRITER, ANALYZED_VIDEO_PATH
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0  # sensible default
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📹 Video loaded: {os.path.basename(video_path)}")
    print(f"📊 Video info: {total_frames} frames, {int(fps)} FPS, {duration/60:.1f} minutes")

    font = cv2.FONT_HERSHEY_SIMPLEX
    white_color = (255, 255, 255)
    cyan_color = (255, 255, 0)
    green_color = (0, 255, 0)
    red_color = (0, 0, 255)
    yellow_color = (0, 255, 255)
    magenta_color = (255, 0, 255)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_skip = 1
    frame_count = 0
    paused = False
    writer_initialized = False

    try:
        while shared_data['is_running']:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("📹 Video ended")
                    break
                
                current_frame = frame.copy()
            
            frame_count += 1
            frame_height, frame_width = frame.shape[:2]
            
            # Initialize video writer after first frame is available (for size)
            if not writer_initialized and ANALYZED_VIDEO_PATH is not None:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
                    VIDEO_WRITER = cv2.VideoWriter(
                        ANALYZED_VIDEO_PATH,
                        fourcc,
                        fps,
                        (frame_width, frame_height)
                    )
                    writer_initialized = True
                    print(f"🎥 Recording analyzed video to: {os.path.abspath(ANALYZED_VIDEO_PATH)}")
                except Exception as e:
                    print(f"❌ Failed to initialize video writer: {e}")
                    writer_initialized = False
            
            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            current_time_sec = current_frame_num / fps if fps > 0 else 0
            
            with data_lock:
                shared_data['video_timestamp'] = current_time_sec
            
            # Emotion detection with smoothing
            if frame_count % frame_skip == 0:
                try:
                    predictions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    current_time = time.time()
                    
                    with data_lock:
                        if predictions and len(predictions) > 0:
                            raw_emotions = predictions[0]['emotion']
                            face_region = predictions[0]['region']
                            
                            smoothed_emotion, smoothed_scores = smooth_emotions(
                                raw_emotions, 
                                shared_data['emotion_history'], 
                                shared_data['emotion_smoothing_window']
                            )
                            
                            if smoothed_emotion and smoothed_scores:
                                is_stable = track_emotion_duration(
                                    smoothed_emotion, 
                                    current_time, 
                                    shared_data['emotion_start_time'], 
                                    shared_data['emotion_duration_threshold']
                                )
                                
                                reset_emotion_tracking(shared_data['emotion_start_time'], smoothed_emotion)
                                
                                shared_data['emotions'] = smoothed_scores
                                shared_data['face_region'] = face_region
                                
                                if is_stable:
                                    shared_data['stable_emotion'] = smoothed_emotion
                                    shared_data['dominant_emotion'] = smoothed_emotion
                                    shared_data['confirmed_emotions'][smoothed_emotion] = current_time
                                else:
                                    shared_data['dominant_emotion'] = f"{smoothed_emotion} (detecting...)"
                            else:
                                shared_data['dominant_emotion'] = "neutral"
                                shared_data['emotions'] = raw_emotions if raw_emotions else {}
                                shared_data['face_region'] = face_region
                                shared_data['emotion_start_time'].clear()
                        else:
                            shared_data['dominant_emotion'] = "No Face"
                            shared_data['stable_emotion'] = "No Face"
                            shared_data['emotions'] = {}
                            shared_data['face_region'] = None
                            shared_data['emotion_start_time'].clear()
                            
                except Exception as e:
                    with data_lock:
                        shared_data['dominant_emotion'] = "No Face"
                        shared_data['emotions'] = {}
                        shared_data['face_region'] = None

            # Hand detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(frame_rgb)
            
            with data_lock:
                if hand_results.multi_hand_landmarks:
                    shared_data['hand_status'] = "Active"
                    shared_data['hand_landmarks'] = hand_results.multi_hand_landmarks
                else:
                    shared_data['hand_status'] = "Inactive"
                    shared_data['hand_landmarks'] = []

            # Eye tracking
            face_results = face_mesh.process(frame_rgb)
            
            with data_lock:
                if face_results.multi_face_landmarks:
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
                            video_time_str = format_video_time(shared_data['video_timestamp'])
                            print(f"🔄 GAZE CHANGED TO: {new_gaze_direction.upper()} at {video_time_str}")
                        
                        if new_gaze_direction in ["left", "right"] and not shared_data['gaze_counted']:
                            time_held = current_time - shared_data['gaze_start_time']
                            
                            if time_held >= shared_data['gaze_hold_duration']:
                                shared_data['eye_turn_count'] += 1
                                shared_data['gaze_counted'] = True
                                
                                if not shared_data['screenshot_taken'] and current_frame is not None:
                                    screenshot_path = save_screenshot(current_frame, user_name, new_gaze_direction, shared_data['video_timestamp'])
                                    shared_data['screenshot_taken'] = True
                                
                                video_time_str = format_video_time(shared_data['video_timestamp'])
                                print(f"✅ EYE TURN COUNTED! Direction: {new_gaze_direction.upper()} at {video_time_str}")
                                print(f"🎯 Total Count: {shared_data['eye_turn_count']}")
                        
                        # Visual debug
                        if left_iris_center is not None:
                            cv2.circle(frame, (int(left_iris_center[0]), int(left_iris_center[1])), 3, (0, 255, 0), -1)
                        if right_iris_center is not None:
                            cv2.circle(frame, (int(right_iris_center[0]), int(right_iris_center[1])), 3, (0, 255, 0), -1)
                        if left_eye_center is not None:
                            cv2.circle(frame, (int(left_eye_center[0]), int(left_eye_center[1])), 2, (255, 0, 0), -1)
                        if right_eye_center is not None:
                            cv2.circle(frame, (int(right_eye_center[0]), int(right_eye_center[1])), 2, (255, 0, 0), -1)
                                
                    except Exception as e:
                        shared_data['current_gaze_direction'] = "center"
                        if frame_count % 60 == 0:
                            print(f"Eye tracking error: {e}")
                else:
                    shared_data['current_gaze_direction'] = "center"

            # Drawing section
            with data_lock:
                if shared_data['face_region']:
                    face_region = shared_data['face_region']
                    x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                if shared_data['hand_landmarks']:
                    for hand_landmarks in shared_data['hand_landmarks']:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                        )

                # Draw text information
                text_y = 30
                video_time_display = format_video_time(shared_data['video_timestamp'])
                total_time_display = format_video_time(duration)
                cv2.putText(frame, f"Video: {os.path.basename(video_path)}", (10, text_y), font, 0.6, cyan_color, 1, cv2.LINE_AA)
                text_y += 25
                cv2.putText(frame, f"Time: {video_time_display} / {total_time_display}", (10, text_y), font, 0.6, cyan_color, 1, cv2.LINE_AA)
                text_y += 35
                
                emotion_display = shared_data['dominant_emotion']
                emotion_color = white_color
                
                if shared_data['dominant_emotion'] == "No Face":
                    emotion_color = (128, 128, 128)
                elif "(detecting...)" in shared_data['dominant_emotion']:
                    emotion_color = yellow_color
                else:
                    stable_emotion = shared_data.get('stable_emotion', 'neutral')
                    if stable_emotion == 'happy':
                        emotion_color = (0, 255, 0)
                    elif stable_emotion == 'sad':
                        emotion_color = (255, 100, 100)
                    elif stable_emotion == 'angry':
                        emotion_color = (0, 0, 255)
                    elif stable_emotion == 'surprise':
                        emotion_color = (255, 255, 0)
                    elif stable_emotion == 'fear':
                        emotion_color = (128, 0, 128)
                    elif stable_emotion == 'disgust':
                        emotion_color = (0, 128, 128)
                
                cv2.putText(frame, f"Emotion: {emotion_display}", (10, text_y), font, 0.8, emotion_color, 2, cv2.LINE_AA)
                text_y += 40
                
                cv2.putText(frame, "Emotion Confidence:", (10, text_y), font, 0.7, white_color, 1, cv2.LINE_AA)
                text_y += 25
                for emotion, score in shared_data['emotions'].items():
                    if score >= 40:
                        conf_color = green_color
                    elif score >= 25:
                        conf_color = yellow_color
                    else:
                        conf_color = (128, 128, 128)
                    
                    bar_width = int(score * 1.5)
                    cv2.rectangle(frame, (200, text_y-10), (200 + bar_width, text_y-5), conf_color, -1)
                    cv2.putText(frame, f"{emotion.capitalize()}: {score:.1f}%", (10, text_y), font, 0.6, conf_color, 1, cv2.LINE_AA)
                    text_y += 20
                
                text_y += 20
                cv2.putText(frame, f"Hand: {shared_data['hand_status']}", (10, text_y), font, 0.8, white_color, 2, cv2.LINE_AA)
                text_y += 30
                
                counter_color = green_color if shared_data['eye_turn_count'] > 0 else white_color
                cv2.putText(frame, f"Eye Turns: {shared_data['eye_turn_count']}", (10, text_y), font, 0.8, counter_color, 2, cv2.LINE_AA)
                text_y += 30
                
                gaze_info = f"Gaze: {shared_data['current_gaze_direction'].upper()}"
                gaze_color = cyan_color
                
                if shared_data['current_gaze_direction'] in ["left", "right"] and shared_data['gaze_start_time'] > 0:
                    time_held = time.time() - shared_data['gaze_start_time']
                    gaze_info += f" ({time_held:.1f}s)"
                    
                    if time_held >= shared_data['gaze_hold_duration']:
                        gaze_color = green_color
                    elif time_held >= shared_data['gaze_hold_duration'] * 0.5:
                        gaze_color = yellow_color
                    else:
                        gaze_color = red_color
                
                cv2.putText(frame, gaze_info, (10, text_y), font, 0.8, gaze_color, 2, cv2.LINE_AA)
                text_y += 30
                
                debug_info = f"L:{shared_data.get('left_iris_x', 0.5):.2f} R:{shared_data.get('right_iris_x', 0.5):.2f} C:{shared_data.get('combined_ratio', 0.5):.2f}"
                cv2.putText(frame, debug_info, (10, text_y), font, 0.6, cyan_color, 1, cv2.LINE_AA)
                text_y += 25
                
                logging_info = f"📁 Logging: {user_name} @ {video_time_display}"
                cv2.putText(frame, logging_info, (10, text_y), font, 0.6, yellow_color, 1, cv2.LINE_AA)
                text_y += 25
                
                # Indicate that analyzed video is being recorded
                if writer_initialized and ANALYZED_VIDEO_PATH:
                    rec_info = f"🎥 Recording analyzed video"
                    cv2.putText(frame, rec_info, (10, text_y), font, 0.6, magenta_color, 1, cv2.LINE_AA)
                    text_y += 25
                
                ai_info = f"🤖 AI Summary: Will generate at end"
                cv2.putText(frame, ai_info, (10, text_y), font, 0.6, (255, 128, 255), 1, cv2.LINE_AA)

            # Write the analyzed frame to the output video
            if writer_initialized and VIDEO_WRITER is not None:
                try:
                    VIDEO_WRITER.write(frame)
                except Exception as e:
                    if frame_count % 60 == 0:
                        print(f"⚠️ Video writer error: {e}")
            
            cv2.imshow('Video Analysis with AI Summary', frame)
            
            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                if paused:
                    print(f"⏸ Video paused at {format_video_time(current_time_sec)}")
                else:
                    print(f"▶ Video resumed from {format_video_time(current_time_sec)}")

    except Exception as e:
        print(f"❌ Video display error: {e}")
    finally:
        cap.release()
        try:
            if VIDEO_WRITER is not None:
                VIDEO_WRITER.release()
                print(f"✅ Saved analyzed video: {os.path.abspath(ANALYZED_VIDEO_PATH) if ANALYZED_VIDEO_PATH else ''}")
        except Exception as e:
            print(f"⚠️ Could not finalize analyzed video: {e}")
        cv2.destroyAllWindows()
        with data_lock:
            shared_data['is_running'] = False

# --- FAST ANALYSIS MODE (optional) ---

def fast_analyze_video(user_name: str, video_path: str):
    """Analyze the video quickly by sampling a limited number of frames across its duration.
    - Skips realtime UI and runs as fast as possible
    - Uses faster detector backend for DeepFace
    - Writes a shorter analyzed video composed of sampled frames
    """
    global current_frame, VIDEO_WRITER, ANALYZED_VIDEO_PATH
    print("⚡ Fast analysis mode enabled (sampling).")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    duration = total_frames / fps if fps > 0 else 0

    # Choose sampled frame indices (uniform across the video)
    num_samples = min(FAST_MAX_FRAMES, total_frames) if total_frames > 0 else 0
    if num_samples <= 0:
        print("❌ No frames to process.")
        cap.release()
        return
    frame_indices = np.linspace(0, max(0, total_frames - 1), num_samples, dtype=int)

    # Read first frame to initialize writer
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_indices[0]))
    ok, first_frame = cap.read()
    if not ok:
        print("❌ Unable to read first frame for fast analysis.")
        cap.release()
        return
    frame_height, frame_width = first_frame.shape[:2]

    # Initialize writer for a shorter analyzed video (sampled frames)
    try:
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        VIDEO_WRITER = cv2.VideoWriter(
            ANALYZED_VIDEO_PATH,
            fourcc,
            max(10.0, min(30.0, fps)),  # reasonable playback rate
            (frame_width, frame_height)
        )
        print(f"🎥 Writing fast analyzed video to: {os.path.abspath(ANALYZED_VIDEO_PATH)}")
    except Exception as e:
        print(f"⚠️ Could not initialize video writer: {e}")
        VIDEO_WRITER = None

    # Prepare detectors (optimized configs)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    mp_drawing = mp.solutions.drawing_utils

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    white_color = (255, 255, 255)
    cyan_color = (255, 255, 0)
    green_color = (0, 255, 0)
    red_color = (0, 0, 255)
    yellow_color = (0, 255, 255)
    magenta_color = (255, 0, 255)

    # Make gaze counting easier in fast mode
    with data_lock:
        shared_data['gaze_hold_duration'] = FAST_GAZE_HOLD_SECONDS

    processed = 0
    log_every = max(1, num_samples // 10)

    try:
        for idx_i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                continue

            current_frame = frame.copy()
            frame_height, frame_width = frame.shape[:2]
            current_time_sec = frame_idx / fps if fps > 0 else 0
            with data_lock:
                shared_data['video_timestamp'] = current_time_sec

            # Optional resize for DeepFace speed
            scale = 1.0
            if frame_width > FAST_RESIZE_MAX_WIDTH:
                scale = FAST_RESIZE_MAX_WIDTH / float(frame_width)
                frame_small = cv2.resize(frame, (int(frame_width * scale), int(frame_height * scale)))
            else:
                frame_small = frame

            # Emotion detection (fast backend)
            try:
                predictions = DeepFace.analyze(
                    img_path = frame_small,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='mediapipe',
                    silent=True
                )
                current_time = time.time()
                with data_lock:
                    if predictions and len(predictions) > 0:
                        raw_emotions = predictions[0]['emotion']
                        region = predictions[0]['region']
                        # Map region to original frame if resized
                        if scale != 1.0 and region is not None:
                            region = {
                                'x': int(region['x'] / scale),
                                'y': int(region['y'] / scale),
                                'w': int(region['w'] / scale),
                                'h': int(region['h'] / scale),
                            }
                        smoothed_emotion, smoothed_scores = smooth_emotions(
                            raw_emotions,
                            shared_data['emotion_history'],
                            shared_data['emotion_smoothing_window']
                        )
                        if smoothed_emotion and smoothed_scores:
                            is_stable = track_emotion_duration(
                                smoothed_emotion,
                                current_time,
                                shared_data['emotion_start_time'],
                                shared_data['emotion_duration_threshold']
                            )
                            reset_emotion_tracking(shared_data['emotion_start_time'], smoothed_emotion)
                            shared_data['emotions'] = smoothed_scores
                            shared_data['face_region'] = region
                            if is_stable:
                                shared_data['stable_emotion'] = smoothed_emotion
                                shared_data['dominant_emotion'] = smoothed_emotion
                                shared_data['confirmed_emotions'][smoothed_emotion] = current_time
                            else:
                                shared_data['dominant_emotion'] = f"{smoothed_emotion} (detecting...)"
                        else:
                            shared_data['dominant_emotion'] = "neutral"
                            shared_data['emotions'] = raw_emotions if raw_emotions else {}
                            shared_data['face_region'] = region
                            shared_data['emotion_start_time'].clear()
                    else:
                        shared_data['dominant_emotion'] = "No Face"
                        shared_data['stable_emotion'] = "No Face"
                        shared_data['emotions'] = {}
                        shared_data['face_region'] = None
                        shared_data['emotion_start_time'].clear()
            except Exception:
                with data_lock:
                    shared_data['dominant_emotion'] = "No Face"
                    shared_data['emotions'] = {}
                    shared_data['face_region'] = None

            # Hand detection (single-frame, static mode)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(frame_rgb)
            with data_lock:
                if hand_results.multi_hand_landmarks:
                    shared_data['hand_status'] = "Active"
                    shared_data['hand_landmarks'] = hand_results.multi_hand_landmarks
                else:
                    shared_data['hand_status'] = "Inactive"
                    shared_data['hand_landmarks'] = []

            # Eye tracking (single-frame)
            face_results = face_mesh.process(frame_rgb)
            with data_lock:
                if face_results.multi_face_landmarks:
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
                                shared_data['gaze_counted'] = True
                                if not shared_data['screenshot_taken'] and current_frame is not None:
                                    _ = save_screenshot(current_frame, user_name, new_gaze_direction, shared_data['video_timestamp'])
                                    shared_data['screenshot_taken'] = True
                    except Exception:
                        shared_data['current_gaze_direction'] = "center"
                else:
                    shared_data['current_gaze_direction'] = "center"

            # Drawing overlays (same as normal mode but no window)
            with data_lock:
                if shared_data['face_region']:
                    fr = shared_data['face_region']
                    cv2.rectangle(frame, (fr['x'], fr['y']), (fr['x'] + fr['w'], fr['y'] + fr['h']), (255, 0, 0), 2)

                if shared_data['hand_landmarks']:
                    for hl in shared_data['hand_landmarks']:
                        mp_drawing.draw_landmarks(
                            frame, hl, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                        )

                text_y = 30
                video_time_display = format_video_time(shared_data['video_timestamp'])
                total_time_display = format_video_time(duration)
                cv2.putText(frame, f"Video: {os.path.basename(video_path)}", (10, text_y), font, 0.6, cyan_color, 1, cv2.LINE_AA)
                text_y += 25
                cv2.putText(frame, f"Time: {video_time_display} / {total_time_display}", (10, text_y), font, 0.6, cyan_color, 1, cv2.LINE_AA)
                text_y += 35
                emotion_display = shared_data['dominant_emotion']
                emotion_color = white_color if "(detecting...)" not in emotion_display else yellow_color
                if emotion_display == "No Face":
                    emotion_color = (128, 128, 128)
                cv2.putText(frame, f"Emotion: {emotion_display}", (10, text_y), font, 0.8, emotion_color, 2, cv2.LINE_AA)
                text_y += 40
                cv2.putText(frame, "Emotion Confidence:", (10, text_y), font, 0.7, white_color, 1, cv2.LINE_AA)
                text_y += 25
                for emotion, score in shared_data['emotions'].items():
                    conf_color = green_color if score >= 40 else (yellow_color if score >= 25 else (128, 128, 128))
                    bar_width = int(score * 1.5)
                    cv2.rectangle(frame, (200, text_y-10), (200 + bar_width, text_y-5), conf_color, -1)
                    cv2.putText(frame, f"{emotion.capitalize()}: {score:.1f}%", (10, text_y), font, 0.6, conf_color, 1, cv2.LINE_AA)
                    text_y += 20
                text_y += 20
                cv2.putText(frame, f"Hand: {shared_data['hand_status']}", (10, text_y), font, 0.8, white_color, 2, cv2.LINE_AA)
                text_y += 30
                counter_color = green_color if shared_data['eye_turn_count'] > 0 else white_color
                cv2.putText(frame, f"Eye Turns: {shared_data['eye_turn_count']}", (10, text_y), font, 0.8, counter_color, 2, cv2.LINE_AA)
                text_y += 30
                gaze_info = f"Gaze: {shared_data['current_gaze_direction'].upper()}"
                cv2.putText(frame, gaze_info, (10, text_y), font, 0.8, cyan_color, 2, cv2.LINE_AA)

            if VIDEO_WRITER is not None:
                try:
                    VIDEO_WRITER.write(frame)
                except Exception:
                    pass

            processed += 1
            if processed % log_every == 0:
                log_summary(user_name)

        print(f"✅ Fast analysis complete. Processed {processed} sampled frames.")
    except Exception as e:
        print(f"❌ Fast analysis error: {e}")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            if VIDEO_WRITER is not None:
                VIDEO_WRITER.release()
        except Exception:
            pass
        with data_lock:
            shared_data['is_running'] = False

# --- ULTRA FAST PARALLEL MODE (optional) ---

def _ultra_worker_pack(frame_bgr, fps, frame_idx, video_path, duration, max_width):
    """Worker-side processing for a single frame. Returns an overlayed JPEG bytes and summary dict."""
    try:
        # Local imports to keep worker lightweight
        import cv2 as _cv
        import numpy as _np
        import time as _time
        import mediapipe as _mp
        from deepface import DeepFace as _DeepFace

        # Optional resize
        fh, fw = frame_bgr.shape[:2]
        scale = 1.0
        if fw > max_width:
            scale = max_width / float(fw)
            frame_proc = _cv.resize(frame_bgr, (int(fw * scale), int(fh * scale)))
        else:
            frame_proc = frame_bgr

        # Init detectors
        mp_hands = _mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
        mp_face_mesh = _mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)
        mp_drawing = _mp.solutions.drawing_utils

        # Emotion
        dominant_emotion = "No Face"
        emotions = {}
        face_region = None
        try:
            preds = _DeepFace.analyze(img_path=frame_proc, actions=['emotion'], enforce_detection=False, detector_backend='mediapipe', silent=True)
            if preds and len(preds) > 0:
                emotions = preds[0]['emotion']
                region = preds[0]['region']
                if region is not None:
                    if scale != 1.0:
                        face_region = {
                            'x': int(region['x'] / scale),
                            'y': int(region['y'] / scale),
                            'w': int(region['w'] / scale),
                            'h': int(region['h'] / scale),
                        }
                    else:
                        face_region = region
                # Pick dominant
                if emotions:
                    dominant_emotion = max(emotions, key=emotions.get)
        except Exception:
            pass

        # Hand detection
        frame_rgb = _cv.cvtColor(frame_bgr, _cv.COLOR_BGR2RGB)
        hand_results = hands.process(frame_rgb)
        hand_status = "Active" if hand_results and hand_results.multi_hand_landmarks else "Inactive"

        # Eye tracking
        gaze_direction = "center"
        left_ratio = 0.5
        right_ratio = 0.5
        combined_ratio = 0.5
        face_results = face_mesh.process(frame_rgb)
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            # Reuse simple center computations
            LEFT_IRIS = [469, 470, 471, 472]
            RIGHT_IRIS = [474, 475, 476, 477]
            LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

            def _center(lms, idxs, w, h):
                pts = []
                for i in idxs:
                    if i < len(lms.landmark):
                        pts.append((_np.clip(lms.landmark[i].x, 0, 1) * w, _np.clip(lms.landmark[i].y, 0, 1) * h))
                if len(pts) < 4:
                    return None
                pts = _np.array(pts)
                return pts.mean(axis=0)

            fh, fw = frame_bgr.shape[:2]
            l_ic = _center(face_landmarks, LEFT_IRIS, fw, fh)
            r_ic = _center(face_landmarks, RIGHT_IRIS, fw, fh)
            l_ec = _center(face_landmarks, LEFT_EYE, fw, fh)
            r_ec = _center(face_landmarks, RIGHT_EYE, fw, fh)

            def _gr(iris, eye):
                if iris is None or eye is None:
                    return 0.5
                disp = (iris[0] - eye[0]) / 25.0
                return max(0.0, min(1.0, 0.5 + disp))

            left_ratio = _gr(l_ic, l_ec)
            right_ratio = _gr(r_ic, r_ec)
            combined_ratio = (left_ratio + right_ratio) / 2.0
            if combined_ratio < 0.40:
                gaze_direction = "left"
            elif combined_ratio > 0.65:
                gaze_direction = "right"
            else:
                gaze_direction = "center"

        # Draw overlays
        font = _cv.FONT_HERSHEY_SIMPLEX
        white = (255, 255, 255)
        cyan = (255, 255, 0)
        green = (0, 255, 0)
        yellow = (0, 255, 255)
        frame_out = frame_bgr.copy()

        # Face box
        if face_region:
            _cv.rectangle(frame_out, (face_region['x'], face_region['y']), (face_region['x'] + face_region['w'], face_region['y'] + face_region['h']), (255, 0, 0), 2)

        # Hands
        if hand_results and hand_results.multi_hand_landmarks:
            for hl in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_out, hl, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

        # Text
        vt = f"Time: {int((frame_idx / fps)//60):02d}:{int((frame_idx / fps)%60):02d} / {int(duration//60):02d}:{int(duration%60):02d}"
        y = 30
        _cv.putText(frame_out, f"Video: {os.path.basename(video_path)}", (10, y), font, 0.6, cyan, 1, _cv.LINE_AA); y += 25
        _cv.putText(frame_out, vt, (10, y), font, 0.6, cyan, 1, _cv.LINE_AA); y += 35
        emo_color = white if dominant_emotion not in ("No Face",) else (128, 128, 128)
        _cv.putText(frame_out, f"Emotion: {dominant_emotion}", (10, y), font, 0.8, emo_color, 2, _cv.LINE_AA); y += 30
        _cv.putText(frame_out, f"Gaze: {gaze_direction.upper()}  L:{left_ratio:.2f} R:{right_ratio:.2f}", (10, y), font, 0.6, cyan, 1, _cv.LINE_AA); y += 25
        _cv.putText(frame_out, f"Hand: {hand_status}", (10, y), font, 0.8, white, 2, _cv.LINE_AA)

        # Encode overlay as JPEG
        ok, enc = _cv.imencode('.jpg', frame_out, [int(_cv.IMWRITE_JPEG_QUALITY), 80])
        jpeg = enc.tobytes() if ok else None

        return {
            'idx': int(frame_idx),
            'dominant_emotion': dominant_emotion,
            'emotions': emotions,
            'hand_status': hand_status,
            'gaze_direction': gaze_direction,
            'left_ratio': float(left_ratio),
            'right_ratio': float(right_ratio),
            'combined_ratio': float(combined_ratio),
            'jpeg': jpeg,
        }
    except Exception as e:
        return {'idx': int(frame_idx), 'error': str(e), 'jpeg': None}


def ultra_fast_analyze_video(user_name: str, video_path: str):
    """Ultra-fast parallel sampling using multiple CPU workers. Produces a short analyzed video from samples."""
    global VIDEO_WRITER, ANALYZED_VIDEO_PATH
    print(f"🚀 Ultra-fast analysis mode: {FAST_WORKERS} workers, up to {FAST_MAX_FRAMES} frames")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    duration = total_frames / fps if fps > 0 else 0

    # Determine frames to sample
    num_samples = min(FAST_MAX_FRAMES, total_frames) if total_frames > 0 else 0
    if num_samples <= 0:
        print("❌ No frames to process.")
        cap.release()
        return
    frame_indices = np.linspace(0, max(0, total_frames - 1), num_samples, dtype=int)

    # Read first frame to initialize writer
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_indices[0]))
    ok, first_frame = cap.read()
    if not ok:
        print("❌ Unable to read first frame for ultra-fast analysis.")
        cap.release()
        return
    fh, fw = first_frame.shape[:2]

    # Initialize video writer for output montage
    try:
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        VIDEO_WRITER = cv2.VideoWriter(ANALYZED_VIDEO_PATH, fourcc, max(15.0, min(30.0, fps)), (fw, fh))
        print(f"🎥 Writing ultra-fast analyzed video to: {os.path.abspath(ANALYZED_VIDEO_PATH)}")
    except Exception as e:
        print(f"⚠️ Could not initialize video writer: {e}")
        VIDEO_WRITER = None

    # Collect frames in memory (as BGR) for workers to avoid re-opening video
    frames_pack = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, fr = cap.read()
        if ret:
            frames_pack.append((fr, idx))
    cap.release()

    # Process in parallel
    results = []
    start = time.time()
    try:
        with ProcessPoolExecutor(max_workers=FAST_WORKERS) as ex:
            futs = [ex.submit(_ultra_worker_pack, fr, fps, int(idx), video_path, duration, FAST_RESIZE_MAX_WIDTH) for fr, idx in frames_pack]
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({'error': str(e)})
    except Exception as e:
        print(f"❌ Parallel execution failed: {e}")
        return

    # Sort by original frame index
    results = [r for r in results if r is not None]
    results.sort(key=lambda r: r.get('idx', 0))

    # Write video and log summaries
    for r in results:
        ts = (r.get('idx', 0) / fps) if fps > 0 else 0.0
        with data_lock:
            shared_data['video_timestamp'] = ts
            shared_data['dominant_emotion'] = r.get('dominant_emotion', 'No Face')
            shared_data['emotions'] = r.get('emotions', {})
            shared_data['hand_status'] = r.get('hand_status', 'Inactive')
            shared_data['current_gaze_direction'] = r.get('gaze_direction', 'center')
            shared_data['left_iris_x'] = r.get('left_ratio', 0.5)
            shared_data['right_iris_x'] = r.get('right_ratio', 0.5)
            shared_data['combined_ratio'] = r.get('combined_ratio', 0.5)
        # periodic logging
        log_summary(user_name)
        # write frame
        if VIDEO_WRITER is not None and r.get('jpeg'):
            try:
                arr = np.frombuffer(r['jpeg'], dtype=np.uint8)
                frame_overlay = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                VIDEO_WRITER.write(frame_overlay)
            except Exception:
                pass

    if VIDEO_WRITER is not None:
        VIDEO_WRITER.release()
    took = time.time() - start
    print(f"✅ Ultra-fast analysis complete in {took:.2f}s for {len(results)} frames.")


def cleanup_and_exit(user_name):
    """Clean up resources and generate AI summary"""
    global LOG_FILE, ANALYZED_VIDEO_PATH
    
    print(f"\n🔄 Cleaning up and generating AI summary for {user_name}...")
    
    # Stop all threads
    with data_lock:
        shared_data['is_running'] = False
    
    # Wait a moment for threads to finish
    time.sleep(1)
    
    if LOG_FILE and os.path.exists(LOG_FILE):
        try:
            safe_name = "".join(c for c in user_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')
            user_screenshot_folder = os.path.join(SCREENSHOT_FOLDER, safe_name)
            
            # Add session end info to log
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*80 + "\n")
                f.write(f"Video analysis session ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"User: {user_name}\n")
                f.write(f"Total eye turns detected: {shared_data.get('eye_turn_count', 0)}\n")
                f.write(f"Screenshots saved in: {user_screenshot_folder}\n")
                if ANALYZED_VIDEO_PATH:
                    f.write(f"Analyzed video saved: {os.path.abspath(ANALYZED_VIDEO_PATH)}\n")
                f.write("="*80 + "\n")
            
            print(f"✅ Session ended. Log file: {os.path.basename(LOG_FILE)}")
            print(f"📸 Screenshots in: {user_screenshot_folder}")
            if ANALYZED_VIDEO_PATH:
                print(f"🎥 Analyzed video: {os.path.abspath(ANALYZED_VIDEO_PATH)}")
            
            # Check log file size before generating AI summary
            file_size = os.path.getsize(LOG_FILE)
            print(f"📄 Log file size: {file_size} bytes")
            
            if file_size > 500:  # Only generate summary if log has sufficient content
                summary_result = generate_llm_summary(LOG_FILE, user_name)
                
                if summary_result:
                    summary_text, summary_file = summary_result
                    print(f"\n🎉 ANALYSIS COMPLETE!")
                    print(f"📄 Main log: {os.path.basename(LOG_FILE)}")
                    print(f"🤖 AI Summary: {os.path.basename(summary_file)}")
                    print(f"📁 All files saved in: {os.path.abspath(LOG_FOLDER)}")
                else:
                    print(f"\n⚠️  Analysis complete but AI summary generation failed")
                    print(f"📄 Manual log available: {os.path.basename(LOG_FILE)}")
            else:
                print(f"⚠️  Log file too small ({file_size} bytes) for meaningful AI summary")
                print(f"📄 Log file available: {os.path.basename(LOG_FILE)}")
                
        except Exception as e:
            print(f"❌ Error in cleanup: {e}")
    else:
        print("❌ No log file found for AI summary generation")


if __name__ == '__main__':
    print("🎯 VIDEO MULTIMODAL ANALYSIS WITH AI SUMMARY")
    print("=" * 60)
    print("🤖 AI-powered behavioral summary using a higher-quality model")
    print("📊 Professional behavioral analysis at session end!")
    
    user_name = None
    while user_name is None:
        try:
            user_input = input("👤 Please enter your name: ").strip()
            if user_input:
                user_name = user_input
                print(f"✅ Hello {user_name}! Starting video analysis...")
            else:
                print("❌ Please enter a valid name.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit()
    
    # Select video file
    video_path = select_video_file()
    
    if not video_path:
        print("❌ No video file selected. Exiting...")
        sys.exit()
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        sys.exit()
    
    print(f"\n📹 Selected video: {os.path.basename(video_path)}")
    print(f"\n🎯 Starting Analysis for {user_name}...")
    print("👁  IRIS-BASED Eye Tracking with VIDEO TIMESTAMPS")
    print("⚡ Look FAR LEFT or FAR RIGHT and hold for 2 seconds")
    print("📸 Screenshots with video timestamps will be saved")
    print("🎥 Analyzed video (with overlays) will be saved in the same folder as screenshots")
    print(f"📝 Logging every {LOG_INTERVAL} seconds to personalized log file")
    print("🤖 AI Summary will be generated at session end")
    print("❌ Press 'q' to quit | ⏸ Press SPACE to pause/resume")
    
    # Setup logging
    safe_name = setup_logging(user_name)
    if not safe_name:
        print("❌ Failed to setup logging. Exiting...")
        sys.exit()

    # Fast mode trigger via CLI or env (optional)
    fast_flag = ('--fast' in sys.argv) or ('-F' in sys.argv) or (os.getenv('FAST_MODE', '').lower() in ('1', 'true', 'yes'))
    ultra_flag = ('--ultrafast' in sys.argv) or (os.getenv('FAST_MODE', '').lower() in ('ultra', 'ultrafast'))
    if ultra_flag:
        ultra_fast_analyze_video(user_name, video_path)
        cleanup_and_exit(user_name)
        sys.exit(0)
    if fast_flag:
        fast_analyze_video(user_name, video_path)
        cleanup_and_exit(user_name)
        sys.exit(0)

    # Start threads (normal realtime mode)
    try:
        # Initialize shared data
        with data_lock:
            shared_data['is_running'] = True
        
        # Start audio thread
        audio_thread = threading.Thread(target=audio_listener)
        audio_thread.daemon = True
        audio_thread.start()
        
        # Start logging thread
        log_thread = threading.Thread(target=logging_thread, args=(user_name,))
        log_thread.daemon = True
        log_thread.start()
        
        # Start video display
        video_display(user_name, video_path)
        
    except KeyboardInterrupt:
        print(f"\n🛑 Interrupted by user")
        with data_lock:
            shared_data['is_running'] = False
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        with data_lock:
            shared_data['is_running'] = False
    finally:
        cleanup_and_exit(user_name)