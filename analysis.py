import os
import glob
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
import cv2
import numpy as np
import pandas as pd
import joblib
from deepface import DeepFace
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm

# --- Correct imports for moviepy 1.0.3 ---
from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
    AudioFileClip
)
import moviepy.video.fx.all as vfx
import moviepy.audio.fx.all as afx
# --- End correct imports ---

# --- Import PyTorch & Transformers ---
import torch
import decord
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
# --- End PyTorch Imports ---


# --- MODEL CONFIGURATION ---
VIDEOMAE_MODEL_PATH = 'models/videomae_finetuned'
RF_MODEL_PATH = 'models/vividreel_classifier.pkl' # Your RandomForest model
RF_SCALER_PATH = 'models/scaler.pkl'            # Your RandomForest scaler
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
MIN_CONFIDENCE_THRESHOLD = 0.50 # (50%)
# ---------------------------

# Set decord bridge
decord.bridge.set_bridge('torch')

# --- Helper functions for CV2-based filters ---
def _boost_saturation(frame, factor=1.2):
    frame_uint8 = frame.astype(np.uint8)
    hsv = cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def _apply_sepia_tint(frame):
    frame_uint8 = frame.astype(np.uint8)
    sepia_matrix = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])
    sepia_frame = cv2.transform(frame_uint8, sepia_matrix)
    sepia_frame = np.clip(sepia_frame, 0, 255).astype(np.uint8)
    return sepia_frame
# --- End helper functions ---

def split_videos_into_scenes(session_path: str) -> list:
    print(f"Starting scene detection for session: {session_path}")
    scenes_path = os.path.join(session_path, "scenes")
    os.makedirs(scenes_path, exist_ok=True)
    video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv"]
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(session_path, ext)))
        
    if not video_files:
        print("No video files found to process.")
        return []
        
    for video_file in video_files:
        video = None
        try:
            print(f"Processing file: {video_file}")
            video = open_video(video_file)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector())
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list()
            if not scene_list:
                print(f"No scenes detected in {video_file}.")
                continue
            print(f"Detected {len(scene_list)} scenes in {video_file}.")
            split_video_ffmpeg(
                video_file,
                scene_list,
                output_file_template=f"{scenes_path}/$VIDEO_NAME-scene-$SCENE_NUMBER.mp4",
                show_progress=False
            )
        except Exception as e:
            print(f"Error processing file {video_file}: {e}")
        finally:
            if video:
                del video
    all_scene_files = glob.glob(os.path.join(scenes_path, "*.mp4"))
    print(f"Total scenes created: {len(all_scene_files)}")
    return all_scene_files

# --- Feature Extraction Helpers (Used by RF & Heuristics) ---

def _calculate_focus_score(frame, is_rgb=True) -> float:
    """Calculates focus score (Laplacian variance)."""
    if frame is None: return 0
    if is_rgb:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else: # Assumes BGR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def _calculate_motion_score(frame1, frame2, is_rgb=True) -> float:
    """Calculates motion score between two frames."""
    if frame1 is None or frame2 is None: return 0
    if is_rgb:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    else: # Assumes BGR
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return np.mean(thresh)

def analyze_emotions_from_frame(frame, is_rgb=True):
    """Analyzes an RGB or BGR frame for emotion."""
    try:
        frame_to_analyze = frame
        if is_rgb:
            frame_to_analyze = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        analysis = DeepFace.analyze(
            frame_to_analyze, # Pass BGR frame
            actions=['emotion'], 
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )
        return analysis[0]['dominant_emotion']
    except Exception as e:
        return 'none'

def get_frames(full_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Loads first, middle, and last frame (as RGB) from an MP4 file using decord."""
    try:
        vr = decord.VideoReader(full_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        if total_frames < 2: return None
        
        first_frame = vr.get_batch([0]).asnumpy()[0]
        middle_frame = vr.get_batch([total_frames // 2]).asnumpy()[0]
        last_frame = vr.get_batch([total_frames - 1]).asnumpy()[0]
        
        return first_frame, middle_frame, last_frame
    except Exception as e:
        return None

def get_scene_features_for_rf(scene_file: str, face_cascade) -> Optional[np.ndarray]:
    """Extracts the 11-feature vector for the RandomForest model."""
    frames = get_frames(scene_file)
    if frames is None:
        return None
        
    first_frame, middle_frame, last_frame = frames
    
    try:
        focus_score = _calculate_focus_score(middle_frame, is_rgb=True)
        motion_score = _calculate_motion_score(first_frame, last_frame, is_rgb=True)
        
        num_faces = 0
        if middle_frame is not None:
            gray_middle = cv2.cvtColor(middle_frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray_middle, 1.1, 5, minSize=(30, 30))
            num_faces = len(faces)
        
        dominant_emotion = analyze_emotions_from_frame(middle_frame, is_rgb=True)

        features = [
            focus_score,
            motion_score,
            num_faces,
            1 if dominant_emotion == 'happy' else 0,
            1 if dominant_emotion == 'sad' else 0,
            1 if dominant_emotion == 'neutral' else 0,
            1 if dominant_emotion == 'angry' else 0,
            1 if dominant_emotion == 'fear' else 0,
            1 if dominant_emotion == 'disgust' else 0,
            1 if dominant_emotion == 'surprise' else 0,
            1 if dominant_emotion == 'none' else 0,
        ]
        return np.array(features)

    except Exception as e:
        return None

# --- MODEL 1: VideoMAE Prediction Function ---
def predict_scene_scores_transformer(all_scene_files: list) -> dict:
    print(f"Loading fine-tuned VideoMAE model from {VIDEOMAE_MODEL_PATH}...")
    try:
        feature_extractor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_MODEL_PATH)
        model = VideoMAEForVideoClassification.from_pretrained(VIDEOMAE_MODEL_PATH)
    except Exception as e:
         raise FileNotFoundError(f"Error loading VideoMAE model from {VIDEOMAE_MODEL_PATH}: {e}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"VideoMAE model loaded and on device: {device}")

    # !!! IMPORTANT: YOU MUST UPDATE THESE NUMBERS TO MATCH YOUR LABELS !!!
    GOOD_ACTION_IDS = [0, 1, 2, 3, 4, 5, 6] # Assumes 0-6 are 'good' and 7 is 'other'
    
    scored_scenes = {}
    print(f"Analyzing {len(all_scene_files)} scenes with VideoMAE...")
    
    for scene_file in tqdm(all_scene_files, desc="Scoring Scenes (VideoMAE)"):
        try:
            vr = decord.VideoReader(scene_file, ctx=decord.cpu(0))
            total_frames = len(vr)
            if total_frames < 16: continue
            indices = np.linspace(0, total_frames - 1, num=16, dtype=int)
            
            frames_tensor = vr.get_batch(indices) 
            frames_numpy = frames_tensor.numpy() 
            frames_list = [frame for frame in frames_numpy]

            inputs = feature_extractor(frames_list, return_tensors="pt").to(device)
            transformer_score = 0.0
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                
                for good_id in GOOD_ACTION_IDS:
                    if good_id < len(probs):
                        if probs[good_id].item() > transformer_score:
                            transformer_score = probs[good_id].item()

            if transformer_score >= MIN_CONFIDENCE_THRESHOLD:
                scored_scenes[scene_file] = transformer_score
            else:
                scored_scenes[scene_file] = 0.0
        
        except Exception as e:
            print(f"Analysis Complete")
            scored_scenes[scene_file] = 0.0
            
    print(f"VideoMAE analysis complete. Switching to Random Forest...")
    return scored_scenes

# --- MODEL 2: RandomForest Prediction Function (Fallback) ---
def predict_scene_scores_randomforest(all_scene_files: list) -> dict:
    print(f"Loading RandomForest model from {RF_MODEL_PATH}...")
    try:
        model = joblib.load(RF_MODEL_PATH)
        scaler = joblib.load(RF_SCALER_PATH)
    except Exception as e:
         raise FileNotFoundError(f"Error loading RandomForest model/scaler: {e}")
    
    if not os.path.exists(HAAR_CASCADE_PATH):
        raise FileNotFoundError(f"Face detection model not found: {HAAR_CASCADE_PATH}")
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    scored_scenes = {}
    print(f"Analyzing {len(all_scene_files)} scenes with RandomForest...")
    for scene_file in tqdm(all_scene_files, desc="Scoring Scenes (RandomForest)"):
        features = get_scene_features_for_rf(scene_file, face_cascade)
        if features is not None:
            scaled_features = scaler.transform([features])
            all_probs = model.predict_proba(scaled_features)[0]
            
            other_class_id = len(model.classes_) - 1 
            probability_score = 1.0 - all_probs[other_class_id]
            
            if probability_score >= MIN_CONFIDENCE_THRESHOLD:
                scored_scenes[scene_file] = probability_score
            else:
                scored_scenes[scene_file] = 0.0
        else:
            scored_scenes[scene_file] = 0.0
            
    print(f"RandomForest analysis complete. Switching to Heuristic Analysis...")
    return scored_scenes


# --- MODEL 3: Smart Heuristics (Safest Fallback) ---
def predict_scene_scores_heuristics(all_scene_files: list) -> dict:
    print(f"Loading face detection model from {HAAR_CASCADE_PATH}...")
    if not os.path.exists(HAAR_CASCADE_PATH):
        raise FileNotFoundError(f"Face detection model not found: {HAAR_CASCADE_PATH}")
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    FOCUS_THRESHOLD = 100.0 
    MOTION_THRESHOLD = 1.0  
    FACE_BONUS = 100.0      
    
    scored_scenes = {}
    print(f"Analyzing {len(all_scene_files)} with Smart Heuristics...")
    
    for scene_file in tqdm(all_scene_files, desc="Scoring Scenes (Heuristics)"):
        cap = None
        try:
            cap = cv2.VideoCapture(scene_file)
            if not cap.isOpened():
                scored_scenes[scene_file] = 0.0
                continue

            ret, first_frame = cap.read() # BGR frame
            if not ret: continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 2: continue
                
            middle_frame_index = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
            ret, middle_frame = cap.read() # BGR frame
            if not ret: middle_frame = first_frame

            last_frame_index = max(0, total_frames - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_index)
            ret, last_frame = cap.read() # BGR frame
            if not ret: last_frame = first_frame

            focus_score = _calculate_focus_score(middle_frame, is_rgb=False)
            motion_score = _calculate_motion_score(first_frame, last_frame, is_rgb=False)
            
            if focus_score < FOCUS_THRESHOLD or motion_score < MOTION_THRESHOLD:
                scored_scenes[scene_file] = 0.0 
                continue
            
            priority_score = 100.0 
            
            num_faces = 0
            if middle_frame is not None:
                gray_middle = cv2.cvtColor(middle_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_middle, 1.1, 5, minSize=(30, 30))
                num_faces = len(faces)
                priority_score += (num_faces * FACE_BONUS)
            
            # NOTE: We remove DeepFace from the *safest* fallback
            # to guarantee it runs even if TF is broken.
            
            scored_scenes[scene_file] = priority_score

        except Exception as e:
            print(f"Error analyzing scene {scene_file} (Heuristics): {e}")
            scored_scenes[scene_file] = 0.0
        finally:
            if cap:
                cap.release()
                
    print(f"Heuristic analysis complete.")
    return scored_scenes


# --- Duration Helper Function (Unchanged) ---
def get_scene_durations(scene_files: list) -> dict:
    durations = {}
    print(f"Getting durations for {len(scene_files)} scenes...")
    for scene_file in scene_files:
        clip = None
        try:
            clip = VideoFileClip(scene_file)
            durations[scene_file] = clip.duration
        except Exception as e:
            print(f"Warning: Could not get duration for {scene_file}: {e}")
            durations[scene_file] = 0
        finally:
            if clip:
                clip.close()
    return durations

# --- "Mood Engine" Filter Functions ---
MUSIC_DIR = "music"

def apply_energetic_filter(clip):
    # --- FIXED: Use clip.fl_image and clip.fx ---
    clip = clip.fl_image(lambda frame: _boost_saturation(frame, 1.2))
    clip = clip.fx(vfx.lum_contrast, lum=0, contrast=1.5, contrast_thr=128) 
    return clip

def apply_nostalgic_filter(clip):
    # --- FIXED: Use clip.fl_image ---
    return clip.fl_image(_apply_sepia_tint)

def apply_classic_filter(clip):
    # --- FIXED: Use clip.fx ---
    return clip.fx(vfx.blackwhite)

def apply_cinematic_filter(clip):
    # --- FIXED: Use clip.fx ---
    return clip.fx(vfx.lum_contrast, lum=0, contrast=1.5, contrast_thr=128)

def apply_ambient_filter(clip):
    return clip

MOOD_ENGINE = {
    "Energetic": { "filter_func": apply_energetic_filter, "transition_duration": 0.1 },
    "Nostalgic": { "filter_func": apply_nostalgic_filter, "transition_duration": 1.5 },
    "Classic":   { "filter_func": apply_classic_filter,   "transition_duration": 1.5 },
    "Cinematic": { "filter_func": apply_cinematic_filter, "transition_duration": 1.5 },
    "Ambient":   { "filter_func": apply_ambient_filter,   "transition_duration": 2.0 },
    "default":   { "filter_func": apply_ambient_filter,   "transition_duration": 0.5 }
}

MUSIC_MAP = {
    "Wedding": "romantic-wedding.mp3",
    "Birthday": "happy-uplifting.mp3",
    "Travel": "inspiring-travel.mp3",
    "Party": "upbeat-pop.mp3",
    "Other": "default-music.mp3",
    "default": "default-music.mp3"
}

# --- Final Video Assembly Function ---
def create_final_video(
    session_path: str,
    best_scenes_with_scores: list, 
    mood: str,
    event_type: str
) -> str:
    import gc, time  # Added for cleanup safety
    
    print(f"Assembling final video for mood: {mood}, event: {event_type}")
    
    mood_params = MOOD_ENGINE.get(mood, MOOD_ENGINE["default"])
    filter_to_apply = mood_params["filter_func"]
    CROSSFADE_DURATION = mood_params["transition_duration"]
    
    music_file = MUSIC_MAP.get(event_type, MUSIC_MAP["default"])
    music_path = os.path.join(MUSIC_DIR, music_file)
    
    if not os.path.exists(music_path):
        print(f"⚠️ Warning: Music file '{music_file}' not found. Using default.")
        music_path = os.path.join(MUSIC_DIR, MUSIC_MAP["default"])
        if not os.path.exists(music_path):
            raise FileNotFoundError(f"Default music file '{MUSIC_MAP['default']}' not found in 'music' folder.")
    
    processed_clips = []
    
    # --- Process each selected scene ---
    for file_path, score in best_scenes_with_scores:
        clip = None
        try:
            clip = VideoFileClip(file_path)
            clip = clip.without_audio()

            # Apply cinematic slow motion for high scores
            is_high_ml_score = (score > 0.85 and score <= 1.0)
            is_high_heuristic_score = (score > 200)
            
            if mood == "Cinematic" and (is_high_ml_score or is_high_heuristic_score):
                print(f"🎬 Applying slow-mo to {os.path.basename(file_path)} (Score: {score:.2f})")
                clip = clip.fx(vfx.speedx, 0.5)

            # Add transitions
            clip = clip.fx(vfx.fadein, CROSSFADE_DURATION)
            clip = clip.fx(vfx.fadeout, CROSSFADE_DURATION)

            # Apply color/mood filters
            if filter_to_apply:
                clip = filter_to_apply(clip)
            
            processed_clips.append(clip)
        
        except Exception as e:
            print(f"❌ Error processing clip {file_path}: {e}")
            if clip:
                clip.close()

    if not processed_clips:
        raise ValueError("No clips were successfully processed.")
    
    # --- Concatenate the processed clips ---
    final_video = concatenate_videoclips(
        processed_clips, 
        padding=-CROSSFADE_DURATION, 
        method="compose"
    )

    # --- Add background music ---
    audio_clip = None
    try:
        audio_clip = AudioFileClip(music_path)
        final_video_duration = final_video.duration
        
        if final_video_duration > audio_clip.duration:
            audio_clip = audio_clip.fx(afx.audio_loop, duration=final_video_duration)
        else:
            audio_clip = audio_clip.set_duration(final_video_duration)
        
        audio_clip = audio_clip.fx(afx.audio_fadeout, 1.0)
        final_video = final_video.set_audio(audio_clip)
    
    except Exception as e:
        print(f"⚠️ Error adding audio: {e}")
    
    # --- Write the final video (fixed MoviePy 2.x syntax) ---
    output_filename = "VividReel_Final.mp4"
    output_path = os.path.join(session_path, output_filename)
    
    print(f"📝 Writing final video to: {output_path}")
    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac"
        # ⚠️ Removed temp_audiofile_path (deprecated in MoviePy ≥2.0)
    )
    
    # --- Safe cleanup ---
    print("🧹 Cleaning up resources...")
    if audio_clip:
        audio_clip.close()
    final_video.close()
    for clip in processed_clips:
        clip.close()
    
    gc.collect()
    time.sleep(1)  # Prevent PermissionError on Windows file handles
    
    print("✅ Assembly complete.")
    return output_filename
