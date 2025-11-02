import os
import glob
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from deepface import DeepFace
from typing import Tuple, Optional, Dict, List

# --- Correct imports for moviepy 2.x ---
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

# --- Feature Extraction Helpers (Used by BOTH models) ---

def _calculate_focus_score(frame) -> float:
    if frame is None:
        return 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def _calculate_motion_score(frame1, frame2) -> float:
    if frame1 is None or frame2 is None:
        return 0
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_pct = np.mean(thresh)
    return motion_pct

def analyze_emotions_from_frame(frame):
    try:
        analysis = DeepFace.analyze(
            frame, 
            actions=['emotion'], 
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )
        dominant_emotion = analysis[0]['dominant_emotion']
        return True, dominant_emotion
    except Exception as e:
        return False, 'none'

def get_scene_features_for_rf(scene_file: str, face_cascade) -> Optional[np.ndarray]:
    """Extracts the 11-feature vector for the RandomForest model."""
    cap = None
    try:
        cap = cv2.VideoCapture(scene_file)
        if not cap.isOpened(): return None
        ret, first_frame = cap.read()
        if not ret: return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 2: return None
        middle_frame_index = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
        ret, middle_frame = cap.read()
        if not ret: middle_frame = first_frame
        last_frame_index = max(0, total_frames - 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_index)
        ret, last_frame = cap.read()
        if not ret: last_frame = first_frame

        focus_score = _calculate_focus_score(middle_frame)
        motion_score = _calculate_motion_score(first_frame, last_frame)
        
        num_faces = 0
        if middle_frame is not None:
            gray_middle = cv2.cvtColor(middle_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_middle, 1.1, 5, minSize=(30, 30))
            num_faces = len(faces)
        
        has_emotion, dominant_emotion = analyze_emotions_from_frame(middle_frame)

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
        print(f"Error extracting features for {scene_file}: {e}")
        return None
    finally:
        if cap:
            cap.release()

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

    # --- Define "Good" Action IDs ---
    # This must match the classes you trained on (from videodataset.py / 2_generate_labels_csv.py)
    # Check your 'test_loader.py' output for the 'Class mapping'
    # EXAMPLE: {'cheering_or_clapping': 0, 'dancing': 1, 'hugging_or_emotional': 2, 'other': 3, ...}
    
    # !!! IMPORTANT: YOU MUST UPDATE THESE NUMBERS TO MATCH YOUR LABELS !!!
    GOOD_ACTION_IDS = [0, 1, 2, 3, 4, 5, 6] # Assumes 0-6 are your "good" actions and 7 is 'other'
    
    scored_scenes = {}
    print(f"Analyzing {len(all_scene_files)} scenes with VideoMAE...")
    
    for scene_file in tqdm(all_scene_files, desc="Scoring Scenes (VideoMAE)"):
        try:
            vr = decord.VideoReader(scene_file, ctx=decord.cpu(0))
            total_frames = len(vr)
            if total_frames < 16: continue
            indices = np.linspace(0, total_frames - 1, num=16, dtype=int)
            frames = vr.get_batch(indices)
            frames_list = [frame.numpy() for frame in frames]

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
            print(f"Error analyzing scene {scene_file}: {e}")
            scored_scenes[scene_file] = 0.0
            
    print(f"VideoMAE analysis complete.")
    return scored_scenes

# --- MODEL 2: RandomForest Prediction Function (Fallback) ---
def predict_scene_scores_randomforest(all_scene_files: list) -> dict:
    """
    Analyzes scenes using the PRE-TRAINED RandomForest MODEL to get scores.
    This is the fallback model.
    """
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
            probability_score = model.predict_proba(scaled_features)[0][1] # Assumes 1 is "good"
            if probability_score >= MIN_CONFIDENCE_THRESHOLD:
                scored_scenes[scene_file] = probability_score
            else:
                scored_scenes[scene_file] = 0.0
        else:
            scored_scenes[scene_file] = 0.0
            
    print(f"RandomForest analysis complete.")
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

# --- "Mood Engine" Filter Functions (Unchanged) ---
MUSIC_DIR = "music"

def apply_energetic_filter(clip):
    clip = clip.fl(lambda gf, t: _boost_saturation(gf(t), 1.2))
    effect = vfx.LumContrast(lum=0, contrast=1.5, contrast_thr=128) 
    return clip.with_effects([effect])

def apply_nostalgic_filter(clip):
    return clip.fl(lambda gf, t: _apply_sepia_tint(gf(t)))

def apply_classic_filter(clip):
    return vfx.BlackWhite(clip)

def apply_cinematic_filter(clip):
    effect = vfx.LumContrast(lum=0, contrast=1.5, contrast_thr=128)
    return clip.with_effects([effect])

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

# --- Final Video Assembly Function (Unchanged) ---
def create_final_video(
    session_path: str,
    best_scenes_with_scores: list, 
    mood: str,
    event_type: str
) -> str:
    
    print(f"Assembling final video for mood: {mood}, event: {event_type}")
    
    mood_params = MOOD_ENGINE.get(mood, MOOD_ENGINE["default"])
    filter_to_apply = mood_params["filter_func"]
    CROSSFADE_DURATION = mood_params["transition_duration"]
    
    music_file = MUSIC_MAP.get(event_type, MUSIC_MAP["default"])
    
    music_path = os.path.join(MUSIC_DIR, music_file)
    if not os.path.exists(music_path):
        print(f"Warning: Music file '{music_file}' not found. Using default.")
        music_path = os.path.join(MUSIC_DIR, MUSIC_MAP["default"])
        if not os.path.exists(music_path):
             raise FileNotFoundError(f"Default music file '{MUSIC_MAP['default']}' not found in 'music' folder.")
    
    processed_clips = []
    
    for file_path, score in best_scenes_with_scores:
        clip = None
        try:
            clip = VideoFileClip(file_path)
            clip = clip.without_audio()

            if mood == "Cinematic" and score > 0.85:
                print(f"Applying slow-mo to {os.path.basename(file_path)} (Score: {score:.2f})")
                clip = vfx.MultiplySpeed(clip, 0.5) 
            
            fade_in_effect = vfx.FadeIn(CROSSFADE_DURATION)
            clip = clip.with_effects([fade_in_effect])
            fade_out_effect = vfx.FadeOut(CROSSFADE_DURATION)
            clip = clip.with_effects([fade_out_effect])

            if filter_to_apply:
                clip = filter_to_apply(clip)
            
            processed_clips.append(clip)
            
        except Exception as e:
            print(f"Error processing clip {file_path}: {e}")
            if clip:
                clip.close()

    if not processed_clips:
        raise ValueError("No clips were successfully processed.")

    final_video = concatenate_videoclips(
        processed_clips, 
        padding=-CROSSFADE_DURATION, 
        method="compose"
    )

    audio_clip = None
    try:
        audio_clip = AudioFileClip(music_path)
        final_video_duration = final_video.duration 
        if final_video_duration > audio_clip.duration:
             loop_effect = afx.AudioLoop(duration=final_video_duration)
             audio_clip = audio_clip.with_effects([loop_effect])
        else:
            audio_clip = audio_clip.with_duration(final_video_duration)
        fade_out_audio_effect = afx.AudioFadeOut(1.0)
        audio_clip = audio_clip.with_effects([fade_out_audio_effect])
        final_video = final_video.with_audio(audio_clip)
    except Exception as e:
        print(f"Error adding audio: {e}")

    output_filename = "VividReel_Final.mp4"
    output_path = os.path.join(session_path, output_filename)
    
    print(f"Writing final video to: {output_path}")
    final_video.write_videofile(
        output_path, 
        codec="libx264", 
        audio_codec="aac",
        temp_audiofile_path=session_path
    )
    
    print("Cleaning up resources...")
    if audio_clip:
        audio_clip.close()
    final_video.close()
    for clip in processed_clips:
        clip.close()
        
    print("Assembly complete.")
    return output_filename