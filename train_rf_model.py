import os
import glob
import cv2
import numpy as np
import pandas as pd
import decord
from deepface import DeepFace
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, List

# --- CONFIGURATION ---
DATASET_ROOT_DIR = Path("data/labeled_dataset")
TRAINING_DATA_CSV = Path("data/training_data.csv") # INPUT: The CSV you made
OUTPUT_FEATURE_FILE = Path("data/feature_dataset.csv") # OUTPUT: The new CSV this script makes
OUTPUT_MODEL_FILE = Path("models/vividreel_classifier.pkl")
OUTPUT_SCALER_FILE = Path("models/scaler.pkl")
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
# ---------------------

# Set decord to use PyTorch bridge (but we'll get numpy arrays)
decord.bridge.set_bridge('torch')

# --- Feature Extraction Helpers ---

def _calculate_focus_score(frame) -> float:
    """Calculates focus score (Laplacian variance) on an RGB frame."""
    if frame is None: return 0
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Convert RGB to Gray
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def _calculate_motion_score(frame1, frame2) -> float:
    """Calculates motion score between two RGB frames."""
    if frame1 is None or frame2 is None: return 0
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return np.mean(thresh)

def analyze_emotions_from_frame(frame):
    """Analyzes an RGB frame for emotion."""
    try:
        # Convert RGB (from get_frames) to BGR (for DeepFace)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        analysis = DeepFace.analyze(
            frame_bgr, 
            actions=['emotion'], 
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )
        return analysis[0]['dominant_emotion']
    except Exception as e:
        return 'none'

def get_frames(full_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Loads first, middle, and last frame (as RGB) from either an MP4 or a JPG folder."""
    first_frame, middle_frame, last_frame = None, None, None
    
    try:
        if full_path.is_dir(): # It's a folder of JPGs
            frame_paths = sorted(
                glob.glob(os.path.join(full_path, "*.jpg")),
                key=lambda x: int(os.path.basename(x).split('.')[0])
            )
            if len(frame_paths) < 2: return None
            
            first_frame_bgr = cv2.imread(frame_paths[0])
            middle_frame_bgr = cv2.imread(frame_paths[len(frame_paths) // 2])
            last_frame_bgr = cv2.imread(frame_paths[-1])
            
            first_frame = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)
            middle_frame = cv2.cvtColor(middle_frame_bgr, cv2.COLOR_BGR2RGB)
            last_frame = cv2.cvtColor(last_frame_bgr, cv2.COLOR_BGR2RGB)

        elif full_path.is_file(): # It's an .mp4
            vr = decord.VideoReader(str(full_path), ctx=decord.cpu(0))
            total_frames = len(vr)
            if total_frames < 2: return None
            
            first_frame = vr.get_batch([0]).asnumpy()[0]
            middle_frame = vr.get_batch([total_frames // 2]).asnumpy()[0]
            last_frame = vr.get_batch([total_frames - 1]).asnumpy()[0]
        
        return first_frame, middle_frame, last_frame

    except Exception as e:
        print(f"Warning: Failed to load frames from {full_path}: {e}")
        return None

def extract_features():
    """
    Part 1: Reads the CSV, loads all clips, extracts 11 features,
    and saves them to a new CSV.
    """
    print("--- Starting Part 1: Feature Extraction ---")
    if not TRAINING_DATA_CSV.exists():
        print(f"Error: {TRAINING_DATA_CSV} not found. Did you run '2_generate_labels_csv.py'?")
        return

    labels_df = pd.read_csv(TRAINING_DATA_CSV)
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if face_cascade.empty():
        print(f"Error: Could not load Haar Cascade from {HAAR_CASCADE_PATH}")
        return

    all_features = []
    
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Extracting Features"):
        full_path = DATASET_ROOT_DIR / row['clip_path']
        label = row['category_id']
        
        frames = get_frames(full_path)
        if frames is None:
            continue
            
        first_frame, middle_frame, last_frame = frames
        
        focus_score = _calculate_focus_score(middle_frame)
        motion_score = _calculate_motion_score(first_frame, last_frame)
        
        num_faces = 0
        if middle_frame is not None:
            gray_middle = cv2.cvtColor(middle_frame, cv2.COLOR_RGB2GRAY) # Use RGB frame
            faces = face_cascade.detectMultiScale(gray_middle, 1.1, 5, minSize=(30, 30))
            num_faces = len(faces)
        
        dominant_emotion = analyze_emotions_from_frame(middle_frame)
        
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
            label 
        ]
        all_features.append(features)

    feature_names = [
        'focus', 'motion', 'num_faces', 'happy', 'sad', 'neutral', 
        'angry', 'fear', 'disgust', 'surprise', 'none', 'label'
    ]
    feature_df = pd.DataFrame(all_features, columns=feature_names)
    OUTPUT_FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(OUTPUT_FEATURE_FILE, index=False)
    
    print(f"\nFeature extraction complete. Saved {len(feature_df)} feature vectors to {OUTPUT_FEATURE_FILE}")

def train_model():
    """
    Part 2: Loads the features, scales them, trains the RandomForest model,
    and saves the final model and scaler.
    """
    print("\n--- Starting Part 2: Model Training ---")
    if not OUTPUT_FEATURE_FILE.exists():
        print(f"Error: {OUTPUT_FEATURE_FILE} not found. Run feature extraction first.")
        return

    data = pd.read_csv(OUTPUT_FEATURE_FILE)
    data = data.dropna()
    
    X = data.drop('label', axis=1)
    y = data['label']
    
    if len(X) == 0:
        print("Error: No data to train on.")
        return
        
    print(f"Training model on {len(X)} samples...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    class_map_from_csv = pd.read_csv(TRAINING_DATA_CSV).groupby('category_id')['clip_path'].apply(lambda x: x.iloc[0].split('/')[0]).to_dict()
    all_label_ids = sorted(class_map_from_csv.keys())
    class_names = [class_map_from_csv[i] for i in all_label_ids]
    
    print(classification_report(
        y_test, 
        y_pred, 
        labels=all_label_ids,
        target_names=class_names, 
        zero_division=0
    ))
    
    print("\n--- Re-training on all data for final model ---")
    final_scaler = StandardScaler()
    X_scaled_full = final_scaler.fit_transform(X)
    
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    final_model.fit(X_scaled_full, y)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, OUTPUT_MODEL_FILE)
    joblib.dump(final_scaler, OUTPUT_SCALER_FILE)
    
    print(f"\nTraining complete.")
    print(f"Model saved to: {OUTPUT_MODEL_FILE}")
    print(f"Scaler saved to: {OUTPUT_SCALER_FILE}")

if __name__ == "__main__":
    
    # --- THIS IS THE NEW LOGIC ---
    if not OUTPUT_FEATURE_FILE.exists():
        print("Feature dataset not found. Running feature extraction...")
        extract_features()
    else:
        print("--- Skipping Part 1: Feature Extraction (feature_dataset.csv already exists) ---")
    
    # Always run training
    train_model()
    # --- END NEW LOGIC ---