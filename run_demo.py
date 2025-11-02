import os
import glob
import uuid
import shutil
import argparse
import analysis # Your logic file
from tqdm import tqdm

# --- CONFIGURATION ---
TARGET_DURATION_SECONDS = 420 # 7 minutes
OUTPUT_DIR = "output"
MINIMUM_SCENE_COUNT = 10 # Min number of clips to find before falling back
# ---------------------

def main(input_folder, mood, event_type):
    """
    Main processing pipeline with fallback logic.
    """
    print(f"--- VividReel AI Pipeline Started ---")
    print(f"Input Folder: {input_folder}")
    print(f"Mood: {mood}, Event: {event_type}")

    session_id = str(uuid.uuid4())
    session_path = os.path.join("uploads", session_id)
    os.makedirs(session_path, exist_ok=True)
    
    print(f"Created temp session: {session_path}")

    print(f"Copying files from {input_folder} to {session_path}...")
    video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv"]
    video_files_to_copy = []
    for ext in video_extensions:
        video_files_to_copy.extend(glob.glob(os.path.join(input_folder, ext)))
        
    if not video_files_to_copy:
        print(f"Error: No video files found in {input_folder}")
        return

    for f in tqdm(video_files_to_copy, desc="Copying videos"):
        shutil.copy(f, session_path)

    try:
        # Step 1: Split scenes
        print("Starting scene detection...")
        all_scene_files = analysis.split_videos_into_scenes(session_path)
        if not all_scene_files:
             raise Exception("No scenes were found or created.")

        # --- NEW HYBRID LOGIC ---
        
        # Step 2: Try to score with VideoMAE Transformer
        print("--- Running Primary Model (VideoMAE) ---")
        scene_scores = analysis.predict_scene_scores_transformer(all_scene_files)
        
        # Filter out scores below threshold (0)
        good_scenes_tuples = sorted(
            [item for item in scene_scores.items() if item[1] > 0], 
            key=lambda item: item[1], 
            reverse=True
        )

        # Step 3: Check if VideoMAE failed (found too few clips)
        if len(good_scenes_tuples) < MINIMUM_SCENE_COUNT:
            print(f"WARNING: VideoMAE found only {len(good_scenes_tuples)} clips.")
            print("--- Falling Back to RandomForest Model ---")
            
            # Step 2 (Fallback): Score with RandomForest
            scene_scores = analysis.predict_scene_scores_randomforest(all_scene_files)
            
            good_scenes_tuples = sorted(
                [item for item in scene_scores.items() if item[1] > 0], 
                key=lambda item: item[1], 
                reverse=True
            )
            
            if len(good_scenes_tuples) < MINIMUM_SCENE_COUNT:
                 print("WARNING: Fallback model also found few clips. Proceeding anyway.")

        sorted_scenes = good_scenes_tuples # This is now our list of (filepath, score)
        
        # --- END HYBRID LOGIC ---

        # Step 4: Limit by duration
        good_scenes_to_check = [f for f, s in sorted_scenes]
        scene_durations = analysis.get_scene_durations(good_scenes_to_check)
        
        final_scene_list_paths = []
        current_duration = 0
        
        for scene_file, score in sorted_scenes:
            duration = scene_durations.get(scene_file, 0)
            
            if mood == "Cinematic" and score > 0.85:
                duration *= 2

            if duration > 0 and (current_duration + duration) <= TARGET_DURATION_SECONDS:
                final_scene_list_paths.append((scene_file, score)) 
                current_duration += duration
            elif current_duration >= TARGET_DURATION_SECONDS:
                break

        if not final_scene_list_paths:
             raise Exception("No scenes fit within the target duration based on scores.")
        print(f"Selected {len(final_scene_list_paths)} scenes totaling {current_duration:.2f} seconds.")

        # Step 5: Assemble final video
        print("Assembling final video...")
        final_filename = analysis.create_final_video(
            session_path,
            final_scene_list_paths,
            mood,
            event_type
        )
        
        # Step 6: Save final video
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        final_output_path = os.path.join(OUTPUT_DIR, "VividReel_Final.mp4")
        shutil.move(os.path.join(session_path, final_filename), final_output_path)
        
        print(f"--- SUCCESS ---")
        print(f"Final video saved to: {final_output_path}")
        
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 9. Clean up temp folder
        print(f"Cleaning up temp session folder: {session_path}")
        if os.path.exists(session_path):
            shutil.rmtree(session_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VividReel AI Video Editor")
    
    parser.add_argument("input_folder", type=str, help="Path to the local folder containing raw video clips.")
    
    parser.add_argument("--mood", type=str, required=True, 
                        choices=["Energetic", "Nostalgic", "Classic", "Cinematic", "Ambient"],
                        help="The creative mood to apply.")
                        
    parser.add_argument("--event", type=str, required=True,
                        choices=["Wedding", "Birthday", "Travel", "Party", "Other"],
                        help="The event type for music selection.")
                        
    args = parser.parse_args()
    
    main(args.input_folder, args.mood, args.event)