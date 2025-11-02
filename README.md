# 🎬 VividReel — AI-Powered Event Video Summarization

> **Transform raw event footage into cinematic highlight reels — automatically.**

VividReel is an AI-driven system that converts unedited event footage into short, cinematic videos. It intelligently selects the best clips based on visual quality and human presence, then assembles them into a polished 5–10 minute highlight reel with transitions and background music.

---

## 📖 Overview

Video editing is often tedious — requiring hours of trimming, sorting, and color correction.
**VividReel** reimagines this workflow using AI and computer vision.

Given multiple short clips from a single event (weddings, birthdays, trips, etc.), the system:

* Detects and extracts the most visually appealing and socially engaging moments.
* Merges them into a cohesive final video.
* Enhances it with transitions, filters, and suitable background music.


> *This repository currently implements the **Round 1 pipeline**.*

---

## 💻 Tech Stack

| Layer               | Technology                             |
| ------------------- | -------------------------------------- |
| **Backend**         | FastAPI                                |
| **Web Server**      | Uvicorn                                |
| **Video Splitting** | PySceneDetect                          |
| **Video Analysis**  | OpenCV (focus, motion, face detection) |
| **Video Assembly**  | MoviePy                                |
| **Language**        | Python 3.9+                            |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd VividReel
```

### 2️⃣ Create and Activate a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

*(You’ll need to create this file; see below.)*

### 4️⃣ Download Face Detection Model

Download the Haar Cascade model for face detection:
➡️ [https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
Place the `.xml` file in your project’s root directory.

### 5️⃣ Add Background Music

* Create a folder named `music` in the project root.
* Add royalty-free `.mp3` tracks (e.g. `happy-uplifting.mp3`, `romantic-wedding.mp3`, `default-music.mp3`).
* Make sure these filenames match the ones defined in `MUSIC_MAP` within `analysis.py`.

---

## ▶️ Usage Guide

### 1️⃣ Start the API Server

```bash
uvicorn main:app --reload
```

> Remove `--reload` if file watcher errors occur (common in synced/cloud folders).

### 2️⃣ Open API Documentation

Navigate to 👉 `http://127.0.0.1:8000/docs`

### 3️⃣ Upload Event Videos

**Endpoint:** `POST /upload/`
Use the Swagger “Try it out” option:

* Enter an `event_type` (e.g., "Wedding", "Birthday", "Travel").
* Upload one or more `.mp4` files.
* Copy the returned `session_id`.

### 4️⃣ Process Videos

**Endpoint:** `POST /process/`
Paste the copied `session_id` and execute.

> Processing time depends on video length and hardware performance.

Upon completion, a `download_url` will be returned.

### 5️⃣ Download Final Video

Use the provided download link:

```
http://127.0.0.1:8000/download/<session-id>/VividReel_Final.mp4
```

Your cinematic highlight reel will download automatically.

---

## 🧠 Pipeline Breakdown

| Step                       | Description                                                                                                 |
| -------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **1. Scene Detection**     | `PySceneDetect` splits all input videos into micro-scenes based on visual changes.                          |
| **2. Scene Analysis**      | `OpenCV` evaluates each scene for focus, motion, and human presence (face detection).                       |
| **3. Scoring & Filtering** | Scenes are scored and ranked; low-quality or empty scenes are discarded.                                    |
| **4. Duration Control**    | Top-ranked clips are selected until the total duration is ~7 minutes.                                       |
| **5. Video Assembly**      | `MoviePy` merges selected scenes, applies transitions, adds background music, and renders the final output. |

---

## 🚀 Future Enhancements (Round 2 Roadmap)

| Focus Area              | Planned Upgrade                                                                               |
| ----------------------- | --------------------------------------------------------------------------------------------- |
| **GPU Acceleration**    | Optimize the pipeline using the NVIDIA A100 GPU.                                              |
| **Scene Understanding** | Integrate pre-trained models for action and emotion recognition (`PyTorchVideo`, `DeepFace`). |
| **Aesthetic Scoring**   | Add deep learning models to rate cinematic shot quality.                                      |
| **Dynamic Editing**     | Adjust pacing and transitions dynamically based on event type and scene mood.                 |
| **AI Music Generation** | Automatically select or generate soundtracks that match video tone and emotion.               |

---

## 📦 Requirements File (`requirements.txt`)

Create a file named `requirements.txt` in the root directory with:

```txt
fastapi
uvicorn[standard]
python-multipart
scenedetect[opencv]
moviepy
opencv-python
numpy<2.3.0
```

---
