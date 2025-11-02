# VividReel: AI Video Editor

VividReel is an AI-powered pipeline that automatically edits raw event footage into cinematic highlight reels. It uses a "hybrid-model" approach, combining a deep-learning **VideoMAE Transformer** for content analysis and a **RandomForest** model for quality scoring and fallback.

---

## ⚠️ Critical System Requirements

This project has specific dependencies. Please follow these steps *exactly*.

1.  **Python Version:** You **must** use **Python 3.10.x**. The `deepface` (TensorFlow) dependency will fail on other versions like 3.11, 3.12, or 3.13.

2.  **FFmpeg:** `moviepy` and `scenedetect` require the `ffmpeg` executable. On a new system, this can be installed via a package manager:
    * **On Windows (Admin PowerShell):** `choco install ffmpeg`
    * **On Linux:** `sudo apt-get install ffmpeg`

---

## ⚙️ Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/06hardik/VividReel.git](https://github.com/06hardik/VividReel.git)
    cd VividReel
    ```

2.  **Create Python 3.10 Virtual Environment:**
    *Use the `py -3.10` command (on Windows) or `python3.10` (on Linux) to ensure you are using the correct Python version.*
    ```bash
    # On Windows
    py -3.10 -m venv venv
    
    # On macOS / Linux
    # python3.10 -m venv venv
    ```

3.  **Activate the Environment:**
    ```bash
    # On Windows
    .\venv\Scripts\activate
    
    # On macOS / Linux
    # source venv/bin/activate
    ```

4.  **Install All Dependencies:**
    *Install from the provided `requirements.txt` file.*
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 How to Run the Demonstration

We have provided a single command-line script, `run_demo.py`, to execute the entire pipeline.

1.  **Open your terminal** and ensure your `(venv)` is activated.
2.  **Run the script** by providing three arguments:
    * `input_folder`: The path to the judges' local folder of test videos.
    * `--mood`: The desired creative style.
    * `--event`: The event theme for music selection.

**Example Command:**
```bash
python run_demo.py "D:\Hackathon\Test_Dataset" --mood "Cinematic" --event "Wedding"