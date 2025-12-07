# Vehicle Calculator for Civil Engg

This is a ready-to-deploy Streamlit web app that detects and counts vehicles by type from a recorded video.

## Files
- detect_and_count.py  -> CLI script using YOLOv8 + tracker to detect and count vehicles and produce an annotated video.
- streamlit_app.py    -> Streamlit UI to upload video and run the detector.
- requirements.txt    -> Python dependencies

## Quick local run (Windows/Linux)
1. Create a Python virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate  # mac/linux
   venv\Scripts\activate   # windows

2. Install dependencies:
   pip install -r requirements.txt

3. Run Streamlit:
   streamlit run streamlit_app.py

## Deploy to Streamlit Cloud (no coding once files are in GitHub)
1. Create a free GitHub account (if you don't have one): https://github.com
2. Create a new repository and upload all files from this project.
3. Go to https://streamlit.io/cloud and sign in with GitHub.
4. Click 'New app', pick the repository and branch, and set the main file to `streamlit_app.py`.
5. Click 'Deploy'. The app will open and give you a shareable link.

## Notes
- The first run may auto-download YOLOv8 weights (yolov8n.pt).
- If you prefer, you can ask me to produce a Dockerfile or Flask API wrapper next.
