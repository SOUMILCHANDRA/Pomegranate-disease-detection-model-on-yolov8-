# Pomegranate Disease Detection System

This project uses Computer Vision and Deep Learning (YOLOv8) to detect diseases in pomegranate plants.

## Project Structure
- `src/`: Source code for the API server and analysis logic.
- `model/`: Trained YOLOv8 models.
- `data/`: Sample data and database files (mostly ignored by git).

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the API Server:
   ```bash
   cd src
   python api_server.py
   ```
   Or from root:
   ```bash
   python -m src.api_server
   ```

## Usage
- The API runs on port 5000 by default.
- Use `src/main_analyzer.py` for testing individual images.
