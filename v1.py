import os
import subprocess
import json
import tempfile
from pathlib import Path

import whisper
import cv2
import librosa
import numpy as np
from fer import FER
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
import faiss
import requests

# --- Configuration ---
VIDEO_PATH = "input.mp4"
FPS = 2  # frames per second
FRAME_DIR = Path("frames")
AUDIO_PATH = Path("audio.wav")
TRANSCRIPT_PATH = Path("transcript.json")
RICH_TRANSCRIPT_PATH = Path("rich_transcript.json")

# Local Llama3 API endpoint
LLAMA_API_URL = "http://localhost:8000/v1/chat/completions"
LLAMA_MODEL = "llama3"

# --- Step 1: Extract frames ---
def extract_frames(video_path: str, out_dir: Path, fps: int = 2):
    out_dir.mkdir(exist_ok=True)
    # ffmpeg command: extract frames at given fps
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        str(out_dir / "frame_%05d.jpg")
    ]
    subprocess.run(cmd, check=True)

# --- Step 2: Extract audio ---
def extract_audio(video_path: str, out_audio: Path):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", str(out_audio)
    ]
    subprocess.run(cmd, check=True)

# --- Step 3: Transcribe with Whisper ---
def transcribe_audio(audio_path: Path):
    model = whisper.load_model("base")
    result = model.transcribe(str(audio_path), word_timestamps=True)
    # Save raw transcript
    with open(TRANSCRIPT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    return result["segments"]  # list of {start, end, text}

# --- Step 4: Video emotion detection ---
def detect_video_emotions(frames_dir: Path, fps: int = 2):
    detector = FER(mtcnn=True)
    emotions = {}
    frame_files = sorted(frames_dir.glob("*.jpg"))
    for idx, frame_file in enumerate(frame_files):
        timestamp = idx / fps
        img = cv2.imread(str(frame_file))[:, :, ::-1]  # BGR to RGB
        emo, score = detector.top_emotion(img)
        emotions[round(timestamp, 1)] = {"face": emo, "confidence": score}
    return emotions

# --- Step 5: Audio emotion detection ---
def detect_audio_emotions(audio_path: Path, fps: int = 2):
    # Load audio
    y, sr = librosa.load(str(audio_path), sr=16000)
    hop_length = int(sr / fps)
    audio_emotions = {}
    # Use HF pipeline for emotion
    emo_pipe = hf_pipeline("audio-classification", model="j-hartmann/emotion-english-distilroberta-base")
    for i in range(0, len(y), hop_length):
        chunk = y[i:i + hop_length]
        if len(chunk) < hop_length:
            break
        timestamp = round(i / sr, 1)
        # Save to temp
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            librosa.output.write_wav(tmp.name, chunk, sr)
            preds = emo_pipe(tmp.name)
            os.unlink(tmp.name)
        # pick top
        top = max(preds, key=lambda x: x['score'])
        audio_emotions[timestamp] = {"voice": top['label'], "confidence": top['score']}
    return audio_emotions

# --- Step 6: Build rich transcript ---
def build_rich_transcript(segments, vid_emos, aud_emos):
    rich = []
    for seg in segments:
        start = round(seg['start'], 1)
        text = seg['text'].strip()
        face_emo = vid_emos.get(start, {})
        voice_emo = aud_emos.get(start, {})
        entry = {
            'timestamp': start,
            'text': text,
            'face_emotion': face_emo.get('face'),
            'face_confidence': face_emo.get('confidence'),
            'voice_emotion': voice_emo.get('voice'),
            'voice_confidence': voice_emo.get('confidence')
        }
        rich.append(entry)
    with open(RICH_TRANSCRIPT_PATH, 'w') as f:
        json.dump(rich, f, indent=2)
    return rich

# --- Step 7: Indexing for retrieval ---
def build_index(chunks, embed_model_name="all-MiniLM-L6-v2", index_path="index.faiss"):
    embedder = SentenceTransformer(embed_model_name)
    texts = [json.dumps(chunk) for chunk in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    # save texts for retrieval
    with open("chunks.json", 'w') as f:
        json.dump(texts, f)
    return index

# --- Step 8: Chat interface ---
def chat_with_llama(question, k=3):
    # Load index
    index = faiss.read_index("index.faiss")
    with open("chunks.json") as f:
        texts = json.load(f)
    # Embed question
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = embedder.encode([question])
    D, I = index.search(q_emb, k)
    # Build prompt
    context = "".join([texts[i] for i in I[0]])
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are an assistant aware of video transcript and emotions."},
            {"role": "user", "content": context},
            {"role": "user", "content": question}
        ]
    }
    resp = requests.post(LLAMA_API_URL, json=payload).json()
    return resp['choices'][0]['message']['content']

# --- Main ---
if __name__ == "__main__":
    # Step 1 & 2
    extract_frames(VIDEO_PATH, FRAME_DIR, FPS)
    extract_audio(VIDEO_PATH, AUDIO_PATH)

    # Step 3
    segments = transcribe_audio(AUDIO_PATH)

    # Step 4 & 5
    vid_emos = detect_video_emotions(FRAME_DIR, FPS)
    aud_emos = detect_audio_emotions(AUDIO_PATH, FPS)

    # Step 6
    chunks = build_rich_transcript(segments, vid_emos, aud_emos)

    # Step 7
    build_index(chunks)

    # Interactive chat
    print("Emotion-aware pipeline ready. Ask questions about the video.")
    while True:
        q = input("You: ")
        if q.lower() in {"exit", "quit"}:
            break
        ans = chat_with_llama(q)
        print(f"Assistant: {ans}")
