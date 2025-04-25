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
import gradio as gr

# --- Configuration ---
FPS = 2  # frames per second
EMBED_MODEL = "all-MiniLM-L6-v2"
LLAMA_API_URL = "http://localhost:8000/v1/chat/completions"
LLAMA_MODEL = "llama3"

# Temporary directories
BASE_DIR = Path(tempfile.gettempdir()) / "emotion_aware"
FRAME_DIR = BASE_DIR / "frames"
AUDIO_PATH = BASE_DIR / "audio.wav"
TRANSCRIPT_PATH = BASE_DIR / "transcript.json"
RICH_TRANSCRIPT_PATH = BASE_DIR / "rich_transcript.json"
INDEX_PATH = BASE_DIR / "index.faiss"
CHUNKS_PATH = BASE_DIR / "chunks.json"

# Ensure base dir exists
BASE_DIR.mkdir(parents=True, exist_ok=True)

# --- Step Functions ---
def extract_frames(video_path: str, out_dir: Path, fps: int = FPS):
    out_dir.mkdir(exist_ok=True, parents=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        str(out_dir / "frame_%05d.jpg")
    ]
    subprocess.run(cmd, check=True)


def extract_audio(video_path: str, out_audio: Path):
    out_audio.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", str(out_audio)
    ]
    subprocess.run(cmd, check=True)


def transcribe_audio(audio_path: Path):
    model = whisper.load_model("base")
    result = model.transcribe(str(audio_path), word_timestamps=True)
    with open(TRANSCRIPT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    return result['segments']


def detect_video_emotions(frames_dir: Path, fps: int = FPS):
    detector = FER(mtcnn=True)
    emotions = {}
    frame_files = sorted(frames_dir.glob("*.jpg"))
    for idx, frame_file in enumerate(frame_files):
        timestamp = round(idx / fps, 1)
        img = cv2.imread(str(frame_file))[:, :, ::-1]
        emo, score = detector.top_emotion(img)
        emotions[timestamp] = {"face": emo, "confidence": score}
    return emotions


def detect_audio_emotions(audio_path: Path, fps: int = FPS):
    y, sr = librosa.load(str(audio_path), sr=16000)
    hop_length = int(sr / fps)
    audio_emotions = {}
    emo_pipe = hf_pipeline("audio-classification", model="j-hartmann/emotion-english-distilroberta-base")
    for i in range(0, len(y), hop_length):
        chunk = y[i:i + hop_length]
        if len(chunk) < hop_length:
            break
        timestamp = round(i / sr, 1)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            librosa.output.write_wav(tmp.name, chunk, sr)
            preds = emo_pipe(tmp.name)
        os.unlink(tmp.name)
        top = max(preds, key=lambda x: x['score'])
        audio_emotions[timestamp] = {"voice": top['label'], "confidence": top['score']}
    return audio_emotions


def build_rich_transcript(segments, vid_emos, aud_emos):
    rich = []
    for seg in segments:
        start = round(seg['start'], 1)
        text = seg['text'].strip()
        face = vid_emos.get(start, {}).get('face')
        f_conf = vid_emos.get(start, {}).get('confidence')
        voice = aud_emos.get(start, {}).get('voice')
        v_conf = aud_emos.get(start, {}).get('confidence')
        rich.append({
            'timestamp': start,
            'text': text,
            'face_emotion': face,
            'face_confidence': f_conf,
            'voice_emotion': voice,
            'voice_confidence': v_conf
        })
    with open(RICH_TRANSCRIPT_PATH, 'w') as f:
        json.dump(rich, f, indent=2)
    return rich


def build_index(chunks, model_name=EMBED_MODEL):
    embedder = SentenceTransformer(model_name)
    texts = [json.dumps(c) for c in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, 'w') as f:
        json.dump(texts, f)
    return index, texts


def chat_with_llama(question, index, texts, k=3):
    embedder = SentenceTransformer(EMBED_MODEL)
    q_emb = embedder.encode([question])
    D, I = index.search(np.array(q_emb), k)
    context = "".join(texts[i] for i in I[0])
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

# --- Gradio Interface ---

def process_video(file):
    # Save upload
    path = BASE_DIR / Path(file.name).name
    with open(path, 'wb') as f:
        f.write(file.read())
    # Run pipeline
    FRAME_DIR.rmdir() if FRAME_DIR.exists() else None
    extract_frames(str(path), FRAME_DIR)
    extract_audio(str(path), AUDIO_PATH)
    segments = transcribe_audio(AUDIO_PATH)
    vid_emos = detect_video_emotions(FRAME_DIR)
    aud_emos = detect_audio_emotions(AUDIO_PATH)
    chunks = build_rich_transcript(segments, vid_emos, aud_emos)
    index, texts = build_index(chunks)
    return "Video processed. You can now ask questions!", (index, texts)


def chat_fn(message, history, state):
    index, texts = state
    answer = chat_with_llama(message, index, texts)
    history = history + [(message, answer)]
    return history, state

with gr.Blocks() as demo:
    gr.Markdown("## Emotion-Aware Video Chat")
    with gr.Row():
        video_in = gr.File(label="Upload Video (.mp4)")
        proc_btn = gr.Button("Process Video")
    status = gr.Textbox(label="Status", interactive=False)
    chat_state = gr.State(None)
    chatbot = gr.Chatbot()
    user_msg = gr.Textbox(label="Ask a question", placeholder="Type here...")

    proc_btn.click(process_video, inputs=video_in, outputs=[status, chat_state])
    user_msg.submit(chat_fn, inputs=[user_msg, chatbot, chat_state], outputs=[chatbot, chat_state])

if __name__ == "__main__":
    demo.launch()
