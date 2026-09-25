#!/usr/bin/env python3
"""
MTSS Pipeline - Media Utilities
Video / audio preprocessing helpers extracted from qwen_infer.py.
"""

import os
import subprocess
import tempfile
import uuid
import base64
import mimetypes
import logging

import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from . import config

logger = logging.getLogger("omnicapbench_pipeline")

# ======================================================================
# MIME / Base64 helpers
# ======================================================================

def get_mime_type(file_path: str) -> str:
    """Get MIME type for a media file."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        ext = os.path.splitext(file_path)[1].lower()
        mime_map = {
            ".mp4": "video/mp4", ".avi": "video/x-msvideo",
            ".mov": "video/quicktime", ".mkv": "video/x-matroska",
            ".webm": "video/webm", ".mp3": "audio/mpeg",
            ".wav": "audio/wav", ".flac": "audio/flac",
            ".aac": "audio/aac", ".ogg": "audio/ogg", ".m4a": "audio/mp4",
        }
        mime_type = mime_map.get(ext, "application/octet-stream")
    return mime_type


def encode_file_to_base64(file_path: str) -> str:
    """Read a local file and return its base64-encoded string."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_video_url(video_path: str) -> str:
    """Convert a local video file to a base64 data URL."""
    b64 = encode_file_to_base64(video_path)
    mime = get_mime_type(video_path)
    return f"data:{mime};base64,{b64}"


def build_audio_url(audio_path: str) -> str:
    """Convert a local audio file to a base64 data URL."""
    b64 = encode_file_to_base64(audio_path)
    mime = get_mime_type(audio_path)
    return f"data:{mime};base64,{b64}"


# ======================================================================
# Video info / compression
# ======================================================================

def get_video_info(video_path: str) -> dict:
    """Get video resolution via ffprobe."""
    try:
        cmd = [
            config.FFPROBE_PATH, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json", video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        info = __import__("json").loads(result.stdout)
        streams = info.get("streams", [])
        if streams:
            return {"width": streams[0].get("width", 0),
                    "height": streams[0].get("height", 0)}
    except Exception as e:
        logger.warning(f"Failed to get video info: {e}")
    return {"width": 0, "height": 0}


def compress_video(input_path: str, output_path: str,
                   max_height: int = 720) -> bool:
    """Compress video to a target max height."""
    try:
        info = get_video_info(input_path)
        h, w = info["height"], info["width"]
        if h == 0:
            subprocess.run(["cp", input_path, output_path], check=True)
            return True
        if h <= max_height:
            subprocess.run(["cp", input_path, output_path], check=True)
            return True

        cmd = [
            config.FFMPEG_PATH, "-y", "-i", input_path,
            "-vf", f"scale=-2:{max_height}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", output_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        logger.warning(f"Video compression failed: {e}")
        return False


# ======================================================================
# Timestamp overlay
# ======================================================================

def _adaptive_font_scale(w, h, base_h=1080, base_w=1520, base_scale=2):
    ratio = h / base_h if h >= w else w / base_w
    return base_scale * ratio


def process_video_with_timestamp(input_path: str, output_path: str) -> bool:
    """Burn a red timestamp (seconds) into the bottom-right corner."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_out = output_path.replace(".mp4", "_temp.mp4")
    out = cv2.VideoWriter(temp_out, fourcc, fps, (w, h))

    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_no / fps
        text = f"{t:.1f}s"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = _adaptive_font_scale(w, h)
        thick = max(3, int(5 * (h / 1080 if h >= w else w / 1520)))
        sz, _ = cv2.getTextSize(text, font, scale, thick)
        pos = (w - sz[0] - 10, h - sz[1] - 10)
        # white outline
        for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
            cv2.putText(frame, text, (pos[0]+dx, pos[1]+dy),
                        font, scale, (255, 255, 255), thick * 2)
        # red text
        cv2.putText(frame, text, pos, font, scale, (0, 0, 255), thick)
        out.write(frame)
        frame_no += 1

    cap.release()
    out.release()

    # Merge audio back
    try:
        cmd = [
            config.FFMPEG_PATH, "-y", "-i", temp_out, "-i", input_path,
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest", output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        os.remove(temp_out)
    except Exception:
        if os.path.exists(temp_out):
            os.rename(temp_out, output_path)
    return True


def add_timestamp_to_video(video_path: str) -> str:
    """Add timestamp overlay; return path to the new file."""
    if not video_path or not os.path.exists(video_path):
        return video_path
    try:
        out = os.path.join(tempfile.gettempdir(),
                           f"ts_{uuid.uuid4().hex[:8]}.mp4")
        if process_video_with_timestamp(video_path, out):
            if os.path.exists(out) and os.path.getsize(out) > 0:
                return out
    except Exception as e:
        logger.warning(f"Timestamp overlay failed: {e}")
    return video_path


# ======================================================================
# Audio extraction
# ======================================================================

def extract_audio_from_video(video_path: str) -> str:
    """Extract audio track from video; return path or ''."""
    if not video_path or not os.path.exists(video_path):
        return ""
    try:
        out = os.path.join(tempfile.gettempdir(),
                           f"audio_{uuid.uuid4().hex[:8]}.mp3")
        cmd = [config.FFMPEG_PATH, "-y", "-i", video_path, "-vn",
               "-acodec", "libmp3lame", "-q:a", "4", out]
        subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return out
        # fallback
        cmd2 = [config.FFMPEG_PATH, "-y", "-i", video_path,
                "-vn", "-f", "mp3", out]
        subprocess.run(cmd2, capture_output=True, text=True, timeout=60)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return out
    except Exception as e:
        logger.warning(f"Audio extraction failed: {e}")
    return ""


# ======================================================================
# Shot detection
# ======================================================================

def detect_shots(video_path: str, threshold: float = 27.0) -> list:
    """Detect shot boundaries using PySceneDetect."""
    if not video_path or not os.path.exists(video_path):
        return []
    try:
        video = open_video(video_path)
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=threshold))
        sm.detect_scenes(video)
        scenes = sm.get_scene_list()
        if not scenes:
            dur = video.duration.get_seconds()
            return [[0, round(dur, 2)]]
        return [[round(s[0].get_seconds(), 2), round(s[1].get_seconds(), 2)]
                for s in scenes]
    except Exception as e:
        logger.warning(f"Shot detection error: {e}")
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                dur = fc / fps if fps > 0 else 0
                cap.release()
                if dur > 0:
                    return [[0, round(dur, 2)]]
        except Exception:
            pass
    return []


# ======================================================================
# Build multimodal content list for model input
# ======================================================================

def build_multimodal_content(video_path: str, audio_path: str,
                             text_prompt: str,
                             use_timestamp_video: bool = True) -> list:
    """
    Build the content list (video + audio + text) for model input.

    Args:
        video_path:           Path to the original video
        audio_path:           Path to audio (or '' to auto-extract)
        text_prompt:          The text prompt to append
        use_timestamp_video:  Whether to add timestamp overlay first

    Returns:
        list of content dicts ready for model_client.inference()
    """
    content = []

    # --- Video ---
    vp = video_path
    if use_timestamp_video:
        # Compress first
        tmp = os.path.join(tempfile.gettempdir(),
                           f"compress_{uuid.uuid4().hex[:8]}.mp4")
        if compress_video(video_path, tmp, 720):
            vp = add_timestamp_to_video(tmp)
        else:
            vp = add_timestamp_to_video(video_path)

    video_url = build_video_url(vp)
    content.append({"type": "video_url", "video_url": {"url": video_url}})

    # --- Audio ---
    ap = audio_path
    if not ap or not os.path.exists(ap):
        ap = extract_audio_from_video(video_path)
    if ap and os.path.exists(ap):
        audio_url = build_audio_url(ap)
        content.append({"type": "audio_url", "audio_url": {"url": audio_url}})

    # --- Text ---
    content.append({"type": "text", "text": text_prompt})

    return content
