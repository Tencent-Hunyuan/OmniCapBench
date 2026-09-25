#!/usr/bin/env python3
"""
MTSS Eval - Video Processing Utilities
Handles video compression, frame sampling, audio extraction, and base64 encoding.
"""

import os
import subprocess
import tempfile
import uuid
import base64
import hashlib
import json
import logging
import math

from . import eval_config as config

logger = logging.getLogger("mtss_eval")


# ======================================================================
# Media path resolution
# ======================================================================

def resolve_media_path(video_item: dict) -> str:
    """Return the on-disk video path for a benchmark item.

    ``video_path`` is the canonical record identity used to join predictions
    against the ground truth, so it is never rewritten. When the annotations use
    paths relative to a downloaded dataset root, ``load_input_data`` stores the
    resolved location in ``media_path`` instead.
    """
    return video_item.get("media_path") or video_item.get("video_path", "")


# ======================================================================
# Video info
# ======================================================================

def get_video_info(video_path: str) -> dict:
    """Get video resolution and duration via ffprobe."""
    info = {"width": 0, "height": 0, "duration": 0.0}
    try:
        cmd = [
            config.FFPROBE_PATH, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration",
            "-show_entries", "format=duration",
            "-of", "json", video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if streams:
            info["width"] = streams[0].get("width", 0)
            info["height"] = streams[0].get("height", 0)
            if "duration" in streams[0]:
                info["duration"] = float(streams[0]["duration"])
        fmt = data.get("format", {})
        if "duration" in fmt and info["duration"] == 0:
            info["duration"] = float(fmt["duration"])
    except Exception as e:
        logger.warning(f"Failed to get video info for {video_path}: {e}")
    return info


# ======================================================================
# Audio track detection
# ======================================================================

def has_audio_track(video_path: str) -> bool:
    """Check if a video file contains an audio track using ffprobe."""
    if not video_path or not os.path.exists(video_path):
        return False
    try:
        cmd = [
            config.FFPROBE_PATH, "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return result.stdout.strip() == "audio"
    except Exception as e:
        logger.warning(f"Failed to check audio track for {video_path}: {e}")
        return False


# ======================================================================
# Video compression to target resolution
# ======================================================================

def compress_video_to_resolution(input_path: str, output_path: str,
                                  resolution: int = 448) -> bool:
    """
    Compress video to resolution x resolution (e.g., 448x448).
    Preserves audio track.
    """
    try:
        cmd = [
            config.FFMPEG_PATH, "-y", "-i", input_path,
            "-vf", f"scale={resolution}:{resolution}:force_original_aspect_ratio=decrease,"
                   f"pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        logger.warning(f"Video compression failed: {e}")
        return False


# ======================================================================
# Video segment extraction (for 2-pass: cut video by time range)
# ======================================================================

def extract_video_segment(input_path: str, output_path: str,
                          start_time: float, end_time: float) -> bool:
    """Extract a segment of video from start_time to end_time (seconds)."""
    try:
        duration = end_time - start_time
        cmd = [
            config.FFMPEG_PATH, "-y",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        logger.warning(f"Video segment extraction failed: {e}")
        return False


# ======================================================================
# Audio extraction
# ======================================================================

def extract_audio(video_path: str, cache_dir: str = None) -> str:
    """Extract audio track from video. Returns path to audio file or ''.
    If cache_dir is provided, audio is stored there for later cleanup.
    Otherwise uses system temp dir."""
    if not video_path or not os.path.exists(video_path):
        return ""
    try:
        target_dir = cache_dir if cache_dir else tempfile.gettempdir()
        os.makedirs(target_dir, exist_ok=True)
        out = os.path.join(target_dir,
                           f"eval_audio_{uuid.uuid4().hex[:8]}.mp3")
        cmd = [config.FFMPEG_PATH, "-y", "-i", video_path, "-vn",
               "-acodec", "libmp3lame", "-q:a", "4", out]
        subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return out
    except Exception as e:
        logger.warning(f"Audio extraction failed: {e}")
    return ""


# ======================================================================
# Base64 encoding helpers
# ======================================================================

def encode_file_to_base64(file_path: str) -> str:
    """Read a local file and return its base64-encoded string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_video_data_url(video_path: str) -> str:
    """Convert a local video file to a base64 data URL."""
    b64 = encode_file_to_base64(video_path)
    return f"data:video/mp4;base64,{b64}"


def build_audio_data_url(audio_path: str) -> str:
    """Convert a local audio file to a base64 data URL."""
    b64 = encode_file_to_base64(audio_path)
    return f"data:audio/mpeg;base64,{b64}"


# ======================================================================
# Timestamp overlay (burn red timestamp into video frames)
# ======================================================================

def add_timestamp_overlay(input_path: str, output_path: str) -> bool:
    """Burn a red timestamp (seconds) into the bottom-right corner using ffmpeg."""
    try:
        cmd = [
            config.FFMPEG_PATH, "-y", "-i", input_path,
            "-vf", (
                "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
                "text='%{pts\\:hms}':"
                "x=w-tw-10:y=h-th-10:"
                "fontsize=24:fontcolor=red:"
                "borderw=2:bordercolor=white"
            ),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        logger.warning(f"Timestamp overlay failed: {e}")
        return False


# ======================================================================
# Full video preprocessing pipeline
# ======================================================================

def preprocess_video(video_path: str, resolution: int = 448,
                     add_timestamp: bool = True,
                     cache_dir: str = None) -> str:
    """
    Full preprocessing: compress to resolution + add timestamp overlay.
    Returns path to the processed video file.
    Uses cache_dir to avoid re-processing.
    """
    if cache_dir is None:
        cache_dir = tempfile.gettempdir()
    os.makedirs(cache_dir, exist_ok=True)

    # Generate a deterministic cache filename based on full input path + params
    # Use hash of full path to avoid collisions between videos with same basename
    path_hash = hashlib.md5(video_path.encode()).hexdigest()[:12]
    basename = os.path.splitext(os.path.basename(video_path))[0]
    cache_name = f"{basename}_{path_hash}_{resolution}{'_ts' if add_timestamp else ''}.mp4"
    cached_path = os.path.join(cache_dir, cache_name)

    # Wait briefly for potential in-progress writes from other workers
    if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
        return cached_path

    # Use PID + UUID in temp filenames to avoid race conditions
    # when multiple workers process different videos concurrently
    unique_id = f"{os.getpid()}_{uuid.uuid4().hex[:6]}"

    # Step 1: Compress
    tmp_compressed = os.path.join(cache_dir, f"_tmp_compress_{unique_id}.mp4")
    if not compress_video_to_resolution(video_path, tmp_compressed, resolution):
        logger.warning(f"Compression failed, using original: {video_path}")
        tmp_compressed = video_path

    # Step 2: Add timestamp
    if add_timestamp:
        tmp_ts = os.path.join(cache_dir, f"_tmp_ts_{unique_id}.mp4")
        if add_timestamp_overlay(tmp_compressed, tmp_ts):
            # Clean up temp compressed
            if tmp_compressed != video_path and os.path.exists(tmp_compressed):
                try:
                    os.remove(tmp_compressed)
                except OSError:
                    pass
            # Atomic rename to final cache path (avoids partial reads)
            try:
                os.rename(tmp_ts, cached_path)
            except OSError:
                # Another worker may have already created it
                if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
                    try:
                        os.remove(tmp_ts)
                    except OSError:
                        pass
                else:
                    # rename failed and no cache exists — use tmp directly
                    cached_path = tmp_ts
            # Final existence check — guard against race conditions
            if not os.path.exists(cached_path) or os.path.getsize(cached_path) == 0:
                logger.warning(f"Preprocessed video missing after rename: {cached_path}, falling back to original")
                return video_path
            return cached_path
        else:
            # Fallback: use compressed without timestamp
            if tmp_compressed != video_path:
                try:
                    os.rename(tmp_compressed, cached_path)
                except OSError:
                    if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
                        try:
                            os.remove(tmp_compressed)
                        except OSError:
                            pass
                    else:
                        cached_path = tmp_compressed
            return cached_path
    else:
        if tmp_compressed != video_path:
            try:
                os.rename(tmp_compressed, cached_path)
            except OSError:
                if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
                    try:
                        os.remove(tmp_compressed)
                    except OSError:
                        pass
                else:
                    cached_path = tmp_compressed
        else:
            # Copy to cache
            subprocess.run(["cp", video_path, cached_path], check=True)
        return cached_path


# ======================================================================
# Shot grouping for 2-pass long video inference
# ======================================================================

def group_shots_by_duration(shot_times: list,
                            target_duration: float = 60.0) -> list:
    """
    Group shots into chunks where each chunk targets ~target_duration seconds.
    Splits at shot boundaries, not at arbitrary time points.

    Args:
        shot_times: list of [start, end] pairs
        target_duration: target duration per group in seconds

    Returns:
        list of groups, each group is a list of [start, end] pairs
    """
    if not shot_times:
        return []

    groups = []
    current_group = []
    current_duration = 0.0

    for shot in shot_times:
        shot_dur = shot[1] - shot[0]
        # If adding this shot would exceed target and we already have shots in group
        if current_duration + shot_dur > target_duration * 1.3 and current_group:
            groups.append(current_group)
            current_group = [shot]
            current_duration = shot_dur
        else:
            current_group.append(shot)
            current_duration += shot_dur

    if current_group:
        groups.append(current_group)

    return groups
