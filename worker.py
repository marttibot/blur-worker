"""
RunPod Serverless Worker for Blur
Handles video motion blur via RIFE interpolation + ffmpeg frame blending.
Uses Practical-RIFE v4.25 with RIFE_HDv3 model.

Pipeline (matches teknos blur v2.42):
  1. Decode source frames (one at a time, memory-efficient)
  2. Deduplicate: skip interpolation on near-identical consecutive frames
  3. RIFE interpolate → stream ALL frames at interpolated FPS to ffmpeg
  4. ffmpeg: tmix motion blur → fps=60 downsample → eq color filters → h265 encode
  5. Upload direct to R2 via S3 API
"""

import os
import sys
import json
import math
import subprocess
import tempfile
import shutil
import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import requests
import boto3
from botocore.config import Config as BotoConfig

# Detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[init] PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, device: {DEVICE}")
if torch.cuda.is_available():
    print(f"[init] GPU: {torch.cuda.get_device_name(0)}")


def download_file(url: str, dest: str) -> None:
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def get_video_info(video_path: str) -> dict:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", "-show_format", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffprobe failed: {result.stderr}")
    return json.loads(result.stdout)


def handler(event: dict) -> dict:
    job_input = event.get("input", {})
    try:
        config = job_input.get("config", {})
        video_url = job_input.get("videoUrl", "")
        job_id = job_input.get("jobId", "unknown")

        print(f"[handler] Job {job_id} | device: {DEVICE} | CUDA: {torch.cuda.is_available()}")

        if not video_url:
            return {"error": "No video URL provided"}

        # Parse config
        multiplier_str = str(config.get("interpolationMultiplier", "5x"))
        multiplier = int(multiplier_str.replace("x", ""))
        blur_amount = float(config.get("blurAmount", 1.0))
        quality = int(config.get("quality", 20))
        brightness = float(config.get("brightness", 1.0))
        saturation = float(config.get("saturation", 1.0))
        contrast = float(config.get("contrast", 1.0))
        codec = str(config.get("codec", "h265")).lower()

        # Target output FPS (teknos blur default: 60)
        target_output_fps = 60

        work_dir = tempfile.mkdtemp()

        try:
            # ===== Download =====
            video_path = os.path.join(work_dir, "input.mp4")
            print(f"Downloading video from {video_url}...")
            download_file(video_url, video_path)

            info = get_video_info(video_path)
            duration = float(info["format"]["duration"])
            print(f"Video duration: {duration}s")
            if duration > 65:
                raise ValueError("Video exceeds 60 second limit")

            video_stream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
            if not video_stream:
                raise ValueError("No video stream found")

            source_fps = eval(video_stream["r_frame_rate"])
            width = int(video_stream["width"])
            height = int(video_stream["height"])
            h, w = height, width

            # Use full interpolation multiplier (output FPS handled by ffmpeg fps filter)
            actual_multiplier = multiplier
            interpolated_fps = source_fps * actual_multiplier

            # tmix blend frames: ratio of interpolated to output FPS * blur_amount
            # E.g., 150fps interp / 60fps output * 1.2 blur = 3 blend frames
            ratio = interpolated_fps / target_output_fps
            blend_frames = max(2, round(ratio * blur_amount))

            print(f"Source: {w}x{h} @ {source_fps:.2f}fps")
            print(f"Interpolation: {actual_multiplier}x → {interpolated_fps:.0f}fps")
            print(f"Blur: amount={blur_amount}, blend_frames={blend_frames}")
            print(f"Output: {target_output_fps}fps, codec={codec}, crf={quality}")

            # ===== Build ffmpeg filter chain =====
            # Order: tmix (subtle blend at high fps) → fps=60 (downsample) → eq (color)
            filters = []

            if blur_amount > 0:
                weights = " ".join(["1"] * blend_frames)
                filters.append(f"tmix=frames={blend_frames}:weights='{weights}'")

            # Downsample to target output FPS
            if interpolated_fps > target_output_fps:
                filters.append(f"fps={target_output_fps}")

            # Color filters
            eq_parts = []
            if brightness != 1.0:
                eq_parts.append(f"brightness={brightness - 1}")
            if saturation != 1.0:
                eq_parts.append(f"saturation={saturation}")
            if contrast != 1.0:
                eq_parts.append(f"contrast={contrast}")
            if eq_parts:
                filters.append(f"eq={' : '.join(eq_parts)}")

            filter_str = ",".join(filters) if filters else "null"

            # ===== Start ffmpeg encoder =====
            output_path = os.path.join(work_dir, "output.mp4")
            vcodec = "libx265" if codec == "h265" else "libx264"

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                # Input 0: raw video from stdin
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}",
                "-r", str(interpolated_fps),
                "-i", "-",
                # Input 1: source video (for audio track)
                "-i", video_path,
                # Video: filter + encode
                "-vf", filter_str,
                "-c:v", vcodec, "-crf", str(quality),
                "-pix_fmt", "yuv420p", "-preset", "veryfast",
                # Audio: copy from source (if present)
                "-map", "0:v",
                "-map", "1:a?",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                output_path
            ]

            print(f"ffmpeg: {' '.join(ffmpeg_cmd)}")
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Thread to capture ffmpeg stderr for debugging
            stderr_lines = []
            def read_stderr():
                for line in proc.stderr:
                    stderr_lines.append(line.decode(errors='replace').strip())
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()

            # ===== Load RIFE model =====
            sys.path.insert(0, "/workspace/RIFE")

            # Download model weights from R2 if not present
            weights_dir = "/workspace/RIFE/train_log"
            if not os.path.exists(os.path.join(weights_dir, "flownet.pkl")):
                print("Downloading RIFE model weights from R2...")
                os.makedirs(weights_dir, exist_ok=True)
                r2_model = boto3.client(
                    "s3",
                    endpoint_url="https://db62c194342e7bde5f3a192d3879680c.r2.cloudflarestorage.com",
                    aws_access_key_id="0d24cbc72ed92460208b48ac016fa264",
                    aws_secret_access_key="a312c485489cbf320ee6ecc3e1c5b5cd17c7f06765637f93e6c49d1831b3d133",
                    config=BotoConfig(region_name="auto"),
                )
                for key in ["RIFE_HDv3.py", "IFNet_HDv3.py", "refine.py", "flownet.pkl"]:
                    r2_model.download_file("blur-renders", f"models/rife/train_log/{key}", os.path.join(weights_dir, key))
                    print(f"  Downloaded {key}")
            else:
                print("Model weights already cached locally.")

            sys.path.insert(0, weights_dir)
            from RIFE_HDv3 import Model as RIFEModel

            print("Loading RIFE v4.25 model...")
            model = RIFEModel()
            model.load_model(weights_dir, -1)
            model.eval()
            model.device()
            print(f"RIFE model loaded on {DEVICE}.")

            # Padding for RIFE (must be multiple of 128)
            tmp = 128
            ph = ((h - 1) // tmp + 1) * tmp
            pw = ((w - 1) // tmp + 1) * tmp
            padding = (0, pw - w, 0, ph - h)

            def pad_image(img):
                return F.pad(img, padding)

            # ===== Stream: decode → dedup → interpolate ALL frames → pipe to ffmpeg =====
            cap = cv2.VideoCapture(video_path)
            prev_frame = None
            frame_idx = 0
            frames_written = 0
            dedup_skipped = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Write source frame (BGR) to ffmpeg
                try:
                    proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    print(f"[ERROR] ffmpeg stdin broken at frame {frame_idx}")
                    break
                frames_written += 1

                if prev_frame is not None:
                    # Dedup check: mean absolute difference
                    diff = np.abs(prev_frame.astype(np.float32) - frame.astype(np.float32)).mean()

                    if diff > 1.5:
                        # Frames are different — interpolate
                        # Convert BGR (cv2) to RGB for RIFE
                        img0_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
                        img1_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        img0 = torch.from_numpy(img0_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                        img1 = torch.from_numpy(img1_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                        img0 = pad_image(img0).to(DEVICE)
                        img1 = pad_image(img1).to(DEVICE)

                        with torch.no_grad():
                            for j in range(1, actual_multiplier):
                                t = j / actual_multiplier
                                mid = model.inference(img0, img1, timestep=t, scale=1.0)
                                # RIFE outputs RGB, convert to BGR for ffmpeg
                                mid_np = (mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
                                mid_bgr = cv2.cvtColor(mid_np, cv2.COLOR_RGB2BGR)
                                try:
                                    proc.stdin.write(mid_bgr.tobytes())
                                except BrokenPipeError:
                                    print("[ERROR] ffmpeg stdin broken during interpolation")
                                    break
                                frames_written += 1

                        del img0, img1
                    else:
                        # Frames too similar — repeat current frame (dedup)
                        dedup_skipped += 1
                        for _ in range(1, actual_multiplier):
                            try:
                                proc.stdin.write(frame.tobytes())
                            except BrokenPipeError:
                                break
                            frames_written += 1

                    # Free GPU memory periodically
                    if frame_idx % 30 == 0:
                        torch.cuda.empty_cache()

                prev_frame = frame
                frame_idx += 1

                if frame_idx % 30 == 0:
                    est_total = int(duration * source_fps)
                    pct = frame_idx / est_total * 100
                    print(f"Progress: {frame_idx}/{est_total} src ({pct:.0f}%), "
                          f"{frames_written} out, {dedup_skipped} deduped")

            cap.release()
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
            proc.wait(timeout=300)
            stderr_thread.join(timeout=5)

            if proc.returncode != 0:
                err_output = "\n".join(stderr_lines[-20:])
                raise ValueError(f"ffmpeg failed (code {proc.returncode}): {err_output}")

            print(f"Encoding complete: {frames_written} frames piped, {dedup_skipped} pairs deduped")

            # ===== Upload to R2 =====
            file_size = os.path.getsize(output_path)
            print(f"Output: {file_size / 1024 / 1024:.1f} MB")

            output_key = f"jobs/{job_id}/output.mp4"
            print(f"Uploading to R2: {output_key}...")
            r2 = boto3.client(
                "s3",
                endpoint_url="https://db62c194342e7bde5f3a192d3879680c.r2.cloudflarestorage.com",
                aws_access_key_id="0d24cbc72ed92460208b48ac016fa264",
                aws_secret_access_key="a312c485489cbf320ee6ecc3e1c5b5cd17c7f06765637f93e6c49d1831b3d133",
                config=BotoConfig(region_name="auto"),
            )
            r2.upload_file(output_path, "blur-renders", output_key, ExtraArgs={"ContentType": "video/mp4"})
            print(f"Upload complete: {output_key}")

            return {
                "status": "completed",
                "outputKey": output_key,
                "duration": duration,
                "outputFps": target_output_fps,
                "interpolatedFps": interpolated_fps,
                "blendFrames": blend_frames,
                "dedupSkipped": dedup_skipped,
            }

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
