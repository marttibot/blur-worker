"""
RunPod Serverless Worker for Blur
Handles video motion blur via RIFE interpolation + ffmpeg frame blending.
Uses Practical-RIFE v4.25 with RIFE_HDv3 model.

Optimized: streams interpolated frames directly into ffmpeg via pipe.
No intermediate PNG files for output — much faster encoding.
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
import struct
import numpy as np

import requests
import cv2
import torch
import torch.nn.functional as F
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

        multiplier_str = str(config.get("interpolationMultiplier", "5x"))
        multiplier = int(multiplier_str.replace("x", ""))
        work_dir = tempfile.mkdtemp()

        try:
            # Download video
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
            
            fps = eval(video_stream["r_frame_rate"])
            width = int(video_stream["width"])
            height = int(video_stream["height"])
            output_fps = fps * multiplier

            print(f"Video: {width}x{height} @ {fps:.2f}fps → {multiplier}x → {output_fps:.0f}fps")

            # ===== PHASE 1: Decode all frames to numpy arrays in memory =====
            print("Decoding frames...")
            cap = cv2.VideoCapture(video_path)
            frames_bgr = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_bgr.append(frame)
            cap.release()
            print(f"Decoded {len(frames_bgr)} frames")

            if len(frames_bgr) < 2:
                raise ValueError("Video has fewer than 2 frames")

            # ===== PHASE 2: Set up RIFE model =====
            sys.path.insert(0, "/workspace/RIFE")
            sys.path.insert(0, "/workspace/RIFE/train_log")
            from RIFE_HDv3 import Model as RIFEModel

            print("Loading RIFE v4.25 model...")
            model = RIFEModel()
            model.load_model("/workspace/RIFE/train_log", -1)
            model.eval()
            model.device()
            print(f"RIFE model loaded on {DEVICE}.")

            h, w = frames_bgr[0].shape[:2]
            tmp = 128
            ph = ((h - 1) // tmp + 1) * tmp
            pw = ((w - 1) // tmp + 1) * tmp
            padding = (0, pw - w, 0, ph - h)

            def pad_image(img):
                return F.pad(img, padding)

            # ===== PHASE 3: Build ffmpeg encoder pipeline =====
            blur_amount = float(config.get("blurAmount", 1.0))
            quality = int(config.get("quality", 20))
            brightness = float(config.get("brightness", 1.0))
            saturation = float(config.get("saturation", 1.0))
            contrast = float(config.get("contrast", 1.0))

            filters = []
            if blur_amount > 0:
                blend_frames = max(2, int(blur_amount * 5) + 1)
                weights = " ".join(["1"] * blend_frames)
                filters.append(f"tmix=frames={blend_frames}:weights='{weights}'")

            if brightness != 1.0 or saturation != 1.0 or contrast != 1.0:
                eq_parts = []
                if brightness != 1.0: eq_parts.append(f"brightness={brightness - 1}")
                if saturation != 1.0: eq_parts.append(f"saturation={saturation}")
                if contrast != 1.0: eq_parts.append(f"contrast={contrast}")
                filters.append(f"eq={' : '.join(eq_parts)}")

            filter_str = ",".join(filters) if filters else "null"

            output_path = os.path.join(work_dir, "output.mp4")
            total_expected = len(frames_bgr) + (len(frames_bgr) - 1) * (multiplier - 1)

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}",
                "-r", str(output_fps),
                "-i", "-",
                "-vf", filter_str,
                "-c:v", "libx264", "-crf", str(quality),
                "-pix_fmt", "yuv420p", "-preset", "veryfast",
                "-movflags", "+faststart",
                output_path
            ]

            print(f"Starting ffmpeg encoder (expecting {total_expected} frames)...")
            print(f"  cmd: {' '.join(ffmpeg_cmd)}")
            
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # ===== PHASE 4: Interpolate + stream frames directly to ffmpeg =====
            frames_written = 0
            for i in range(len(frames_bgr)):
                # Write original frame
                proc.stdin.write(frames_bgr[i].tobytes())
                frames_written += 1

                if i >= len(frames_bgr) - 1:
                    break

                # Interpolate between frame i and frame i+1
                img0 = torch.from_numpy(frames_bgr[i]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                img1 = torch.from_numpy(frames_bgr[i + 1]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                img0 = pad_image(img0).to(DEVICE)
                img1 = pad_image(img1).to(DEVICE)

                with torch.no_grad():
                    for j in range(1, multiplier):
                        t = j / multiplier
                        mid = model.inference(img0, img1, timestep=t, scale=1.0)
                        mid_np = (mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
                        # mid_np is RGB, convert to BGR for ffmpeg raw bgr24
                        mid_bgr = cv2.cvtColor(mid_np, cv2.COLOR_RGB2BGR)
                        proc.stdin.write(mid_bgr.tobytes())
                        frames_written += 1

                if (i + 1) % 10 == 0:
                    pct = (i + 1) / len(frames_bgr) * 100
                    print(f"Interpolation: {i + 1}/{len(frames_bgr)} pairs ({pct:.0f}%), {frames_written} frames written")

            proc.stdin.close()
            stdout, stderr = proc.communicate(timeout=300)

            if proc.returncode != 0:
                raise ValueError(f"ffmpeg encoding failed (code {proc.returncode}): {stderr.decode()[-500:]}")

            print(f"Encoding complete: {frames_written} frames → {output_path}")

            # ===== PHASE 5: Upload to R2 =====
            file_size = os.path.getsize(output_path)
            print(f"Output size: {file_size / 1024 / 1024:.1f} MB")

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
                "outputFps": output_fps,
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
