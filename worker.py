"""
RunPod Serverless Worker for Blur
Handles video motion blur via RIFE interpolation + ffmpeg frame blending.
Uses Practical-RIFE v4.25 with RIFE_HDv3 model.

Pipeline:
  1. Pre-blur: Apply tmix motion blur + color filters on original video
  2. Deduplicate: Skip interpolation on near-identical frame pairs
  3. Interpolate: RIFE frame interpolation, capped at target FPS
  4. Encode: Stream directly to ffmpeg (h265 or h264)
  5. Upload: Direct to R2 via S3 API

Memory-efficient: only 2 frames in memory at any time, raw pipe to ffmpeg.
"""

import os
import sys
import json
import math
import subprocess
import tempfile
import shutil

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
        max_multiplier = int(multiplier_str.replace("x", ""))
        blur_amount = float(config.get("blurAmount", 1.0))
        quality = int(config.get("quality", 20))
        brightness = float(config.get("brightness", 1.0))
        saturation = float(config.get("saturation", 1.0))
        contrast = float(config.get("contrast", 1.0))
        codec = str(config.get("codec", "h265")).lower()
        try:
            target_output_fps = int(config.get("outputFps", 60))
        except (ValueError, TypeError):
            target_output_fps = 60

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

            source_fps = eval(video_stream["r_frame_rate"])
            width = int(video_stream["width"])
            height = int(video_stream["height"])
            h, w = height, width

            # Calculate interpolation multiplier, capped at target output FPS
            actual_multiplier = min(max_multiplier, max(1, math.ceil(target_output_fps / source_fps)))
            output_fps = source_fps * actual_multiplier
            print(f"Source: {w}x{h} @ {source_fps:.2f}fps | Multiplier: {actual_multiplier}x → {output_fps:.0f}fps | Codec: {codec}")

            # ===== PHASE 1: Pre-blur + color filters on original video =====
            pre_filters = []
            if blur_amount > 0:
                blend_frames = max(2, int(blur_amount * 5) + 1)
                weights = " ".join(["1"] * blend_frames)
                pre_filters.append(f"tmix=frames={blend_frames}:weights='{weights}'")

            eq_parts = []
            if brightness != 1.0:
                eq_parts.append(f"brightness={brightness - 1}")
            if saturation != 1.0:
                eq_parts.append(f"saturation={saturation}")
            if contrast != 1.0:
                eq_parts.append(f"contrast={contrast}")
            if eq_parts:
                pre_filters.append(f"eq={' : '.join(eq_parts)}")

            if pre_filters:
                pre_filter_str = ",".join(pre_filters)
                blurred_path = os.path.join(work_dir, "blurred.mp4")
                print(f"Pre-processing (blur + filters): {pre_filter_str}")
                blur_cmd = [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vf", pre_filter_str,
                    "-c:v", "libx264", "-crf", "16", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p",
                    blurred_path
                ]
                result = subprocess.run(blur_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise ValueError(f"Pre-blur failed: {result.stderr[-500:]}")
                print(f"Pre-blur complete ({os.path.getsize(blurred_path)/1024/1024:.1f} MB)")
                interp_source = blurred_path
            else:
                interp_source = video_path

            # ===== PHASE 2: Start ffmpeg encoder (receives raw frames via pipe) =====
            output_path = os.path.join(work_dir, "output.mp4")
            vcodec = "libx265" if codec == "h265" else "libx264"

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}",
                "-r", str(output_fps),
                "-i", "-",
                "-c:v", vcodec, "-crf", str(quality),
                "-pix_fmt", "yuv420p", "-preset", "veryfast",
                "-movflags", "+faststart",
                output_path
            ]

            print(f"Starting ffmpeg encoder ({vcodec} crf{quality})...")
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Thread to capture ffmpeg stderr for debugging
            import threading
            stderr_lines = []
            def read_stderr():
                for line in proc.stderr:
                    stderr_lines.append(line.decode(errors='replace').strip())
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()

            # ===== PHASE 3: Load RIFE model =====
            sys.path.insert(0, "/workspace/RIFE")
            sys.path.insert(0, "/workspace/RIFE/train_log")
            from RIFE_HDv3 import Model as RIFEModel

            print("Loading RIFE v4.25 model...")
            model = RIFEModel()
            model.load_model("/workspace/RIFE/train_log", -1)
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

            # ===== PHASE 4: Stream frames → dedup → interpolate → pipe to ffmpeg =====
            cap = cv2.VideoCapture(interp_source)
            prev_frame = None
            frame_idx = 0
            frames_written = 0
            dedup_skipped = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Always write the current frame
                try:
                    proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    print(f"[ERROR] ffmpeg stdin broken at frame {frame_idx}. ffmpeg may have crashed.")
                    break
                frames_written += 1

                if prev_frame is not None:
                    # Dedup check: compute mean absolute difference
                    diff = np.abs(prev_frame.astype(np.float32) - frame.astype(np.float32)).mean()

                    if diff > 1.5:
                        # Frames are different — interpolate
                        img0 = torch.from_numpy(prev_frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                        img1 = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                        img0 = pad_image(img0).to(DEVICE)
                        img1 = pad_image(img1).to(DEVICE)

                        with torch.no_grad():
                            for j in range(1, actual_multiplier):
                                t = j / actual_multiplier
                                mid = model.inference(img0, img1, timestep=t, scale=1.0)
                                mid_np = (mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
                                mid_bgr = cv2.cvtColor(mid_np, cv2.COLOR_RGB2BGR)
                                try:
                                    proc.stdin.write(mid_bgr.tobytes())
                                except BrokenPipeError:
                                    print(f"[ERROR] ffmpeg stdin broken during interpolation.")
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
                                print(f"[ERROR] ffmpeg stdin broken during dedup.")
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
                    print(f"Progress: {frame_idx}/{est_total} src frames ({pct:.0f}%), "
                          f"{frames_written} out frames, {dedup_skipped} deduped")

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

            print(f"Encoding complete: {frames_written} frames, {dedup_skipped} pairs deduped")

            # ===== PHASE 5: Upload to R2 =====
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
                "outputFps": output_fps,
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
