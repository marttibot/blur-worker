"""
RunPod Serverless Worker for Blur
Handles video motion blur via RIFE interpolation + ffmpeg frame blending.
Uses Practical-RIFE v4.25 with RIFE_HDv3 model.

Memory-efficient: processes frame pairs from video one at a time,
streams interpolated frames directly into ffmpeg via pipe.
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil

import cv2
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

            # ===== Build ffmpeg encoder pipeline =====
            blur_amount = float(config.get("blurAmount", 1.0))
            quality = int(config.get("quality", 20))
            brightness = float(config.get("brightness", 1.0))
            saturation = float(config.get("saturation", 1.0))
            contrast = float(config.get("contrast", 1.0))

            # Build two-pass pipeline:
            # Pass 1: Apply motion blur (tmix) on original frames → temp video
            # Pass 2: Interpolate blurred frames → output
            # This is MUCH faster than applying tmix after interpolation
            # (tmix on 1290 frames vs 6450 frames)

            if blur_amount > 0:
                blend_frames = max(2, int(blur_amount * 5) + 1)
                weights = " ".join(["1"] * blend_frames)
                tmix_filter = f"tmix=frames={blend_frames}:weights='{weights}'"
            else:
                tmix_filter = None

            eq_parts = []
            if brightness != 1.0: eq_parts.append(f"brightness={brightness - 1}")
            if saturation != 1.0: eq_parts.append(f"saturation={saturation}")
            if contrast != 1.0: eq_parts.append(f"contrast={contrast}")
            eq_filter = f"eq={' : '.join(eq_parts)}" if eq_parts else None

            # Build combined pre-interpolation filter
            pre_filters = []
            if tmix_filter:
                pre_filters.append(tmix_filter)
            if eq_filter:
                pre_filters.append(eq_filter)
            pre_filter_str = ",".join(pre_filters) if pre_filters else "null"

            # If blur is enabled, first create a pre-blurred video to interpolate from
            if blur_amount > 0:
                blurred_path = os.path.join(work_dir, "blurred.mp4")
                print(f"Applying motion blur (tmix) on original frames...")
                blur_cmd = [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vf", pre_filter_str,
                    "-c:v", "libx264", "-crf", str(quality),
                    "-pix_fmt", "yuv420p", "-preset", "veryfast",
                    blurred_path
                ]
                result = subprocess.run(blur_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise ValueError(f"Pre-blur failed: {result.stderr}")
                print(f"Pre-blur complete")
                interp_source = blurred_path
            else:
                interp_source = video_path

            if blur_amount > 0:
                blur_info = get_video_info(blurred_path)
                blur_stream = next((s for s in blur_info["streams"] if s["codec_type"] == "video"), None)
                source_fps = eval(blur_stream["r_frame_rate"])
                width = int(blur_stream["width"])
                height = int(blur_stream["height"])
                output_fps = source_fps * multiplier
                print(f"Blurred video: {width}x{height} @ {source_fps:.2f}fps")
            else:
                source_fps = fps
                output_fps = fps * multiplier

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
                "-r", str(output_fps),
                "-i", "-",
                "-vf", post_filter_str,
                "-c:v", "libx264", "-crf", str(quality),
                "-pix_fmt", "yuv420p", "-preset", "veryfast",
                "-movflags", "+faststart",
                output_path
            ]

            print(f"Starting ffmpeg encoder...")
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # ===== Set up RIFE model =====
            sys.path.insert(0, "/workspace/RIFE")
            sys.path.insert(0, "/workspace/RIFE/train_log")
            from RIFE_HDv3 import Model as RIFEModel

            print("Loading RIFE v4.25 model...")
            model = RIFEModel()
            model.load_model("/workspace/RIFE/train_log", -1)
            model.eval()
            model.device()
            print(f"RIFE model loaded on {DEVICE}.")

            h, w = height, width
            tmp = 128
            ph = ((h - 1) // tmp + 1) * tmp
            pw = ((w - 1) // tmp + 1) * tmp
            padding = (0, pw - w, 0, ph - h)

            def pad_image(img):
                return F.pad(img, padding)

            # ===== Stream: read frame pairs → interpolate → pipe to ffmpeg =====
            cap = cv2.VideoCapture(interp_source)
            frames_written = 0
            prev_frame = None
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Write current frame to ffmpeg
                proc.stdin.write(frame.tobytes())
                frames_written += 1

                if prev_frame is not None:
                    # Interpolate between prev and current
                    img0 = torch.from_numpy(prev_frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    img1 = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    img0 = pad_image(img0).to(DEVICE)
                    img1 = pad_image(img1).to(DEVICE)

                    with torch.no_grad():
                        for j in range(1, multiplier):
                            t = j / multiplier
                            mid = model.inference(img0, img1, timestep=t, scale=1.0)
                            mid_np = (mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
                            mid_bgr = cv2.cvtColor(mid_np, cv2.COLOR_RGB2BGR)
                            proc.stdin.write(mid_bgr.tobytes())
                            frames_written += 1

                    # Free GPU memory
                    del img0, img1
                    if frame_idx % 20 == 0:
                        torch.cuda.empty_cache()

                prev_frame = frame
                frame_idx += 1

                if frame_idx % 30 == 0:
                    pct = frame_idx / (duration * fps) * 100
                    print(f"Progress: {frame_idx}/{int(duration * fps)} frames ({pct:.0f}%), {frames_written} output frames")

            cap.release()
            proc.stdin.close()
            stdout, stderr = proc.communicate(timeout=300)

            if proc.returncode != 0:
                raise ValueError(f"ffmpeg encoding failed (code {proc.returncode}): {stderr.decode()[-500:]}")

            print(f"Done: {frames_written} frames → {output_path}")

            # ===== Upload to R2 =====
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
