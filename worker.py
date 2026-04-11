"""
RunPod Serverless Worker for Blur
Handles video motion blur via RIFE interpolation + ffmpeg frame blending.
Uses Practical-RIFE v4.25 with RIFE_HDv3 model.
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil

import requests
import cv2
import torch
import torch.nn.functional as F

# Detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[init] PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, device: {DEVICE}")
if torch.cuda.is_available():
    print(f"[init] GPU: {torch.cuda.get_device_name(0)}")


def download_file(url: str, dest: str) -> None:
    """Download a file from URL to destination."""
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffprobe failed: {result.stderr}")
    return json.loads(result.stdout)


def extract_frames(video_path: str, output_dir: str) -> int:
    """Extract frames from video using ffmpeg. Returns the source FPS."""
    os.makedirs(output_dir, exist_ok=True)
    output_pattern = os.path.join(output_dir, "frame_%06d.png")

    cmd = [
        "ffmpeg", "-i", video_path,
        "-qscale:v", "1",
        "-vsync", "0",
        output_pattern
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"Frame extraction failed: {result.stderr}")

    info = get_video_info(video_path)
    video_stream = next(
        (s for s in info["streams"] if s["codec_type"] == "video"),
        None
    )
    if not video_stream:
        raise ValueError("No video stream found")

    fps = eval(video_stream["r_frame_rate"])
    frames = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".png")
    ]
    print(f"Extracted {len(frames)} frames at {fps:.2f} fps")
    return fps


def interpolate_frames(
    input_dir: str,
    output_dir: str,
    multiplier: int
) -> None:
    """Interpolate frames using RIFE v4.25 HDv3 model."""
    os.makedirs(output_dir, exist_ok=True)

    # Add RIFE to path and import the model from train_log
    sys.path.insert(0, "/workspace/RIFE")
    sys.path.insert(0, "/workspace/RIFE/train_log")

    from RIFE_HDv3 import Model as RIFEModel

    print("Loading RIFE v4.25 model...")
    model = RIFEModel()
    model.load_model("/workspace/RIFE/train_log", -1)
    model.eval()
    model.device()
    print(f"RIFE model loaded on {DEVICE}.")

    # Get input frames
    frame_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".png")
    ])

    output_idx = 0

    for i in range(len(frame_files)):
        # Write original frame
        img_bgr = cv2.imread(frame_files[i])
        out_path = os.path.join(output_dir, f"interp_{output_idx:06d}.png")
        cv2.imwrite(out_path, img_bgr)
        output_idx += 1

        if i >= len(frame_files) - 1:
            break

        # Read frames for interpolation
        img0 = torch.from_numpy(cv2.imread(frame_files[i])).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img1 = torch.from_numpy(cv2.imread(frame_files[i + 1])).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        img0 = img0.to(DEVICE)
        img1 = img1.to(DEVICE)

        # Pad to multiples of 32 (required by RIFE)
        _, _, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        # Generate intermediate frames
        with torch.no_grad():
            for j in range(1, multiplier):
                t = j / multiplier
                mid = model.inference(img0, img1, timestep=t)
                # Crop back to original size
                mid_np = (mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
                mid_bgr = cv2.cvtColor(mid_np, cv2.COLOR_RGB2BGR)

                out_path = os.path.join(output_dir, f"interp_{output_idx:06d}.png")
                cv2.imwrite(out_path, mid_bgr)
                output_idx += 1

        if i % 10 == 0:
            print(f"Interpolation progress: {i + 1}/{len(frame_files)}")

    print(f"Generated {output_idx} interpolated frames")


def blend_and_encode(
    frames_dir: str,
    output_path: str,
    config: dict,
    output_fps: float
) -> None:
    """Blend frames with motion blur and encode to H.265."""
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
        if brightness != 1.0:
            eq_parts.append(f"brightness={brightness - 1}")
        if saturation != 1.0:
            eq_parts.append(f"saturation={saturation}")
        if contrast != 1.0:
            eq_parts.append(f"contrast={contrast}")
        filters.append(f"eq={' : '.join(eq_parts)}")

    filter_str = ",".join(filters) if filters else "null"

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(output_fps),
        "-i", os.path.join(frames_dir, "interp_%06d.png"),
        "-vf", filter_str,
        "-c:v", "libx265",
        "-crf", str(quality),
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-movflags", "+faststart",
        output_path
    ]

    print(f"Encoding with: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"Encoding failed: {result.stderr}")
    print("Encoding complete.")


def handler(event: dict) -> dict:
    """Main RunPod serverless handler."""
    job_input = event.get("input", {})

    try:
        config = job_input.get("config", {})
        video_url = job_input.get("videoUrl", "")
        output_callback_url = job_input.get("outputCallbackUrl", "")
        job_id = job_input.get("jobId", "unknown")

        print(f"[handler] Job {job_id} | device: {DEVICE} | CUDA: {torch.cuda.is_available()}")

        if not video_url:
            return {"error": "No video URL provided"}

        multiplier_str = str(config.get("interpolationMultiplier", "5x"))
        multiplier = int(multiplier_str.replace("x", ""))

        work_dir = tempfile.mkdtemp()

        try:
            # Step 1: Download video
            video_path = os.path.join(work_dir, "input.mp4")
            print(f"Downloading video from {video_url}...")
            download_file(video_url, video_path)

            info = get_video_info(video_path)
            duration = float(info["format"]["duration"])
            print(f"Video duration: {duration}s")

            if duration > 65:
                raise ValueError("Video exceeds 60 second limit")

            # Step 2: Extract frames
            print("Extracting frames...")
            frames_dir = os.path.join(work_dir, "frames")
            source_fps = extract_frames(video_path, frames_dir)

            # Step 3: Interpolate with RIFE
            print(f"Interpolating {multiplier}x...")
            interp_dir = os.path.join(work_dir, "interpolated")
            interpolate_frames(frames_dir, interp_dir, multiplier)

            # Step 4: Blend and encode
            output_fps = source_fps * multiplier
            output_path = os.path.join(work_dir, "output.mp4")
            print(f"Encoding at {output_fps:.0f} fps...")
            blend_and_encode(interp_dir, output_path, config, output_fps)

            # Step 5: Upload result
            output_url = None
            if output_callback_url:
                with open(output_path, "rb") as f:
                    upload_resp = requests.put(
                        output_callback_url,
                        files={"file": ("output.mp4", f, "video/mp4")},
                        timeout=600
                    )
                    if upload_resp.ok:
                        result = upload_resp.json()
                        output_url = result.get("url", "")

            return {
                "status": "completed",
                "outputPath": output_url,
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
