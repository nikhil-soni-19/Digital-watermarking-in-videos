import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Define the model classes
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels//8, 1, kernel_size=1)

    def forward(self, x):
        # Generate attention map
        attn = F.relu(self.conv1(x))
        attn = torch.sigmoid(self.conv2(attn))

        # Apply attention
        return x * attn.expand_as(x)

class ImprovedDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        # Attention mechanism
        self.attention = AttentionModule(64)

        # Downsampling to watermark size
        self.downsampling = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        # Final watermark extraction
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()  # Bounded output for watermark
        )

    def forward(self, x):
        # Extract features
        features = self.feature_extraction(x)
        residual_features = self.residual_blocks(features)
        attended_features = self.attention(residual_features)

        # Downsample to approximate watermark size
        downsampled = self.downsampling(attended_features)

        # Extract watermark
        extracted_wm = self.final(downsampled)

        return extracted_wm

def extract_watermark_from_video(video_path, decoder_path, output_dir, frame_interval=1,
                               batch_size=8, watermark_size=(64, 64), frame_size=(256, 256)):
    """
    Extract watermarks from frames of a watermarked video

    Args:
        video_path: Path to the watermarked video
        decoder_path: Path to the saved decoder model
        output_dir: Directory to save extracted watermarks
        frame_interval: Process every Nth frame
        batch_size: Number of frames to process at once
        watermark_size: Expected size of the watermark
        frame_size: Size to resize video frames
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load decoder model
    decoder = ImprovedDecoder().to(device)
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    decoder.eval()
    print("Decoder model loaded successfully")

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video properties:")
    print(f"- FPS: {fps}")
    print(f"- Total frames: {total_frames}")
    print(f"- Duration: {duration:.2f} seconds")
    print(f"- Processing every {frame_interval} frame(s)")

    # Define transform for frames
    transform = transforms.Compose([
        transforms.Resize(frame_size),
        transforms.ToTensor()
    ])

    # Process frames in batches
    frames_to_process = range(0, total_frames, frame_interval)
    num_batches = (len(frames_to_process) + batch_size - 1) // batch_size

    # Create a summary image to show watermark extraction over time
    summary_width = min(10, len(frames_to_process))
    summary_height = (len(frames_to_process) + summary_width - 1) // summary_width
    summary_fig, summary_axs = plt.subplots(summary_height, summary_width,
                                          figsize=(summary_width*2, summary_height*2))
    summary_axs = summary_axs.flatten() if isinstance(summary_axs, np.ndarray) else [summary_axs]

    # Initialize variables for average watermark
    all_watermarks = []

    # Process video
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Extracting watermarks"):
            batch_frames = []
            batch_timestamps = []

            # Collect frames for this batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(frames_to_process))

            for i in range(start_idx, end_idx):
                frame_idx = frames_to_process[i]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    print(f"Warning: Could not read frame {frame_idx}")
                    continue

                # Convert frame to RGB and apply transform
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_tensor = transform(frame_pil)

                batch_frames.append(frame_tensor)
                timestamp = frame_idx / fps
                batch_timestamps.append(timestamp)

            if not batch_frames:
                continue

            # Stack frames and move to device
            frames_tensor = torch.stack(batch_frames).to(device)

            # Extract watermarks
            extracted_watermarks = decoder(frames_tensor)

            # Process each extracted watermark
            for j, (watermark, timestamp, frame_idx) in enumerate(zip(
                extracted_watermarks, batch_timestamps, range(start_idx, end_idx))):

                # Convert to PIL image for saving
                watermark_np = watermark.squeeze(0).cpu().numpy()
                all_watermarks.append(watermark_np)

                # Save watermark image
                watermark_pil = Image.fromarray((watermark_np * 255).astype(np.uint8))
                watermark_pil = watermark_pil.resize(watermark_size, Image.LANCZOS)
                watermark_path = os.path.join(output_dir, f"watermark_frame_{frames_to_process[frame_idx]:05d}.png")
                watermark_pil.save(watermark_path)

                # Add to summary plot
                if frame_idx < len(summary_axs):
                    summary_axs[frame_idx].imshow(watermark_np, cmap='gray')
                    summary_axs[frame_idx].set_title(f"Frame {frames_to_process[frame_idx]}\n{timestamp:.2f}s")
                    summary_axs[frame_idx].axis('off')

    # Hide any unused subplots
    for i in range(len(frames_to_process), len(summary_axs)):
        summary_axs[i].axis('off')

    # Save summary figure
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "watermark_extraction_summary.png")
    plt.savefig(summary_path)
    plt.close()

    # Calculate and save average watermark
    if all_watermarks:
        avg_watermark = np.mean(all_watermarks, axis=0)
        avg_watermark_pil = Image.fromarray((avg_watermark * 255).astype(np.uint8))
        avg_watermark_pil = avg_watermark_pil.resize(watermark_size, Image.LANCZOS)
        avg_watermark_path = os.path.join(output_dir, "average_watermark.png")
        avg_watermark_pil.save(avg_watermark_path)

        # Display average watermark
        plt.figure(figsize=(5, 5))
        plt.imshow(avg_watermark, cmap='gray')
        plt.title("Average Extracted Watermark")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "average_watermark_display.png"))
        plt.close()

    # Release video capture
    cap.release()

    print(f"Watermark extraction complete. Results saved to {output_dir}")
    return os.path.join(output_dir, "average_watermark.png")

def compare_with_original_watermark(extracted_path, original_path, output_dir):
    """
    Compare the extracted watermark with the original watermark

    Args:
        extracted_path: Path to the extracted watermark
        original_path: Path to the original watermark
        output_dir: Directory to save comparison results
    """
    # Load watermarks
    extracted = Image.open(extracted_path).convert('L')
    original = Image.open(original_path).convert('L')

    # Resize original to match extracted if needed
    original = original.resize(extracted.size, Image.LANCZOS)

    # Convert to numpy arrays
    extracted_np = np.array(extracted) / 255.0
    original_np = np.array(original) / 255.0

    # Calculate metrics
    mse = np.mean((extracted_np - original_np) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

    # Create comparison figure
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_np, cmap='gray')
    plt.title("Original Watermark")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(extracted_np, cmap='gray')
    plt.title("Extracted Watermark")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    diff = np.abs(original_np - extracted_np)
    plt.imshow(diff, cmap='hot')
    plt.title("Difference")
    plt.axis('off')
    plt.colorbar(shrink=0.7)

    plt.suptitle(f"Watermark Comparison (PSNR: {psnr:.2f} dB)")
    plt.tight_layout()

    # Save comparison
    comparison_path = os.path.join(output_dir, "watermark_comparison.png")
    plt.savefig(comparison_path)
    plt.close()

    # Save metrics
    with open(os.path.join(output_dir, "comparison_metrics.txt"), 'w') as f:
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"PSNR: {psnr:.2f} dB\n")

    print(f"Comparison complete. Results saved to {output_dir}")
    print(f"PSNR: {psnr:.2f} dB")

    return psnr

def main():
    # Paths
    video_path = "/content/watermarked_video.mp4"  # Path to watermarked video
    decoder_path = "/content/wm_model/decoder_final.pth"  # Path to decoder model
    original_watermark_path = "/content/wm.jpeg"  # Path to original watermark
    output_dir = "/content/extracted_watermarks"  # Directory to save results

    # Extract watermarks
    avg_watermark_path = extract_watermark_from_video(
        video_path=video_path,
        decoder_path=decoder_path,
        output_dir=output_dir,
        frame_interval=10,  # Process every 10th frame for speed
        batch_size=8,
        watermark_size=(64, 64),
        frame_size=(256, 256)
    )

    # Compare with original watermark
    compare_with_original_watermark(
        extracted_path=avg_watermark_path,
        original_path=original_watermark_path,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()
