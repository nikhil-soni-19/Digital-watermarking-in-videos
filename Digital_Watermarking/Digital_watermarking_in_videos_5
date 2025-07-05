import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import os

# Define the model classes first
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

class ImprovedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial feature extraction
        self.init_conv = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        # Residual blocks for deeper feature extraction
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        # Attention mechanism to focus on important regions
        self.attention = AttentionModule(64)

        # Final layers to produce watermarked image
        self.final_layers = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 1),
            nn.Tanh()  # Bounded output
        )

        # Skip connection from input
        self.alpha = nn.Parameter(torch.tensor(0.8))  # Learnable alpha for visibility control

    def forward(self, frame, wm):
        # Resize watermark to match frame dimensions
        wm_resized = F.interpolate(wm, size=(frame.shape[2], frame.shape[3]))

        # Concatenate frame and watermark
        x = torch.cat([frame, wm_resized], dim=1)

        # Extract features
        features = self.init_conv(x)
        residual_features = self.residual_blocks(features)
        attended_features = self.attention(residual_features)

        # Generate residual watermark pattern
        watermark_residual = self.final_layers(attended_features)

        # Add residual to original image with learnable strength
        watermarked = frame * self.alpha + watermark_residual * (1 - self.alpha)

        # Ensure output is in valid range [0, 1]
        return torch.clamp(watermarked, 0, 1)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path to model file
encoder_path = "/content/wm_model/encoder_final.pth"

# Define transformations
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

transform_wm = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Initialize and load encoder model
encoder = ImprovedEncoder().to(device)
encoder.load_state_dict(torch.load(encoder_path, map_location=device))
encoder.eval()  # Set to evaluation mode

# ----------- Video Processing -----------
input_video_path = "/content/Video01_Seahorse.mp4"  # Replace with your video path
output_video_path = "/content/watermarked_video.mp4"
watermark_path = "/content/wm.jpeg"  # Replace with your watermark path

# Create output directory if it doesn't exist
output_dir = os.path.dirname(output_video_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load watermark
try:
    watermark_img = Image.open(watermark_path).convert("L")
    wm_tensor = transform_wm(watermark_img).unsqueeze(0).to(device)
    print(f"Watermark loaded with shape: {wm_tensor.shape}")
except Exception as e:
    print(f"Error loading watermark: {str(e)}")
    exit(1)

# Open the video file
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {input_video_path}")
    exit(1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video properties: {width}x{height} at {fps} FPS, {total_frames} frames")

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' for better compatibility
out = cv2.VideoWriter(output_video_path, fourcc, fps, (256, 256))  # Using 256x256 as per transform

if not out.isOpened():
    print(f"Error: Could not create output video file {output_video_path}")
    cap.release()
    exit(1)

# Process video frame by frame
frame_count = 0
with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Resize and transform the frame
        frame_tensor = transform_img(frame_pil).unsqueeze(0).to(device)

        # Watermark the frame
        watermarked_frame = encoder(frame_tensor, wm_tensor)

        # Convert back to OpenCV format
        watermarked_np = watermarked_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        watermarked_np = (np.clip(watermarked_np, 0, 1) * 255).astype(np.uint8)
        watermarked_bgr = cv2.cvtColor(watermarked_np, cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(watermarked_bgr)

        # Update progress
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")

# Release resources
cap.release()
out.release()

print(f"Watermarked video saved to {output_video_path}")
print(f"Processed {frame_count} frames in total")

# Optional: Display a sample frame to verify watermarking
sample_frame = cv2.imread(output_video_path.replace('.mp4', '_sample.jpg'))
if sample_frame is not None:
    cv2.imwrite(output_video_path.replace('.mp4', '_sample.jpg'), watermarked_bgr)
    print(f"Sample frame saved to {output_video_path.replace('.mp4', '_sample.jpg')}")
