import torch, torch.nn as nn, torch.nn.functional as F
import cv2, os
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ----------- Model Classes (same as training) -----------
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

# ----------- Transforms -----------
transform_frame = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
transform_wm = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# ----------- Paths -----------
input_video_path = "/content/Video01_Seahorse.mp4"
output_video_path = "/content/watermarked_video.mp4"
watermark_path = "/content/sample_wm.jpg"
output_debug_folder = "/content/frames_debug"
os.makedirs(output_debug_folder, exist_ok=True)

# ----------- Load Models -----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(device)
decoder = Decoder().to(device)
encoder.load_state_dict(torch.load("/content/wm_model/encoder_final.pth", map_location=device))
decoder.load_state_dict(torch.load("/content/wm_model/decoder_final.pth", map_location=device))
encoder.eval(); decoder.eval()

# ----------- Load Watermark Image (grayscale) -----------
wm = transform_wm(Image.open(watermark_path).convert("L")).unsqueeze(0).to(device)  # (1, 1, 32, 32)

# ----------- Video Setup -----------
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (128, 128)
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Try 'avc1' or 'XVID' if 'mp4v' fails
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

frame_count = 0
with torch.no_grad():
    while cap.isOpened():
        ret, frame_raw = cap.read()
        if not ret or frame_raw is None:
            break

        # Resize & Preprocess
        frame_pil = Image.fromarray(cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)).resize((128, 128))
        frame_tensor = transform_frame(frame_pil).unsqueeze(0).to(device)

        # Encode watermark
        watermarked = encoder(frame_tensor, wm)
        watermarked_np = watermarked.squeeze(0).permute(1, 2, 0).cpu().numpy()
        watermarked_np = (np.clip(watermarked_np, 0, 1) * 255).astype(np.uint8)

        # Write to video
        if watermarked_np.shape[0:2] == (128, 128):
            out.write(cv2.cvtColor(watermarked_np, cv2.COLOR_RGB2BGR))

        # Save some debug frames
        if frame_count % 10 == 0:
            debug_frame = transforms.ToPILImage()(watermarked.squeeze(0).cpu())
            debug_frame.save(f"{output_debug_folder}/wm_frame_{frame_count:03d}.png")

        frame_count += 1

cap.release()
out.release()

# ----------- Check Video File Saved -----------
if os.path.exists(output_video_path):
    print(f"✅ Watermarked video saved to: {output_video_path}")
else:
    print("❌ OpenCV VideoWriter failed. Using ffmpeg fallback.")
    os.system(f"ffmpeg -y -r {fps} -i {output_debug_folder}/wm_frame_%03d.png -vcodec libx264 {output_video_path}")
    print(f"✅ Video saved using ffmpeg at: {output_video_path}")

# ----------- Decode & Show Sample Watermark -----------
test_img = Image.open(f"{output_debug_folder}/wm_frame_010.png").convert("RGB")
test_tensor = transform_frame(test_img).unsqueeze(0).to(device)

with torch.no_grad():
    decoded = decoder(test_tensor)

plt.imshow(decoded.squeeze().cpu().numpy(), cmap='gray')
plt.title("Decoded Watermark"); plt.axis('off'); plt.show()
