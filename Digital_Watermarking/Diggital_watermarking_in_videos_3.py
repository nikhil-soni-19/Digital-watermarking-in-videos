import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Define the model classes first (must match the architecture used during training)
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths to model files
encoder_path = "/content/wm_model/encoder_final.pth"
decoder_path = "/content/wm_model/decoder_final.pth"  # Fixed path

# Path to test image and watermark
image_path = '/content/wm_model/extracted_images/photos_no_class/asparagus-g4c4164115_640.jpg'
watermark_path = '/content/wm.jpeg'  # Path to your watermark image

# Initialize models
encoder = ImprovedEncoder().to(device)
decoder = ImprovedDecoder().to(device)

# Load state dictionaries
encoder.load_state_dict(torch.load(encoder_path, map_location=device))
decoder.load_state_dict(torch.load(decoder_path, map_location=device))

# Set models to evaluation mode
encoder.eval()
decoder.eval()

# Load and preprocess the image
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()  # No normalization as per training code
])

transform_wm = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load images
original_img = Image.open(image_path).convert("RGB")
watermark_img = Image.open(watermark_path).convert("L")  # Load as grayscale

# Transform to tensors
img_tensor = transform_img(original_img).unsqueeze(0).to(device)
wm_tensor = transform_wm(watermark_img).unsqueeze(0).to(device)

# Generate watermarked image and extract watermark
with torch.no_grad():
    watermarked_img = encoder(img_tensor, wm_tensor)
    decoded_wm = decoder(watermarked_img)

# Convert tensors to PIL images for display
to_pil = transforms.ToPILImage()
original_pil = to_pil(img_tensor.squeeze(0).cpu())
watermarked_pil = to_pil(watermarked_img.squeeze(0).cpu())
watermark_pil = to_pil(wm_tensor.squeeze(0).cpu())
decoded_pil = to_pil(decoded_wm.squeeze(0).cpu())

# Display the images
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(original_pil)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Watermarked Image")
plt.imshow(watermarked_pil)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Original Watermark")
plt.imshow(watermark_pil, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Extracted Watermark")
plt.imshow(decoded_pil, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Calculate and display metrics
mse = F.mse_loss(img_tensor, watermarked_img).item()
psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item()

# Resize watermark for comparison
wm_resized = F.interpolate(wm_tensor, size=decoded_wm.shape[2:])
wm_extraction_loss = F.binary_cross_entropy(decoded_wm, wm_resized).item()

print(f"Image PSNR: {psnr:.2f} dB")
print(f"Watermark extraction loss: {wm_extraction_loss:.4f}")

# Save the results
output_dir = "/content/watermark_results"
os.makedirs(output_dir, exist_ok=True)

original_pil.save(os.path.join(output_dir, "original.png"))
watermarked_pil.save(os.path.join(output_dir, "watermarked.png"))
watermark_pil.save(os.path.join(output_dir, "watermark.png"))
decoded_pil.save(os.path.join(output_dir, "extracted_watermark.png"))
