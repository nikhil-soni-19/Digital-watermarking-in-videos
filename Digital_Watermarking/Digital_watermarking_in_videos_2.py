import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from PIL import Image
import matplotlib.pyplot as plt
import os
import zipfile
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import numpy as np

# ----------- Custom Dataset for Images and Watermarks -----------

class WatermarkDataset(Dataset):
    def __init__(self, image_folder, watermark_path, frame_transform, wm_transform):
        # Check if the folder exists
        if not os.path.exists(image_folder):
            raise ValueError(f"Image folder not found: {image_folder}")

        # Find image files recursively if needed
        self.image_files = []
        if os.path.isdir(image_folder):
            # Check direct files first
            direct_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if len(direct_files) > 0:
                self.image_files = direct_files
            else:
                # Search subdirectories
                for root, _, files in os.walk(image_folder):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_files.append(os.path.join(root, file))

        print(f"Found {len(self.image_files)} image files")

        # Check watermark path
        if not os.path.exists(watermark_path):
            raise ValueError(f"Watermark image not found: {watermark_path}")

        # Load watermark
        try:
            self.watermark = Image.open(watermark_path).convert('L')
        except Exception as e:
            raise ValueError(f"Error loading watermark image: {str(e)}")

        self.frame_transform = frame_transform
        self.wm_transform = wm_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a placeholder image in case of error
            image = Image.new('RGB', (256, 256), color='gray')

        frame = self.frame_transform(image)
        watermark = self.wm_transform(self.watermark)

        return frame, watermark

# ----------- Function to Extract Images from Zip -----------
def extract_images_from_zip(zip_path, extract_to):
    # Create extraction directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)

    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Check if any image files were extracted
    image_files = [f for f in os.listdir(extract_to)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Extracted {len(image_files)} image files to {extract_to}")

    if len(image_files) == 0:
        # If no images found in top directory, check subdirectories
        for root, dirs, files in os.walk(extract_to):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Found images in subdirectory
                    return root  # Return the directory containing images

    return extract_to

# ----------- Improved Models -----------
class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
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
    """Spatial attention module to focus on important image regions"""
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

# ----------- Loss Functions -----------
def perceptual_loss(vgg, x, y):
    return F.mse_loss(vgg(x), vgg(y))

def ssim_loss(x, y, window_size=11):
    C1 = 0.01**2
    C2 = 0.03**2

    # Calculate mean
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size//2)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size//2)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    # Calculate variance and covariance
    sigma_x_sq = F.avg_pool2d(x * x, window_size, stride=1, padding=window_size//2) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(y * y, window_size, stride=1, padding=window_size//2) - mu_y_sq
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size//2) - mu_xy

    # SSIM formula
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

    # Return dissimilarity
    return 1 - ssim_map.mean()

# ----------- Main Training Function -----------
def train_watermark_network(zip_path, watermark_path, output_dir="./model_output",
                          batch_size=8, num_epochs=100, learning_rate=1e-3):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Extract images from zip
    extract_dir = extract_images_from_zip(zip_path, os.path.join(output_dir, "extracted_images"))

    # Transformations
    transform_frame = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    transform_wm = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Create dataset and dataloader
    dataset = WatermarkDataset(extract_dir, watermark_path, transform_frame, transform_wm)

    # Split dataset into train and validation sets (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Dataset contains {len(dataset)} images")
    print(f"Training set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")

    # Initialize models
    encoder = ImprovedEncoder().to(device)
    decoder = ImprovedDecoder().to(device)

    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()

    # Perceptual loss setup
    vgg = models.vgg16(pretrained=True).features[:16].eval().to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, verbose=True)

    # Loss weights
    alpha = 1.0    # MSE content loss weight
    beta = 0.8     # Perceptual loss weight
    gamma = 10.0   # Watermark extraction loss weight
    delta = 0.5    # SSIM loss weight
    epsilon = 0.2  # L1 loss weight

    # Training loop
    best_loss = float('inf')
    train_losses = []
    val_losses = []

    print("Starting training...")
    for epoch in range(num_epochs):
        # Training phase
        encoder.train()
        decoder.train()
        epoch_loss = 0

        # Progress bar
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for frame_batch, watermark_batch in progress_bar:
            # Move to device
            frame_batch = frame_batch.to(device)
            watermark_batch = watermark_batch.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            watermarked_batch = encoder(frame_batch, watermark_batch)
            decoded_batch = decoder(watermarked_batch)

            # Resize watermark for loss calculation
            wm_resized_batch = F.interpolate(watermark_batch, size=decoded_batch.shape[2:])

            # Calculate losses
            content_loss = mse_loss(watermarked_batch, frame_batch)
            perception_loss = perceptual_loss(vgg, watermarked_batch, frame_batch)
            structure_loss = ssim_loss(watermarked_batch, frame_batch)
            sharpness_loss = l1_loss(watermarked_batch, frame_batch)
            extraction_loss = bce_loss(decoded_batch, wm_resized_batch)

            # Combine losses
            batch_loss = (alpha * content_loss +
                        beta * perception_loss +
                        gamma * extraction_loss +
                        delta * structure_loss +
                        epsilon * sharpness_loss)

            # Backward pass and optimization
            batch_loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{batch_loss.item():.4f}",
                'content': f"{content_loss.item():.4f}",
                'extract': f"{extraction_loss.item():.4f}"
            })

            epoch_loss += batch_loss.item()

        # Calculate average epoch loss
        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation phase
        encoder.eval()
        decoder.eval()
        val_loss = 0

        with torch.no_grad():
            for frame_batch, watermark_batch in val_dataloader:
                # Move to device
                frame_batch = frame_batch.to(device)
                watermark_batch = watermark_batch.to(device)

                # Forward pass
                watermarked_batch = encoder(frame_batch, watermark_batch)
                decoded_batch = decoder(watermarked_batch)

                # Resize watermark for loss calculation
                wm_resized_batch = F.interpolate(watermark_batch, size=decoded_batch.shape[2:])

                # Calculate losses
                content_loss = mse_loss(watermarked_batch, frame_batch)
                perception_loss = perceptual_loss(vgg, watermarked_batch, frame_batch)
                structure_loss = ssim_loss(watermarked_batch, frame_batch)
                sharpness_loss = l1_loss(watermarked_batch, frame_batch)
                extraction_loss = bce_loss(decoded_batch, wm_resized_batch)

                # Combine losses
                batch_loss = (alpha * content_loss +
                            beta * perception_loss +
                            gamma * extraction_loss +
                            delta * structure_loss +
                            epsilon * sharpness_loss)

                val_loss += batch_loss.item()

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        # Update learning rate
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Check early stopping
        if early_stopping(avg_val_loss, encoder, decoder):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Save checkpoints every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth"))

            # Visualize some results every 10 epochs
            visualize_samples(encoder, decoder, dataset, device, epoch, output_dir)

    # Load the best model (saved by early stopping)
    early_stopping.load_best_model(encoder, decoder)

    # Save final model
    torch.save(encoder.state_dict(), os.path.join(output_dir, "encoder_final.pth"))
    torch.save(decoder.state_dict(), os.path.join(output_dir, "decoder_final.pth"))

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_validation_loss.png"))

    return encoder, decoder, dataset


# ----------- Visualization Functions -----------
def visualize_samples(encoder, decoder, dataset, device, epoch, output_dir):
    """Visualize watermarking results for sample images"""
    # Create evaluation directory
    eval_dir = os.path.join(output_dir, f"eval_epoch_{epoch+1}")
    os.makedirs(eval_dir, exist_ok=True)

    # Select random samples
    num_samples = min(5, len(dataset))
    sample_indices = random.sample(range(len(dataset)), num_samples)

    # Set models to eval mode
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            # Get sample
            frame, watermark = dataset[idx]
            frame = frame.unsqueeze(0).to(device)
            watermark = watermark.unsqueeze(0).to(device)

            # Process
            watermarked = encoder(frame, watermark)
            decoded = decoder(watermarked)

            # Resize watermark for visualization
            wm_resized = F.interpolate(watermark, size=decoded.shape[2:])

            # Convert to images
            to_pil = transforms.ToPILImage()

            original_img = to_pil(frame[0].cpu())
            watermarked_img = to_pil(watermarked[0].cpu())
            watermark_img = to_pil(watermark[0].cpu())
            decoded_img = to_pil(decoded[0].cpu())

            # Save individual images
            original_img.save(os.path.join(eval_dir, f"sample_{i}_original.png"))
            watermarked_img.save(os.path.join(eval_dir, f"sample_{i}_watermarked.png"))
            watermark_img.save(os.path.join(eval_dir, f"sample_{i}_watermark.png"))
            decoded_img.save(os.path.join(eval_dir, f"sample_{i}_decoded.png"))

            # Create and save comparison figure
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))

            axs[0].imshow(original_img)
            axs[0].set_title("Original Image")

            axs[1].imshow(watermarked_img)
            axs[1].set_title("Watermarked Image")

            axs[2].imshow(watermark_img, cmap='gray')
            axs[2].set_title("Original Watermark")

            axs[3].imshow(decoded_img, cmap='gray')
            axs[3].set_title("Extracted Watermark")

            for ax in axs:
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(eval_dir, f"sample_{i}_comparison.png"))
            plt.close()

            # Calculate metrics
            mse = F.mse_loss(frame, watermarked).item()
            # Fix: Convert float to tensor for torch.log10
            psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item()
            wm_accuracy = F.binary_cross_entropy(decoded, wm_resized).item()

            # Save metrics
            with open(os.path.join(eval_dir, f"sample_{i}_metrics.txt"), 'w') as f:
                f.write(f"PSNR: {psnr:.2f} dB\n")
                f.write(f"Watermark extraction loss: {wm_accuracy:.4f}\n")

    # Set models back to training mode
    encoder.train()
    decoder.train()

# ----------- Test Robustness -----------
def test_robustness(encoder, decoder, dataset, device, output_dir="./robustness_test"):
    """Test watermark extraction after various attacks"""
    os.makedirs(output_dir, exist_ok=True)

    # Set models to eval mode
    encoder.eval()
    decoder.eval()

    # Select random samples
    num_samples = min(3, len(dataset))
    sample_indices = random.sample(range(len(dataset)), num_samples)

    # Define attacks
    def jpeg_compression(img, quality=80):
        # Convert to PIL, save as JPEG, load back
        pil_img = transforms.ToPILImage()(img.cpu())
        jpeg_path = os.path.join(output_dir, "temp.jpg")
        pil_img.save(jpeg_path, quality=quality)
        compressed = transforms.ToTensor()(Image.open(jpeg_path))
        os.remove(jpeg_path)
        return compressed.unsqueeze(0).to(device)

    def add_noise(img, std=0.05):
        return torch.clamp(img + std * torch.randn_like(img), 0, 1)

    def adjust_brightness(img, factor=1.2):
        return torch.clamp(img * factor, 0, 1)

    def crop_image(img, ratio=0.8):
        h, w = img.shape[2], img.shape[3]
        h_new, w_new = int(h * ratio), int(w * ratio)
        cropped = img[:, :, :h_new, :w_new]
        resized = F.interpolate(cropped, size=(h, w), mode='bilinear')
        return resized

    attacks = {
        "Original": lambda x: x,
        "JPEG Compression": jpeg_compression,
        "Gaussian Noise": add_noise,
        "Brightness Change": adjust_brightness,
        "Cropping": crop_image
    }

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            # Get sample
            frame, watermark = dataset[idx]
            frame = frame.unsqueeze(0).to(device)
            watermark = watermark.unsqueeze(0).to(device)

            # Generate watermarked image
            watermarked = encoder(frame, watermark)

            # Create sample directory
            sample_dir = os.path.join(output_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)

            # Save original images
            transforms.ToPILImage()(frame[0].cpu()).save(os.path.join(sample_dir, "original.png"))
            transforms.ToPILImage()(watermark[0].cpu()).save(os.path.join(sample_dir, "watermark.png"))
            transforms.ToPILImage()(watermarked[0].cpu()).save(os.path.join(sample_dir, "watermarked.png"))

            # Set up visualization
            fig, axs = plt.subplots(len(attacks), 2, figsize=(8, 3*len(attacks)))

            # Apply each attack and test extraction
            for j, (attack_name, attack_fn) in enumerate(attacks.items()):
                # Apply attack
                attacked_img = attack_fn(watermarked)

                # Extract watermark
                extracted_wm = decoder(attacked_img)

                # Save images
                attacked_img_pil = transforms.ToPILImage()(attacked_img[0].cpu())
                extracted_wm_pil = transforms.ToPILImage()(extracted_wm[0].cpu())

                attacked_img_pil.save(os.path.join(sample_dir, f"{attack_name.lower().replace(' ', '_')}.png"))
                extracted_wm_pil.save(os.path.join(sample_dir, f"{attack_name.lower().replace(' ', '_')}_extracted.png"))

                # Display in figure
                axs[j, 0].imshow(attacked_img_pil)
                axs[j, 0].set_title(f"{attack_name}")
                axs[j, 0].axis('off')

                axs[j, 1].imshow(extracted_wm_pil, cmap='gray')
                axs[j, 1].set_title("Extracted Watermark")
                axs[j, 1].axis('off')

                # Calculate extraction quality
                wm_resized = F.interpolate(watermark, size=extracted_wm.shape[2:])
                extraction_loss = F.binary_cross_entropy(extracted_wm, wm_resized).item()

                # Save metrics
                with open(os.path.join(sample_dir, f"{attack_name.lower().replace(' ', '_')}_metrics.txt"), 'w') as f:
                    f.write(f"Extraction Loss: {extraction_loss:.4f}\n")

            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, "robustness_summary.png"))
            plt.close()

# ----------- Main Execution -----------
if __name__ == "__main__":
    # Parameters
    zip_path = "/content/archive (3).zip"
    watermark_path = "/content/wm.jpeg"
    output_dir = "/content/wm_model"
    batch_size = 8
    epochs = 20
    learning_rate = 1e-3
    test_robustness_flag = True

    # Train the models
    encoder, decoder, dataset = train_watermark_network(
        zip_path,
        watermark_path,
        output_dir,
        batch_size,
        epochs,
        learning_rate
    )

    # Test robustness if requested
    if test_robustness_flag:
        test_robustness(encoder, decoder, dataset, torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                      os.path.join(output_dir, "robustness_test"))

    print(f"Training complete. Models saved to {output_dir}")