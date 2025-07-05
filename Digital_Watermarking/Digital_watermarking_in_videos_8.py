# prompt: Give any metrics that can be used for evaluating by comparing watermarked image and orignal image and same with watermark and extracted watermark

import numpy as np
from PIL import Image
import cv2

def calculate_metrics(image1_path, image2_path):
    """Calculates MSE, PSNR, SSIM between two images.

    Args:
        image1_path: Path to the first image.
        image2_path: Path to the second image.

    Returns:
        A dictionary containing MSE, PSNR, and SSIM values.
        Returns None if an error occurs during image loading or processing.
    """
    try:
        # Open images using OpenCV
        image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale


        if image1 is None or image2 is None:
            print("Error: Could not load one or both images.")
            return None

        # Resize images to the same size if necessary
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

        # Calculate MSE
        mse = np.mean((image1 - image2) ** 2)

        # Calculate PSNR
        if mse == 0:
          psnr = float('inf')
        else:
          psnr = 20 * np.log10(255.0 / np.sqrt(mse))

        # Calculate SSIM using OpenCV's function
        ssim = cv2.compareHist(cv2.calcHist([image1], [0], None, [256], [0, 256]),
                               cv2.calcHist([image2], [0], None, [256], [0, 256]), cv2.HISTCMP_CORREL)

        return {"MSE": mse, "PSNR": psnr, "SSIM": ssim}
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage for comparing watermarked image and original image
original_image_path = "/content/watermark_results/original.png"  # Replace with the actual path
watermarked_image_path = "/content/watermark_results/watermarked.png"  # Replace with the actual path
metrics_original_watermarked = calculate_metrics(original_image_path, watermarked_image_path)

if metrics_original_watermarked:
  print("Metrics between original image and watermarked image:")
  for metric, value in metrics_original_watermarked.items():
      print(f"{metric}: {value}")

# Example usage for comparing watermark and extracted watermark
original_watermark_path = "/content/watermark_results/watermark.png" # Replace with path to original watermark
extracted_watermark_path = "/content/watermark_results/extracted_watermark.png"  # Replace with the actual path
metrics_watermark_extracted = calculate_metrics(original_watermark_path, extracted_watermark_path)

if metrics_watermark_extracted:
    print("\nMetrics between original watermark and extracted watermark:")
    for metric, value in metrics_watermark_extracted.items():
        print(f"{metric}: {value}")
