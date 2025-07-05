# prompt: write code to print /content/extracted_watermarks/average_watermark.png as it is. Image was originally gray why are you giving colorful image

from PIL import Image
import matplotlib.pyplot as plt

# Assuming the image exists at the specified path
image_path = "/content/extracted_watermarks/average_watermark.png"

try:
    # Open the image using Pillow library
    img = Image.open(image_path)

    # Display the image
    plt.imshow(img, cmap='gray')  # Use 'gray' cmap to display grayscale image
    plt.axis('off')  # Hide axes
    plt.show()

except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}")
except Exception as e:
    print(f"An error occurred: {e}")
