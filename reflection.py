import cv2
import numpy as np

def remove_glass_reflections(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(filtered, 30, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the detected contours
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, -1)

    # Inpainting to remove reflections
    result = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return result

# Example usage
input_image_path = 'input_image.jpg'
output_image = remove_glass_reflections(input_image_path)
cv2.imwrite('output_image_without_reflections.jpg', output_image)
