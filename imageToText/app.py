import cv2
import pytesseract

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # Update this path if necessary

def imageToText(image_path):
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image at {image_path}. Please check the file path and integrity.")
        return None

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to get a binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform OCR using Tesseract
    text = pytesseract.image_to_string(thresh)

    # Print the extracted text
    # print("Extracted Text:\n", text)

    # Optionally, save the processed image for verification
    # cv2.imwrite('sample1.jpeg', thresh)

    return text