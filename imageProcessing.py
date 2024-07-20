import cv2
import nltk
from imageToText.app import imageToText
from transformers import pipeline

import pytesseract

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # Update this path if necessary

# Summarization model
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')

# Download the Punkt tokenizer
nltk.download('punkt')


def process_captured_frame(frame):
    text = imageToText('captured_image.jpg')

    # Text summerization code here
    summary = summarizer(text, max_length=200, min_length=120, do_sample=False)[0]['summary_text']

    print("Summary:\n", summary)

# Initialize the camera
cam = cv2.VideoCapture(0)
width, height = 720, 1080

cam.set(3, width)  # Set the width
cam.set(4, height)  # Set the height

# Check if the camera opened successfully
if not cam.isOpened():
    print("Error: Could not open camera.")
else:
    while True:
        # Capture frame-by-frame
        ret, frame = cam.read()

        # If frame is read correctly ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Wait for the 'c' key to be pressed to capture the frame
        if cv2.waitKey(1) == ord('c'):
            # Save the frame as an image file
            cv2.imwrite('captured_image.jpg', frame)
            print("Image captured and saved as 'captured_image.jpg'")
            
            # Call the user-defined function with the captured frame
            process_captured_frame(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()