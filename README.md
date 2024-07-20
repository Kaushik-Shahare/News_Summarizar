# News Article Summary Chrome Extension & OpenCV Image Processing App

This project combines the power of Natural Language Processing (NLP) and Optical Character Recognition (OCR) to provide a comprehensive tool for summarizing news articles and extracting text from images. It features a Chrome extension for summarizing news articles directly from the web and an OpenCV-based image processing application for text extraction.

## Features

### Text Summarization

- **News Article URL**

  - Utilizes `newspaper` library to fetch articles.
  - **Output**:
    - Title
    - Author
    - Publish Date
    - Summary

- **BlobText**

  - Accepts text blobs for processing.
  - **Output**:
    - Sentiment Analysis (Positive, Negative, or Neutral)

- **PreTrained Model**: Utilizes `bart-large-cnn` model, fine-tuned with approximately 10-20% of CNN/DailyMail 3.0.0 dataset from Kaggle for enhanced accuracy.

### Image to Text Conversion

- **Article from Newspaper**
  - Uses OpenCV for image processing and `pytesseract` for OCR to extract text from images.

### Web Page Text Summarization

- **Particular Text from a Web Page**
  - The Chrome Extension allows users to summarize selected text or entire pages.
  - Uses a fine-tuned `bart-large-cnn` model with data from CNN/DailyMail 3.0.0 dataset, utilizing only about 10-20% of the data for training.

## Chrome Extension Capabilities

- **Selected Text Summarization**: Summarizes the text selected by the user.
- **Current Page Summarization**: Summarizes the entire content of the current page.
- **URL Page Summarization**: Summarizes content from a provided URL.

## Outputs

- **Summarization**: Provides a concise summary of the text or article.
- **Title**: Extracts or generates the title for the summarized content.

## Getting Started

To use this tool, clone the repository, and follow the setup instructions for both the Chrome extension and the OpenCV image processing application. Ensure you have the necessary dependencies installed, including `newspaper`, `OpenCV`, `pytesseract`, and the `transformers` library for the NLP models.

1. **Chrome Extension Setup**:
   - Load the extension in developer mode.
   - Use the extension to summarize text from web pages.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **OpenCV Image Processing**:
   - Run the 'imageProcessing.py' script to extract text from images.
   - Ensure that the text are clearly visible in the images for accurate extraction.
   - The extracted text will be displayed in the terminal.
   ```bash
   python imageProcessing.py
   ```
   - Press 'c' to capture an image and extract text.
   - Press 'q' to quit the application.
4. **Text Summarization**:
   - Run the backend server to summarize text.
   ```bash
    python backend.py
   ```
   - Select the text to summarize on a news article.
   - Use the Chrome extension to summarize text from web pages.
5. Page Summarization:
   - This feature is not available on Chrome Extension yet but api is ready. So feel free to use it by calling the endpoint.
   ```http
   POST /analyze
   ```
   - Use the Chrome extension to summarize the entire content of the current page.
   - Provide a URL to summarize content from a specific page.
