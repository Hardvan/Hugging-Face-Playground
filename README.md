# Hugging Face Playground

The `huggingface.py` contains various transformers from huggingface, which uses the `pipeline` function to perform various tasks such as text generation, text summarization, text translation, etc.

## Tasks Performed

### 1. Sentiment Analysis

The `sentiment_analysis` function evaluates the sentiment of given textual input. It returns a list of dictionaries, each containing the sentiment label and score.

View the output [here](./sentiment_analysis.md)

### 2. Text Summarization

The `summarize_text` function generates a summary for a given text using the BART-large-CNN model. The result is a list of dictionaries containing the summary text.

View the output [here](./text_summarization.md)

### 3. Depth Estimation

The `depth_estimate` function performs depth estimation on an input image using the Intel DPT-large model. The output is a depth image, and the user can specify the output path.

View the output [here](./depth_estimation.md)

### 4. Object Detection

The `detect_objects` function conducts object detection on an input image using the DETR model. The output is a dictionary containing information about detected objects, including their names, confidence scores, and locations.

View the output [here](./detect_objects.md)
