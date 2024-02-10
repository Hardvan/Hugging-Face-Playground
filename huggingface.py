"""
This module contains functions to run sentiment analysis, text summarization,
depth estimation, and object detection using the Hugging Face Transformers library.

Functions:
    sentiment_analysis: Run sentiment analysis on the input text.
    summarize_text: Run summarization on the input text.
    depth_estimate: Run depth estimation on the input image.
    detect_objects: Run object detection on the input image.
"""


from transformers import pipeline, DPTImageProcessor, DPTForDepthEstimation, DetrImageProcessor, DetrForObjectDetection, ViTImageProcessor, ViTForImageClassification
import torch
import numpy as np
from PIL import Image, ImageDraw
import requests


def sentiment_analysis(text):
    """Run sentiment analysis on the input text.

    Args:
        text (str): The input text to analyze.

    Returns:
        list: A list of dictionaries containing the label and score of the sentiment analysis.
            Structure: [{'label': '...', 'score': ...}]
    """

    classifier = pipeline('sentiment-analysis')
    result = classifier(text)
    return result  # [{'label': '...', 'score': ...}]


def summarize_text(text):
    """Run summarization on the input text.

    Args:
        text (str): The input text to summarize.

    Returns:
        list: A list of dictionaries containing the summary of the input text.
            Structure: [{'summary_text': '...'}]
    """

    summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
    result = summarizer(text, min_length=30, max_length=100, do_sample=False)
    return result  # [{'summary_text': '...'}]


def depth_estimate(image_url):
    """Run depth estimation on the input image.

    Args:
        image_url (str): The URL of the input image to estimate depth.
        output_path (str): The file path to save the depth image. Defaults to "depth_estimate.jpg".

    Returns:
        Image: The depth image.
    """

    image = Image.open(requests.get(image_url, stream=True).raw)

    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    # prepare image for the model
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)

    return depth


def draw_bounding_box(image, box, object_name, color="green", thickness=5):
    """Draw a bounding box around an object in the input image.

    Args:
        image (Image): The input image to draw the bounding box on.
        box (list): The coordinates of the bounding box in the format [x_min, y_min, x_max, y_max].
        object_name (str): The name of the object to include in the outline.
        color (str): The color of the bounding box. Defaults to "green".
        thickness (int): The thickness of the bounding box. Defaults to 5.

    Returns:
        Image: The input image with the bounding box drawn around the object.
    """

    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=color, width=thickness)

    # Include the name of the object in the outline
    draw.text((box[0] + 10, box[1] + 10), object_name, fill=color)

    return image


def detect_objects(image_url):
    """Run object detection on the input image.

    Args:
        image_url (str): The URL of the input image to detect objects.

    Returns:
        dict: A dictionary containing the object detection results.
            Structure: {'Object Detection Results': [{'Object': '...', 'Confidence': ..., 'Location': '...'}]}
        image (Image): The input image with bounding boxes drawn around the detected objects.
    """

    image = Image.open(requests.get(image_url, stream=True).raw)
    image_copy = image.copy()

    # you can specify the revision tag if you don't want the timm dependency
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9)[0]

    results_dict = {
        "Object Detection Results": []
    }

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        object_name = model.config.id2label[label.item()]
        confidence = round(score.item(), 3)
        location = str(box)
        results_dict["Object Detection Results"].append({
            "Object": object_name,
            "Confidence": confidence,
            "Location": location
        })

        # Draw bounding boxes around the detected objects in image_copy
        image_copy = draw_bounding_box(image_copy, box, object_name)

    return results_dict, image_copy


def classify_image(image_url):
    """Run image classification on the input image.

    Args:
        image_url (str): The URL of the input image to classify.

    Returns:
    """

    image = Image.open(requests.get(image_url, stream=True).raw)

    processor = ViTImageProcessor.from_pretrained(
        "google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    # Get the top 3 class labels and scores
    top_k = torch.topk(logits, k=3)
    top_classes = top_k.indices[0]
    top_scores = top_k.values[0]

    predicted_classes = [model.config.id2label[class_idx.item()]
                         for class_idx in top_classes]

    # Convert scores to percentages
    total_score = top_scores.sum().item()
    scores_percentages = [(score.item() / total_score)
                          * 100 for score in top_scores]

    return predicted_classes, scores_percentages


if __name__ == "__main__":

    import time

    def test_sentiment_analysis():

        print("Testing sentiment analysis...")

        start = time.time()

        # Sample input texts
        input_texts = [
            "I love this product.",
            "I hate this product.",
            "I am not sure about this product.",
            "I am feeling happy.",
            "I am feeling sad.",
            "I am feeling neutral.",
            "The product has a good quality, but is too expensive.",
            "The product is cheap, but has a bad quality.",
            "The product is neither good nor bad."
        ]

        # Run sentiment analysis on the input texts
        output = []
        for text in input_texts:
            result = sentiment_analysis(text)
            output.append({
                "input_text": text,
                "label": result[0]['label'],
                "score": result[0]['score']
            })

        # Save the results to a markdown file
        file_name = "test/sentiment_analysis.md"
        file_heading = "# Test Sentiment Analysis\n\n"
        table_header = "| Input text | Label | Score |\n"
        table_divider = "| --- | --- | --- |\n"
        table_rows = ""
        for result in output:
            row = f"| {result['input_text']} | {result['label']} | {result['score']} |\n"
            table_rows += row
        markdown_table = file_heading + table_header + table_divider + table_rows
        with open(file_name, "w") as file:
            file.write(markdown_table)

        end = time.time()
        print(
            f"✅ Test Sentiment Analysis completed in {end - start:.2f} seconds.")

    def test_summarize_text():

        print("Testing summarize text...")

        start = time.time()

        # Input news stories
        news_stories = [
            """New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
    """
        ]

        # Run summarization on the news stories
        output = []
        for story in news_stories:
            result = summarize_text(story)
            output.append({
                "input_text": story,
                "summary_text": result[0]['summary_text']
            })

        # Save the results to a markdown file
        file_name = "test/summarize_text.md"
        with open(file_name, "w") as file:
            for result in output:
                file.write("# Test Summarize Text\n\n")
                input_text = result['input_text']
                input_size = len(input_text.split(" "))
                summary_text = result['summary_text']
                summary_size = len(summary_text.split(" "))
                file.write("## Input Text\n\n")
                file.write(f"{input_text}\n\n")
                file.write("## Input Size\n\n")
                file.write(f"{input_size} words\n\n")
                file.write("## Summary\n\n")
                file.write(f"{summary_text}\n\n")
                file.write("## Summary Size\n\n")
                file.write(f"{summary_size} words\n\n")

        end = time.time()
        print(f"✅ Test Summarize Text completed in {end - start:.2f} seconds.")

    def test_depth_estimate():

        print("Testing depth estimate...")

        start = time.time()

        # Sample input image
        image_url = "https://wallpapers.com/images/featured-full/mountain-t6qhv1lk4j0au09t.jpg"

        # Run depth estimation on the input image
        result_image = depth_estimate(image_url)
        result_image.save("test/depth_estimate.jpg")

        # Save the depth image in a markdown file
        file_name = "test/depth_estimate.md"
        with open(file_name, "w") as file:
            file.write("# Test Depth Estimate\n\n")
            file.write("## Input Image\n\n")
            file.write(f"![Input Image]({image_url})\n\n")
            file.write("## Depth Image\n\n")
            file.write("![Depth Image](depth_estimate.jpg)\n\n")

        end = time.time()
        print(f"✅ Test Depth Estimate completed in {end - start:.2f} seconds.")

    def test_detect_objects():

        print("Testing detect objects...")

        start = time.time()

        # Sample input image
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

        # Run object detection on the input image
        results, box_image = detect_objects(image_url)
        results = results["Object Detection Results"]
        box_image.save("test/detect_objects.png")

        # Save the results to a markdown file
        file_name = "test/detect_objects.md"
        with open(file_name, "w") as file:
            file.write("# Test Detect Objects\n\n")
            file.write("## Input Image\n\n")
            file.write(f"![Input Image]({image_url})\n\n")
            file.write("## Output Image\n\n")
            file.write("![Output Image](detect_objects.png)\n\n")
            file.write("## Object Detection Results\n\n")
            file.write("| Object | Confidence | Location |\n")
            file.write("|--------|------------|----------|\n")
            for result in results:
                object_name = result["Object"]
                confidence = result["Confidence"]
                location = result["Location"]
                file.write(f"| {object_name} | {confidence} | {location} |\n")

        end = time.time()
        print(f"✅ Test Detect Objects completed in {end - start:.2f} seconds.")

    def test_classify_image():

        print("Testing classify image...")

        start = time.time()

        # Sample input image
        image_url = "https://img.freepik.com/free-photo/view-old-tree-lake-with-snow-covered-mountains-cloudy-day_181624-28954.jpg?w=1060&t=st=1707560650~exp=1707561250~hmac=3a61d03b525ab28fb6b50b08cf1e60218f86c0ad3adc863600b8086f80c627c0"

        # Run image classification on the input image
        predicted_classes, scores_percentages = classify_image(image_url)

        # Save the results to a markdown file
        file_name = "test/classify_image.md"
        with open(file_name, "w") as file:
            file.write("# Test Classify Image\n\n")
            file.write("## Input Image\n\n")
            file.write(f"![Input Image]({image_url})\n\n")
            file.write("## Predicted Classes\n\n")
            file.write("| Class | Confidence |\n")
            file.write("|-------|-------|\n")
            for i, (class_name, score) in enumerate(zip(predicted_classes, scores_percentages), 1):
                file.write(f"| {class_name} | {score:.2f}% |\n")

        end = time.time()
        print(f"✅ Test Classify Image completed in {end - start:.2f} seconds.")

    # test_sentiment_analysis()
    # test_summarize_text()
    # test_depth_estimate()
    # test_detect_objects()
    test_classify_image()
