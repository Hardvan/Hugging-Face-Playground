from transformers import pipeline
# from diffusers import StableDiffusionPipeline
# import torch
from dotenv import load_dotenv
import os

# load the .env file
load_dotenv()


def sentiment_analysis(text):

    classifier = pipeline('sentiment-analysis')
    result = classifier(text)
    return result


def diffuser_sentiment_analysis(text):
    pass


def test_sentiment_analysis():

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
    file_name = "test_sentiment_analysis.md"
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


if __name__ == "__main__":
    test_sentiment_analysis()
