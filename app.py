from flask import Flask, render_template, request
import time
import base64
from io import BytesIO
from PIL import Image
import requests

# Custom modules
import huggingface


app = Flask(__name__)


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/sentiment_analysis', methods=['GET', 'POST'])
def sentiment_analysis():

    if request.method == 'POST':

        text = request.form['sentence']
        start = time.time()
        result = huggingface.sentiment_analysis(
            text)[0]  # {'label': '...', 'score': ...}
        end = time.time()

        elapsed_time = round(end - start, 2)  # in seconds

        result['sentence'] = text
        result['score'] = round(result['score'], 4)
        result['elapsed_time'] = elapsed_time

        return render_template("sentiment_analysis.html", result=result)

    return render_template("sentiment_analysis.html")


@app.route('/text_summarization', methods=['GET', 'POST'])
def text_summarization():

    if request.method == 'POST':

        text = request.form['text']
        start = time.time()
        result = huggingface.summarize_text(text)[0]  # {'summary_text': '...'}
        end = time.time()

        elapsed_time = round(end - start, 2)  # in seconds

        result['text'] = text
        result['elapsed_time'] = elapsed_time

        return render_template("text_summarization.html", result=result)

    return render_template("text_summarization.html")


@app.route('/depth_estimation', methods=['GET', 'POST'])
def depth_estimation():

    if request.method == 'POST':

        image_url = request.form['image_url']

        start = time.time()
        depth_image = huggingface.depth_estimate(image_url)
        end = time.time()

        elapsed_time = round(end - start, 2)  # in seconds

        # Convert to base64
        depth_image_base64 = image_to_base64(depth_image)

        input_image = Image.open(requests.get(image_url, stream=True).raw)
        input_image_base64 = image_to_base64(input_image)

        result = {
            'input_image': input_image_base64,
            'depth_image': depth_image_base64,
            'elapsed_time': elapsed_time
        }

        return render_template("depth_estimation.html", result=result)

    return render_template("depth_estimation.html")


@app.route('/object_detection', methods=['GET', 'POST'])
def object_detection():

    if request.method == 'POST':

        image_url = request.form['image_url']

        start = time.time()
        result, box_image = huggingface.detect_objects(image_url)
        result = result['Object Detection Results']
        end = time.time()

        elapsed_time = round(end - start, 2)

        # result = [{'Object': '...', 'Confidence': ..., 'Location': '...'}]

        # Convert to base64
        image = Image.open(requests.get(image_url, stream=True).raw)
        image_base64 = image_to_base64(image)

        box_image_base64 = image_to_base64(box_image)

        result = {
            'image': image_base64,
            'box_image': box_image_base64,
            'list': result,
            'elapsed_time': elapsed_time
        }

        return render_template("object_detection.html", result=result)

    return render_template("object_detection.html")


if __name__ == "__main__":
    app.run(debug=True)
