from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/sentiment_analysis', methods=['GET', 'POST'])
def sentiment_analysis():

    if request.method == 'POST':
        pass

    return render_template("sentiment_analysis.html")


@app.route('/text_summarization', methods=['GET', 'POST'])
def text_summarization():

    if request.method == 'POST':
        pass

    return render_template("text_summarization.html")


@app.route('/depth_estimation', methods=['GET', 'POST'])
def depth_estimation():

    if request.method == 'POST':
        pass

    return render_template("depth_estimation.html")


@app.route('/object_detection', methods=['GET', 'POST'])
def object_detection():

    if request.method == 'POST':
        pass

    return render_template("object_detection.html")


if __name__ == "__main__":
    app.run(debug=True)
