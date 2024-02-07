# Hugging Face Playground

The `huggingface.py` contains various models from huggingface, which uses the `pipeline` function to perform various tasks such as text generation, text summarization, text translation, etc.

## Tasks Performed

### 1. Sentiment Analysis

The `sentiment_analysis` function evaluates the sentiment of given textual input. It returns a list of dictionaries, each containing the sentiment label and score.

#### Result

##### Test Sentiment Analysis

| Input text                                            | Label    | Score              |
| ----------------------------------------------------- | -------- | ------------------ |
| I love this product.                                  | POSITIVE | 0.9998775720596313 |
| I hate this product.                                  | NEGATIVE | 0.9997596144676208 |
| I am not sure about this product.                     | NEGATIVE | 0.9996850490570068 |
| I am feeling happy.                                   | POSITIVE | 0.9998825788497925 |
| I am feeling sad.                                     | NEGATIVE | 0.9991661310195923 |
| I am feeling neutral.                                 | NEGATIVE | 0.9984839558601379 |
| The product has a good quality, but is too expensive. | NEGATIVE | 0.9974652528762817 |
| The product is cheap, but has a bad quality.          | NEGATIVE | 0.9994826316833496 |
| The product is neither good nor bad.                  | NEGATIVE | 0.9848746657371521 |

### 2. Text Summarization

The `summarize_text` function generates a summary for a given text using the BART-large-CNN model. The result is a list of dictionaries containing the summary text.

#### Result

##### Test Summarize Text

###### Input Text

New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
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
The case was referred to the Bronx District Attorney's Office by Immigration and Customs Enforcement and the Department of Homeland Security's
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison. Her next court appearance is scheduled for May 18.

###### Input Size

354 words

###### Summary

Liana Barrientos, 39, is charged with two counts of offering a false instrument for filing in the first degree. In total, she has been married 10 times, with nine of her marriages occurring between 1999 and 2002. If convicted, she faces up to four years in prison.

###### Summary Size

47 words

### 3. Depth Estimation

The `depth_estimate` function performs depth estimation on an input image using the Intel DPT-large model. The output is a depth image, and the user can specify the output path.

#### Result

##### Test Depth Estimate

###### Input Image

![Input Image](http://images.cocodataset.org/val2017/000000039769.jpg)

###### Depth Image

![Depth Image](./test/depth_estimate.jpg)

### 4. Object Detection

The `detect_objects` function conducts object detection on an input image using the DETR model. The output is a dictionary containing information about detected objects, including their names, confidence scores, and locations.

#### Result

##### Test Detect Objects

###### Input Image

![Input Image](http://images.cocodataset.org/val2017/000000039769.jpg)

###### Output Image

![Output Image](./test/detect_objects.png)

###### Object Detection Results

| Object | Confidence | Location                        |
| ------ | ---------- | ------------------------------- |
| remote | 0.998      | [40.16, 70.81, 175.55, 117.98]  |
| remote | 0.996      | [333.24, 72.55, 368.33, 187.66] |
| couch  | 0.995      | [-0.02, 1.15, 639.73, 473.76]   |
| cat    | 0.999      | [13.24, 52.05, 314.02, 470.93]  |
| cat    | 0.999      | [345.4, 23.85, 640.37, 368.72]  |
