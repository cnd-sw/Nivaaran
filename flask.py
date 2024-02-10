from flask import Flask, request, jsonify
import numpy as np
import random
import urllib.request
from tensorflow.keras.preprocessing import image
from PIL import Image
import pickle
import tensorflow as tf
from flask import Flask, request, jsonify
import spacy
from geopy.geocoders import OpenCage
from geopy.exc import GeocoderQueryError
import requests
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Import database module.
from firebase_admin import db
from firebase import firebase

model = tf.keras.models.load_model('./model.h5')

app = Flask(__name__)

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize the OpenCage geocoder
api_key = 'opencage_api_key'
geocoder = OpenCage(api_key)

# Initialize VADER sentiment analyzer
nltk.download("vader_lexicon")
analyzer = SentimentIntensityAnalyzer()

# Define your Instagram API endpoint URL
endpoint_url = 'your_endpoint'

fb_app = firebase.FirebaseApplication('https_address_of_the_app', None)

@app.route('/process_instagram_data')
def process_instagram_data():
    try:
        # Make an API request to Instagram
        response = requests.get(endpoint_url)

        if response.status_code == 200:
            json_data = response.json()
            df = pd.DataFrame(json_data['data'])

            # Drop unnecessary columns
            columns_to_drop = ['username', 'media_type']
            df = df.drop(columns=columns_to_drop)

            captions = df['caption']

            results = []
            index = 0
            for caption in captions:
                url = df.media_url[index]
                index = index + 1
                if takePicturePredict(url) == "0":
                # Process caption using spaCy
                    locations = process_caption(caption)
                    if locations:
                        address_as_string = ",".join(locations)
                        coordinates = get_long_lat(address_as_string)

                        if coordinates:
                            latitude, longitude = coordinates
                            intensity = intensity_predictor(caption)
                            abc = {
                                'latitude': latitude,
                                'longitude': longitude,
                                'intensity': intensity,
                                'desc': caption,
                                'imageurl': url
                            }
                            results.append(abc)
                            fb_app.post('/id', abc)
            return jsonify(results)
        else:
            return jsonify(
                {'error': f'Failed to retrieve data from Instagram API. Status code: {response.status_code}'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def process_caption(paragraph):
    doc = nlp(paragraph)
    locations = []

    for sent in doc.sents:
        for ent in sent.ents:
            if ent.label_ == "GPE":
                locations.append(ent.text)

    return locations


def get_long_lat(address):
    try:
        if not address:
            raise Exception("No location mentioned")

        location = geocoder.geocode(address)

        if location:
            return location.latitude, location.longitude
    except GeocoderQueryError as e:
        print(f"Error geocoding address: {e}")
    except Exception as e:
        print(f"Error geocoding address: {e}")


def intensity_predictor(text):
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores["compound"]

    if compound_score >= 0.8:
        return 5
    elif compound_score >= 0.4:
        return 4
    elif compound_score >= 0.0:
        return 3
    elif compound_score >= -0.4:
        return 2
    else:
        return 1

def preprocess_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
@app.route('/')
def index():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('url')
    # lat = request.form.get('latitude')
    # long = request.form.get('longitude')
    intensity = random.randint(2,3)
    abc = {
        'latitude': 12.839668,
        'longitude': 80.155208,
        'desc': "image_description",
        'intensity': intensity,
        'imageurl': url
    }
    output = takePicturePredict(url)
    if output == "0":
        fb_app.post('/id', abc)
    return output

    #input_query = np.array([url]).reshape(-1, 1)

    #result = model.predict(input_query)[0]


def takePicturePredict(url):
    # url = request.form.get('url')
    urllib.request.urlretrieve(url, "flood_test.jpg")
    preprocessed_image = preprocess_image('./flood_test.jpg')
    predictions = model.predict(preprocessed_image)
    result = np.argmax(predictions)
    # return jsonify({'flood': str(result)})
    return str(result)
if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
