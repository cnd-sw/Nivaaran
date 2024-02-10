#pip install spacy

#python -m spacy download en_core_web_sm
import numpy as np
import matplotlib.pyplot as plt

import spacy

def processCaption(paragraph):
  # Load the English model
  nlp = spacy.load("en_core_web_sm")

  # Sample paragraph
  # paragraph = """
  # Urgent Update from Uttarakhand
  # Dear friends, we have an important message regarding the situation in Roorkee, Uttarakhand. The region has been hit by relentless heavy rains, resulting in multiple landslides. This has raised concerns about road damages and travel safety.
  # *Safety is Paramount!*
  # During such challenging times, our top priority must be safety. We strongly advise against traveling to the affected areas until authorities deem it safe. 6 Your well-being matters the most!
  # Our thoughts are with the people of Uttarakhand, and we hope for their safety and a swift recovery. Let's stand together in solidarity. #
  # #StaySafe #Uttarakhand #SafetyFirst #LandslideAlert #CommunitySupport #InThisTogether.
  # """

  # Process the paragraph using spaCy
  doc = nlp(paragraph)

  # Initialize a list to store extracted locations
  locations = []

  # Iterate through the sentences and extract location entities
  for sent in doc.sents:
      for ent in sent.ents:
          if ent.label_ == "GPE":  # GPE represents geographical locations in spaCy
              locations.append(ent.text)

  # Print the extracted locations
  #print("Locations found in the paragraph:")
  locations=set(locations)
  address=(",".join(locations))
  print(address)
  return address
  #for location in locations:
  #   print(location)

from geopy.geocoders import OpenCage
from geopy.exc import GeocoderQueryError
def getLongLat(address):

  # Replace with your own OpenCage API key
  api_key = 'your_api_key'

  # Initialize the geocoder with your API key
  geocoder = OpenCage(api_key)
  coordinates=[]
  try:
      if address=="":
        raise Exception("No location mentioned")
      location = geocoder.geocode(address)
      if location:
          #print(f"Location: {location.raw['formatted']}")
          #print(f"Latitude: {location.latitude}")
          coordinates.append(location.latitude)
          #print(f"Longitude: {location.longitude}")
          coordinates.append(location.longitude)
          return coordinates
  except GeocoderQueryError as e:
      print(f"Error geocoding address: {e}")
  except Exception as e:
      print(f"Error geocoding address: {e}")

pip install requests pandas

import requests
import pandas as pd

# Replace with your Instagram API access token
#access_token = 'your_access_token_here'

# Define the Instagram API endpoint URL
endpoint_url = 'your_api_key'

# Specify the parameters for your API request
#params = {
 #   'access_token': access_token,
  #  'param1': 'value1',
   # 'param2': 'value2',
    # Add any other required parameters here
#}

# Make an API request
response = requests.get(endpoint_url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON data
    json_data = response.json()

    # Convert the JSON data to a DataFrame
    df = pd.DataFrame(json_data['data'])

    # Now you have your data in a DataFrame
    print(df)
else:
    print(f"Failed to retrieve data from Instagram API. Status code: {response.status_code}")

df.head()

# Drop multiple columns by name
columns_to_drop = ['username', 'media_type']
df = df.drop(columns=columns_to_drop)

df.head()

captions=df.caption
captions

pip install nltk

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")
def intensityPredictor(text):

  # Initialize the VADER sentiment analyzer
  analyzer = SentimentIntensityAnalyzer()

  # Sample text for sentiment analysis
  # text = df.caption[2]

  # Analyze sentiment
  sentiment_scores = analyzer.polarity_scores(text)

  # Calculate sentiment severity level on a scale of 5
  # You can define your own criteria for mapping scores to severity levels
  compound_score = sentiment_scores["compound"]
  if compound_score >= 0.8:
      severity_level = 5
  elif compound_score >= 0.4:
      severity_level = 4
  elif compound_score >= 0.0:
      severity_level = 3
  elif compound_score >= -0.4:
      severity_level = 2
  else:
      severity_level = 1
  return severity_level

  #print("Sentiment Score:", compound_score)
  #print("Severity Level (on a scale of 5):", severity_level)

for caption in captions:
   addressAsString=processCaption(caption)
   coordinates=getLongLat(addressAsString)
   if coordinates!=None:
     latitude=coordinates[0]
     longitude=coordinates[1]
     print(f"Coordinates:",latitude,longitude)
     intensity=intensityPredictor(text=caption)
     print("Severity of Disaster:",intensity)
     print()


# for caption in captions:
#  addressAsString = processCaption(caption)
#  coordinates = getLongLat(addressAsString)
#  if coordinates is not None:
#    latitude, longitude = coordinates
#    print(f"Location Detected: {addressAsString}")
#    print(f"Coordinates: {latitude} {longitude}")
#    intensity = intensityPredictor(text=caption)
#    print(f"Severity= {intensity}")
#    print()  # Add an empty line for spacing

# Function to calculate accuracy
def calculate_accuracy(ground_truth, predicted_data):
    actual_severity = [item['severity_level'] for item in ground_truth]
    predicted_severity = [item['predicted_severity'] for item in predicted_data]

    severity_accuracy = np.mean(np.array(actual_severity) == np.array(predicted_severity))

    # Assuming predicted_data also contains location information
    actual_locations = [(item['actual_latitude'], item['actual_longitude']) for item in ground_truth]
    predicted_locations = [(item['predicted_latitude'], item['predicted_longitude']) for item in predicted_data]

    location_accuracy = np.mean(np.array(actual_locations) == np.array(predicted_locations))

    return severity_accuracy, location_accuracy

# Example ground truth data
ground_truth = [
    {
        'caption': 'Sample caption text',
        'actual_latitude': 123.456,
        'actual_longitude': -78.901,
        'severity_level': 3  # Actual severity level
    },
    # Add more ground truth data as needed
]

# Example predicted data
predicted_data = [
    {
        'caption': 'Sample caption text',
        'predicted_latitude': 123.456,
        'predicted_longitude': -78.901,
        'predicted_severity': 3  # Predicted severity level
    },
    # Add more predicted data as needed
]

# Calculate accuracy
severity_accuracy, location_accuracy = calculate_accuracy(ground_truth, predicted_data)

# Print and display accuracy
print(f"Severity Accuracy: {severity_accuracy * 100:.2f}%")
print(f"Location Accuracy: {location_accuracy * 100:.2f}%")

