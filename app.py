from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import nltk
import pickle
import json
import random
import pymysql
import re
import streamlit as st
from sklearn.utils.multiclass import unique_labels
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt

import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from nltk.corpus import stopwords

import sys
import pandas as pd
import string
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize Flask app
app = Flask(__name__)
app.static_folder = 'static'

# Konfigurasi Database
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'review',
}

# Load the trained image classification model
model = load_model('model/model_image.h5')
modelchatBot = load_model('model/model.h5')

# Define classes for image classification
classes = ['normal', 'ringan', 'sedang', 'parah']

# Load chatbot-related files
intents = json.loads(open('data.json').read())
words = pickle.load(open('model/words.pkl', 'rb'))
classes_chatbot = pickle.load(open('model/classes.pkl', 'rb'))

# Endpoint for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint for the image classification page
@app.route('/predict')
def deteksi():
    return render_template('predict.html')

# Endpoint for processing the uploaded image
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image
    img = request.files['file']

    # Convert the image to a NumPy array
    img_array = cv2.imdecode(np.frombuffer(
        img.read(), np.uint8), cv2.IMREAD_COLOR)

    # Resize and preprocess the image
    img_array = cv2.resize(img_array, (224, 224))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make prediction
    prediction = model.predict(img_array)

    # Get the predicted class
    predicted_class = classes[np.argmax(prediction)]

    # Return the result
    return render_template('predict.html', prediction=predicted_class)


@app.route('/predict_mobile', methods=['POST'])
def predict_mobile():
    # Get the uploaded image
    img = request.files['file']

    # Convert the image to a NumPy array
    img_array = cv2.imdecode(np.frombuffer(
        img.read(), np.uint8), cv2.IMREAD_COLOR)

    # Resize and preprocess the image
    img_array = cv2.resize(img_array, (224, 224))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make prediction
    prediction = model.predict(img_array)

    # Get the predicted class
    predicted_class = classes[np.argmax(prediction)]

    # Return the result
    return jsonify({"label": predicted_class})


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = modelchatBot.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append(
            {"intent": classes_chatbot[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# Endpoint for the chatbot page


@app.route("/chatbot")
def chatbot():
    return render_template("Chatbot.html")

# Endpoint for getting chatbot response
@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    return chatbot_response(user_text)

# Chatbot mobile
@app.route("/chat", methods=["GET", "POST"])
def chatSpiceBot():
    chatInput = request.json['chatInput']
    ints = predict_class(chatInput, modelchatBot)
    return jsonify(spiceBotReply=getResponse(ints, intents))

# Endpoint for getting chatbot response mobile
@app.route("/get_mobile")
def get_bot_response_mobile():
    user_text = request.args.get('msg')
    return jsonify({"label": chatbot_response(user_text)})


# Sentimen analisis
@app.route("/feedback")
def sentimen():
    return render_template('feedback.html')


# Endpoint untuk menangani formulir pengiriman
@app.route('/submit', methods=['POST', 'OPTIONS'])
def submit_form():
    if request.method == 'OPTIONS':
        # Preflight request, respond successfully
        return jsonify({'status': 'success'})

    data_to_insert = request.get_json()
    insert_data_to_mysql(data_to_insert)
    return jsonify({'status': 'success'})


# Fungsi untuk memasukkan data ke MySQL
def insert_data_to_mysql(data):
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            # Modifikasi pernyataan SQL untuk menyertakan id_review sebagai auto-increment
            sql = """
                CREATE TABLE IF NOT EXISTS input_review (
                    id_review INT AUTO_INCREMENT PRIMARY KEY,
                    nama VARCHAR(255) NOT NULL,
                    tanggal DATE NOT NULL,
                    review TEXT NOT NULL
                )
            """
            cursor.execute(sql)

            # Masukkan data ke dalam tabel
            sql_insert = "INSERT INTO input_review (nama, tanggal, review) VALUES (%s, %s, %s)"
            cursor.execute(
                sql_insert, (data['nama'], data['tanggal'], data['review']))
        connection.commit()
        print("Data berhasil dimasukkan!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()

# Fungsi untuk mengecek apakah tabel kosong
def is_table_empty(table_name):
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
            row = cursor.fetchone()
            return row is None
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()

# Route Sentimen
# @app.route('/visualisasi', methods=['GET', 'POST'])
# def dashboard():
#     # Baca data dari MySQL atau CSV
#     table_name = 'hasil_model'
#     if is_table_empty(table_name):
#         data = pd.read_csv('facebook.csv')
#     else:
#         data = read_mysql_table(table_name)

#     data = data[['review', 'label']]

#     # Menghitung jumlah data dengan label positif, negatif, dan netral
#     jumlah_positif = len(data[data['label'] == 1])
#     jumlah_negatif = len(data[data['label'] == 0])
#     jumlah_netral = len(data[data['label'] == -1])

#     # Menyusun data untuk ditampilkan di chart
#     labels = ['Positif (1)', 'Negatif (0)', 'Netral (-1)']
#     jumlah_data = [jumlah_positif, jumlah_negatif, jumlah_netral]
#     colors = ['green', 'red', 'gray']

#     return render_template('visualisasi.html', labels=labels, jumlah_data=jumlah_data, colors=colors)


if __name__ == '__main__':
    app.run(debug=True)
