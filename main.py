# import necessary libraries
import cv2
import numpy as np
import face_recognition
import pandas as pd
import os
import datetime
from flask import Flask, render_template, request

# create a Flask app
app = Flask(__name__ ,template_folder='Template')

# create a list of known face encodings and names
known_face_encodings = []
known_face_names = []

# load images and their encodings
for file in os.listdir('/Users/adityamadichetty/Desktop/Images/'):
    if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
        image = face_recognition.load_image_file('/Users/adityamadichetty/Desktop/Images/' + file)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(file.split('.')[0])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # get the uploaded file
    file = request.files['image']

    # save the uploaded file to a temporary directory
    temp_path = '/Users/adityamadichetty/PycharmProjects/desktop/FLASK/venv/temp' + file.filename
    file.save(temp_path)

    # load the test image
    test_image = face_recognition.load_image_file(temp_path)

    # find the face location and encoding in the test image
    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)

    # initialize variables
    face_names = []

    for face_encoding in face_encodings:
        # compare the face encoding with known faces and get the closest match
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    # add the current date and time to the output
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # create a dataframe with the results
    df = pd.DataFrame({'Name': face_names, 'Time': timestamp})

    # save the dataframe to an Excel sheet with the current date and time as the file name
    file_name = now.strftime("%Y-%m-%d %H-%M-%S") + '.xlsx'
    df.to_excel(file_name, index=False)

    # delete the temporary file
    os.remove(temp_path)

    # render the results template with the recognized face(s) and their names
    return render_template('results.html', face_locations=face_locations, face_names=face_names)

if __name__ == '__main__':
    app.run(debug=True)
