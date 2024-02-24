import face_recognition
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
import face_recognition
import requests
from io import BytesIO
from datetime import datetime
import imutils
from PIL import Image
app = Flask(__name__)

def get_image_orientation(image_path):
    try:
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(image_path)
        #img.show()
        exif_data = img.getexif()
        if exif_data:
            orientation = exif_data.get(274)
            return orientation
        else:
            return None
    except (AttributeError, KeyError, IndexError, IOError):
        return None
def load_image(image_path_or_url):
    try:
        if image_path_or_url.startswith(('http:', 'https:')):
            response = requests.get(image_path_or_url)
            response.raise_for_status()
            image = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)
        else:
            image = cv2.imread(image_path_or_url)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error loading image from {image_path_or_url}: {e}")
        return None
    
def preprocess_image(image_path, resize_height=200, grayscale=True, equalize_histogram=True):
    try:
        image = load_image(image_path)
        length = image.shape[0]
        width = image.shape[1]
        ratio = resize_height / length
        new_width = int(width * ratio)
        print(image_path, image.shape, ratio, new_width)
        
        new_width = int(width * ratio)
        resized_image = cv2.resize(image, (new_width, resize_height))
        ori = get_image_orientation(image_path)
        # print("ori",ori)
        # if ori == 6:
        #     resized_image=imutils.rotate(resized_image, angle=270) 
        
        return resized_image
    except Exception as e:
        print(e)

@app.route('/', methods = ['GET','POST']) 
def home():
    return "Server is running"
@app.route('/compare', methods = ['GET','POST']) 
def face_compare():
    st = datetime.now()
    data = request.get_json()
    image_path1 = data['img1']
    image_path2 = data['img2']

    processed_image1 = preprocess_image(image_path1)
    processed_image2 = preprocess_image(image_path2)
    
    face_locations1 = face_recognition.face_locations(processed_image1)
    face_locations2 = face_recognition.face_locations(processed_image2,number_of_times_to_upsample=4)

    if not face_locations1:
        return jsonify({'result': 'Face 1 not found', 'time':str(datetime.now()-st)})
    if not face_locations2:
        return jsonify({'result': 'Face 2 not found', 'time':str(datetime.now()-st)})

    face_encoding1 = face_recognition.face_encodings(processed_image1, known_face_locations=face_locations1, model = 'small')[0]
    face_encoding2 = face_recognition.face_encodings(processed_image2, known_face_locations=face_locations2, model = 'small')[0]
    
    results = face_recognition.compare_faces([face_encoding1], face_encoding2, tolerance=0.4)
    return jsonify({'result': str(results[0]), 'time':str(datetime.now()-st)})

def get_image_data(image_path_or_url):
    if image_path_or_url.startswith(('http://', 'https://')):
        response = requests.get(image_path_or_url)
        response.raise_for_status()
        image_data = BytesIO(response.content)
    else:
        with open(image_path_or_url, 'rb') as file:
            image_data = BytesIO(file.read())
    exif_data = image_data._getexif()
    if exif_data:
        orientation = exif_data.get(274)
        print("orien", orientation)
        if orientation == None:
            return jsonify({'result': 'Invalid Image'})
        elif orientation == 6:
             print("herte")
             image_data = imutils.rotate(image_data, angle=270) 
    else:
        print("No exif")
    return face_recognition.load_image_file(image_data)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
