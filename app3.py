import face_recognition
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
import face_recognition
import requests
from io import BytesIO
from datetime import datetime
app = Flask(__name__)

#st = datetime.now()
def preprocess_image(image_path, resize_height=200, grayscale=True, equalize_histogram=True):
    image = cv2.imread(image_path)

    ratio = resize_height / image.shape[0]
    new_width = int(image.shape[1] * ratio)
    resized_image = cv2.resize(image, (new_width, resize_height))

    return resized_image



def preprocess_image_hist_eq(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    equalized_image = cv2.equalizeHist(image)
    return equalized_image


def preprocess_image_normalize(image_path):
    image = cv2.imread(image_path).astype(np.float32)
    normalized_image = image / 255.0 
    return normalized_image

def preprocess_image_grayscale(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def preprocess_image_blur(image_path, kernel_size=(5, 5)):
    image = cv2.imread(image_path)
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image


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
    face_locations2 = face_recognition.face_locations(processed_image2)

    if not face_locations1 or not face_locations2:
        return False

    face_encoding1 = face_recognition.face_encodings(processed_image1, known_face_locations=face_locations1, model = 'large')[0]
    face_encoding2 = face_recognition.face_encodings(processed_image2, known_face_locations=face_locations2, model = 'large')[0]

    
    results = face_recognition.compare_faces([face_encoding1], face_encoding2)
    return jsonify({'result': str(results[0]), 'time':str(datetime.now()-st)})
    
    # return results[0]

def get_image_data(image_path_or_url):
    if image_path_or_url.startswith(('http://', 'https://')):
        response = requests.get(image_path_or_url)
        response.raise_for_status()
        image_data = BytesIO(response.content)
    else:
        with open(image_path_or_url, 'rb') as file:
            image_data = BytesIO(file.read())
    
    return face_recognition.load_image_file(image_data)
# image_path1 = "IMG-20240121-WA0000.jpg"
# image_path2 = "IMG-20240121-WA0003.jpg"

# result = face_compare(image_path1, image_path2)

# if result:
#     print("Faces match!")
# else:
#     print("Faces do not match.")
# print(datetime.now()-st)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False, threaded=True)
