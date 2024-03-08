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
app = Flask(__name__)

def load_image(image_path_or_url):
    try:
        if image_path_or_url.startswith(('http:', 'https:')):
            response = requests.get(image_path_or_url)
            response.raise_for_status()
            image = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)
        else:
            image = cv2.imread(image_path_or_url)
        # if image is not None:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error loading image from {image_path_or_url}: {e}")
        return None
    #return cv2.imread(image_path_or_url)
    
def preprocess_image(image_path, resize_height=200, grayscale=True, equalize_histogram=True):
    try:
        image = load_image(image_path)
        length = image.shape[0]
        width = image.shape[1]
        # if width > length:
        #     image = imutils.rotate(image, angle=270) 
        ratio = resize_height / length
        new_width = int(width * ratio)
        print(image_path, image.shape, ratio, new_width)
        
        new_width = int(width * ratio)
        resized_image = cv2.resize(image, (new_width, resize_height))
        cv2.imshow('a', resized_image)
        cv2.waitKey(0)
        return resized_image
    except Exception as e:
        print(e)



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
    
    #known_image = get_image_data(image_path1)
    # known_encoding = face_recognition.face_encodings(known_image, model='large', num_jitters=1)[0]
    
    # unknown_image = get_image_data(image_path2)
    # unknown_encoding = face_recognition.face_encodings(unknown_image, model='large', num_jitters=1)[0]
    
    # results = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=0.4)
    face_locations1 = face_recognition.face_locations(processed_image1)
    face_locations2 = face_recognition.face_locations(processed_image2)

    if not face_locations1:
        return jsonify({'result': 'Face 1 not found', 'time':str(datetime.now()-st)})
    if not face_locations2:
        return jsonify({'result': 'Face 2 not found', 'time':str(datetime.now()-st)})

    face_encoding1 = face_recognition.face_encodings(processed_image1, known_face_locations=face_locations1, model = 'small')[0]
    face_encoding2 = face_recognition.face_encodings(processed_image2, known_face_locations=face_locations2, model = 'small')[0]
    #face_encoding1 = face_recognition.face_encodings(known_image, model = 'small')[0]
    #face_encoding2 = face_recognition.face_encodings(processed_image2, model = 'small')[0]

    
    results = face_recognition.compare_faces([face_encoding1], face_encoding2, tolerance=0.4)
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
    app.run(debug=True, threaded=True)