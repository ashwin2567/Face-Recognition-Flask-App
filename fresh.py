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

def preprocess_image(image_path,resize_height=400):
    if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    exif_data = image.getexif()

    if exif_data:
        if exif_data.get(274) == 6:
            image = image.rotate(270) 
    width, length = image.size 
    ratio = resize_height / length
    new_width = int(width * ratio)
    print(image_path, image.size, ratio, new_width)
    
    new_width = int(width * ratio)
    resized_image = image.resize((new_width, resize_height))

    rgb_image = resized_image.convert("RGB")
    numpy_image = np.array(rgb_image)

    return numpy_image


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
    #show_bounding_box(face_locations1,processed_image1)
    show_bounding_box(face_locations2,processed_image2)
    if not face_locations1:
        return jsonify({'result': 'Face 1 not found', 'time':str(datetime.now()-st)})
    if not face_locations2:
        return jsonify({'result': 'Face 2 not found', 'time':str(datetime.now()-st)})

    face_encoding1 = face_recognition.face_encodings(processed_image1, known_face_locations=face_locations1, model = 'small')[0]
    face_encoding2 = face_recognition.face_encodings(processed_image2, known_face_locations=face_locations2, model = 'small')[0]
    
    results = face_recognition.compare_faces([face_encoding1], face_encoding2, tolerance=0.47)
    return jsonify({'result': str(results[0]), 'time':str(datetime.now()-st)})

def show_bounding_box(face_loc, img):
    bounding_boxes = []
    for face_location in face_loc:
        top, right, bottom, left = face_location
        bounding_boxes.append((left, top, right, bottom))
    for (left, top, right, bottom) in bounding_boxes:
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.imshow("Faces with Bounding Boxes", img)
    cv2.waitKey(0)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)