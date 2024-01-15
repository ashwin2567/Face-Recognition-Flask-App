from flask import Flask, request, jsonify
import face_recognition
import requests
from io import BytesIO
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello :)"

@app.route('/compare', methods = ['GET']) 
def compare(): 
    data = request.get_json()
    if len(data) != 2:
        return jsonify({'result': 'Atleast 2 img required'})
    img1 = data['img1']
    img2 = data['img2']
    result = image_match(img1,img2)
    return jsonify({'result': str(result[0])})

def image_match(img1, img2):
    try:
        known_image = get_image_data(img1)
        known_encoding = face_recognition.face_encodings(known_image)[0]
        
        unknown_image = get_image_data(img2)
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        
        results = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=0.4)
        return results
    except Exception:
        return ['Error in processing']

def get_image_data(image_path_or_url):
    if image_path_or_url.startswith(('http://', 'https://')):
        response = requests.get(image_path_or_url)
        response.raise_for_status()
        image_data = BytesIO(response.content)
    else:
        with open(image_path_or_url, 'rb') as file:
            image_data = BytesIO(file.read())
    
    return face_recognition.load_image_file(image_data)

if __name__ == '__main__':
    app.run(debug=True)
'''
def image_match1(known, unknown):
    try:
        known_image = face_recognition.load_image_file(known)
        known_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_image = face_recognition.load_image_file(unknown)
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        results = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=0.4)
        return results
    except Exception:
        return ['Error in processing']
'''
