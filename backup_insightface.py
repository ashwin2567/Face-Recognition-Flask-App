from flask import Flask, request, jsonify
import cv2
import insightface
import numpy as np
from datetime import datetime
import requests

app = Flask(__name__)

model = insightface.app.FaceAnalysis()


model.prepare(ctx_id=-1)

def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    return embedding / norm


def load_image(image_path_or_url):
    if image_path_or_url.startswith(('http:', 'https:')):
        response = requests.get(image_path_or_url)
        response.raise_for_status()
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)
    else:
        image = cv2.imread(image_path_or_url)
    return image


def compare_faces(image1_path, image2_path):
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

    faces1 = model.get(img1)
    faces2 = model.get(img2)

    if len(faces1) == 0 or len(faces2) == 0:
        return "No faces detected in one or both images"


    feat1 = normalize_embedding(model.get(img1)[0].embedding)
    feat2 = normalize_embedding(model.get(img2)[0].embedding)


    similarity = np.dot(feat1, feat2.T)

    threshold = 0.6
    if similarity > threshold:
        return True, similarity
    else:
        return False, similarity
@app.route('/', methods = ['GET','POST']) 
def home():
    return "Server is running1"

@app.route('/compare', methods=['GET','POST'])
def compare_faces_endpoint():
    st = datetime.now()
    data = request.get_json()
    image1_path = data['img1']
    image2_path = data['img2']

    result, similarity_score = compare_faces(image1_path, image2_path)
    return jsonify({'result': str(result),'similarity_score': str(similarity_score),'time':str(datetime.now()-st)})

if __name__ == '__main__':
    app.run(debug=True)
