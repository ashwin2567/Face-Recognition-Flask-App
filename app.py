from flask import Flask, request, jsonify
import face_recognition
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
    result = image_match1(img1,img2)
    return jsonify({'result': str(result[0])})

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

if __name__ == '__main__':
    app.run(debug=True)
