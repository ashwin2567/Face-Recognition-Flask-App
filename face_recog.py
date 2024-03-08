import face_recognition
def image_match(known, unknown):
    known_image = face_recognition.load_image_file(known)
    known_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_image = face_recognition.load_image_file(unknown)
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
    results = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=0.4)
    print(results)

