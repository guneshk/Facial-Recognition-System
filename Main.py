import face_recognition
import cv2
import os
import glob
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
import time
import datetime

class SimpleFacerec:
    def __init__(self):

        cred = credentials.Certificate(r'your_jason_key_path.json')
        firebase_admin.initialize_app(cred, {'databaseURL': "your_firebase_URL"})

        self.db_ref = db.reference('faces')
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print("{} encoding images found.".format(len(images_path)))

        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)

        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top = int(top / self.frame_resizing)
            right = int(right / self.frame_resizing)
            bottom = int(bottom / self.frame_resizing)
            left = int(left / self.frame_resizing)

            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            if face_recognition.face_distance(self.known_face_encodings, face_encoding).size > 0:
                # Check if there are non-empty face distances
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]


            cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (0, 0, 255), 2)

            face_names.append(name)

        return frame, face_names

    def update_firebase_information(self, face_names, current_date=None):
        current_timestamp = int(time.time())
        current_datetime = datetime.datetime.fromtimestamp(current_timestamp).strftime('%Y-%m-%d %H:%M:%S')

        for name in face_names:
            self.db_ref.child(name).set({
                'last_seen': {
                    'datetime': current_datetime,
                    'date': current_date
                },
            })

if __name__ == "__main__":
    face_recognizer = SimpleFacerec()
    face_recognizer.load_encoding_images("images/")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame, face_names = face_recognizer.detect_known_faces(frame)
        face_recognizer.update_firebase_information(face_names)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
