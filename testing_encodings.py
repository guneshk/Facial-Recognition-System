"""This will test the encodings used in the facial recognition which
will help you to understand the working of the project ,make sure to change the image paths"""
import face_recognition
import os

def load_and_encode(directory):
    encodings = []
    labels = []

    for label in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, label)):
            img_path = os.path.join(directory, label, filename)
            image = face_recognition.load_image_file(img_path)
            face_encoding = face_recognition.face_encodings(image)

            if face_encoding:
                encodings.append(face_encoding[0])
                labels.append(label)

                print(f"Encoding for {label} - {filename}: {face_encoding[0]}")

    return encodings, labels

def compare_encodings(input_encoding, reference_encodings, labels):
    distances = face_recognition.face_distance(reference_encodings, input_encoding)
    min_distance_index = distances.argmin()

    return labels[min_distance_index], distances[min_distance_index]

reference_directory = r"reference_directory"
input_image_path = r"input_image_path"

reference_encodings, reference_labels = load_and_encode(reference_directory)
input_image = face_recognition.load_image_file(input_image_path)
input_encoding = face_recognition.face_encodings(input_image)

if input_encoding:
    input_encoding = input_encoding[0]
    recognized_label, distance = compare_encodings(input_encoding, reference_encodings, reference_labels)

    print(f"Recognized label: {recognized_label}")
    print(f"Distance: {distance}")
else:
    print("No face found in the input image.")
