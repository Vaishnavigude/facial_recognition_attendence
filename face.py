import cv2
import face_recognition

# Load known face encodings
known_image = face_recognition.load_image_file("vaishnavi.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Detect face from webcam
video = cv2.VideoCapture(0)
while True:
    _, frame = video.read()
    faces = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, faces)
    
    for encodeFace, faceLoc in zip(encodings, faces):
        result = face_recognition.compare_faces([known_encoding], encodeFace)
        if result[0]:
            cv2.putText(frame, "Vaishnavi", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) == ord("q"):
        break
