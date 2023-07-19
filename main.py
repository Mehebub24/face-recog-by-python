import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)
# Load known faces
mehebub_images = face_recognition.load_image_file("face/my.jpg")
mehebub_encoding = face_recognition.face_encodings(mehebub_images)[0]

tanmoy_images = face_recognition.load_image_file("face/tanmoy.png")
tanmoy_encoding = face_recognition.face_encodings(tanmoy_images)[0]

#storing there name
known_face_encodings = [mehebub_encoding, tanmoy_encoding]
known_face_name =["Mehebub","Tanmoy"]

#List of known student
students = known_face_name.copy()

face_loctions = []
face_encoding = []

#get the current date and time
now=datetime.now()
current_date = now.strftime("%d-%m-%y")

f=open(f"{current_date}.csv", "w+",newline="")
lnwriter = csv.writer(f)

while True:
    _, frame= video_capture.read()
    small_frame = cv2.resize(frame,(0,0), fx=0.25 ,fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
# recognize faces
    face_loctions = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_loctions)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if(matches[best_match_index]):
           name = known_face_name[best_match_index]

           if name in known_face_name:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerofText=(10,100)
            fontScale =1.5
            fontColor =(225,0,0)
            thikness=3
            lineType=2
            cv2.putText(frame,"HI "+name, bottomLeftCornerofText, font, fontScale, fontColor, thikness, lineType)
            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
            #      lnwriter.writerow(name, current_time )
    cv2.imshow("Attendace",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break