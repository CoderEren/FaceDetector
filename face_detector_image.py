import cv2
from random import randrange

#Load pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detect faces in
img = cv2.imread('image.png')

#Grayscale the image
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
print(face_coordinates)

#Draw a rectangle around the face
#(0,255,0) is the colour. 2 is the thickness.
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

#Shows the image
cv2.imshow('Face Detector', img)

#Wait until a key is pressed
cv2.waitKey()




















print("Code completed!")
