# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 18:59:53 2021

@author: Hrishi_rich
"""
import cv2

# load some pre_trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect face in
img = cv2.imread('Suits.jpg')

#Convert to gray scale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw Rectange around face
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 225, 0), 2)

# Print Coordinates
print(face_coordinates)

# Popup image
cv2.imshow('Face Detector', img)
cv2.waitKey()


print("Code Completed")
