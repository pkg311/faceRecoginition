from importlib import import_module
import cv2
import numpy as np
import dlib
import face_recognition_models
import face_recognition

imgatajalji=face_recognition.load_image_file('image/IMG_5681.jpg')
imgatajalji=cv2.cvtColor(imgatajalji,cv2.COLOR_BGR2RGB)
imgtest=face_recognition.load_image_file('image/IMG_5682.jpg')
imgtest=cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceloc=face_recognition.face_locations(imgatajalji)[0]
encondeatal=face_recognition.face_encodings(imgatajalji)[0]
cv2.rectangle(imgatajalji,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)
print(faceloc)

faceloc1=face_recognition.face_locations(imgtest)[0]
encondtest=face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceloc1[3],faceloc1[0]),(faceloc1[1],faceloc1[2]),(255,0,255),2)

result=face_recognition.compare_faces([encondeatal],encondtest)
facedis=face_recognition.face_distance([encondeatal],encondtest)
print(result,facedis)

cv2.imshow("The Atalji",imgatajalji)
cv2.imshow("test Atalji",imgtest)
cv2.waitKey(0)