import cv2
import face_recognition

#this is used for training the model
imgElon = face_recognition.load_image_file('image/Elon Musk.jpg') # laod the image using the library
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB) #the model can understand only rgb so we convert it as required.

# this is to test
imgTest = face_recognition.load_image_file('image/Bill gates.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

#this we do to identify the image loc and encode it in a given figure
faceLoc = face_recognition.face_locations(imgElon)[0] # this gives us the location of the image [right,left,top,bottom]
encodeElon = face_recognition.face_encodings(imgElon)[0] # then we encode it
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2) # then recnagle box is formed using
                                                                                              #the faceloc , color, thickness
#this we again do it for the test image
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

#now we campre the distancs between the images. #in the backend we use svm to find whether the image match or not.
results = face_recognition.compare_faces([encodeElon], encodeTest) #when both the encoding are same its returns true else false
faceDis = face_recognition.face_distance([encodeElon], encodeTest) # when there are lot of images it very difficult so we find the distances
# between the encodeing to find out how similar they are to each other #lower the distance the better the match
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#then we want to display it on the image

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)