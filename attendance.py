import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab

path = 'ImagesAttendance'  #the folder where are images are stored
images = [] # images are stored.
classNames = []  # list where all the images names are stored.
myList = os.listdir(path)   # this will store the elements in that path.
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')  # read the image one-by-one from the path
    images.append(curImg)  # then all the images are appnded one-by-one
    classNames.append(os.path.splitext(cl)[0]) #split the .jpg and just take the name.
print(classNames)
 #once we are donw with reading the image from the folder then we have to convert it to rgb and encode it.

def findEncodings(images):
    encodeList = []
    for img in images: # we take the each image one-by-one and encode it
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert it to rgb
        encode = face_recognition.face_encodings(img)[0]  # finding the enconding
        encodeList.append(encode) #append it to the list
    return encodeList


def markAttendance(name):   # function to store the attendance
    with open('Attendance.csv', 'r+') as f: #
        myDataList = f.readlines() # read the data from the datalist if he already marked present
        nameList = []
        for line in myDataList:
            entry = line.split(',')  #split by ,
            nameList.append(entry[0]) # append only the names to the list
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'n{name},{dtString}')

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images) # call the enconding function
print('Encoding Complete') # print it when it completed as it takes a lot of time.

cap = cv2.VideoCapture(0)  # now reading a image from the webcam

while True:
    success, img = cap.read()
    # img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25) # we are trying to reduce the image size by 1/4 to speed up the process
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) # convert the found image to rgb
    facesCurFrame = face_recognition.face_locations(imgS) # as webcam can identify multiple images we try to find the loc of the image
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame) # then we enconde it using the face location.


#next we find the diatnces and compare them
for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    # print(faceDis)
    matchIndex = np.argmin(faceDis)# this gives us the list but least the facedistance similar the model. so we take the min of it

    if matches[matchIndex]:
        name = classNames[matchIndex].upper()
# print(name)
y1, x2, y2, x1 = faceLoc # we take the location the images
y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4 # in the above we reduceed by 1/4 so now to get a perfect bounding box we have.
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # draw a bounding box with the face location
cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED) # to display the name
cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2) # put some text to indentify the image
markAttendance(name)

cv2.imshow('Webcam', img) # to show the image from the webcam.
cv2.waitKey(1)