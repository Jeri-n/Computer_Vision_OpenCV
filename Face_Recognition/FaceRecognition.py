# importing libraries
import os
import cv2 
import numpy as np
import face_recognition

path = "Data_faceRecog"

# to collect images
images = []

# to collect labels/names
classNames = []
# collect list of images
mylist = os.listdir(path)
mylist

#to print class names in recognised faces
# print names one by one (to avoide extensions)

for cl in mylist:
    currentimage=cv2.imread(f'{path}/{cl}')
    images.append(currentimage)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# find encodings by passing list of images
def findencodings(images):
    
    #to store enoding
    encodelist=[]
    
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        encode = face_recognition.face_encodings(img)[0]
  
        encodelist.append(encode)

    return encodelist

encodelistknownfaces = findencodings(images)

print("encoding completed")

print(len(encodelistknownfaces))


cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgsmall=cv2.resize(img,(0,0),None,0.25,0.25)
    # imgsmall=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#find location of faces 
    faces_in_frame = face_recognition.face_locations(imgsmall)
#find encoding of webcam  by collecting location of faces
    encoded_faces = face_recognition.face_encodings(imgsmall,faces_in_frame)


#compare all faces in imagelist and webcam faces to match faces
    for encodeface,faceloc in zip(encoded_faces,faces_in_frame): #zip used to do in same loop
        matches=face_recognition.compare_faces(encodelistknownfaces,encodeface)
        facedistance=face_recognition.face_distance(encodelistknownfaces,encodeface)
        

        #lower the distance higher the  maccthing
        # print(facedistance)

        matchIndex=np.argmin(facedistance) # to get index
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1 = faceloc
            # since we scaled down by 4 times
            y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4 #to scale up by 4 times ,because we scaled dowm as imgsmall
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            print(name)
           


    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
