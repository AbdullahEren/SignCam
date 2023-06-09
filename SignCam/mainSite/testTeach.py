import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from collections import Counter
import matplotlib.image as mpimg
from tkinter import messagebox
import threading
from django.shortcuts import render
   

signImageIndex = 0
offset = 20
last_frequent=0
most_frequent_element =27
imgSize = 300
text=''
message = ''
textArray=[]
labels = ["a","b","c","d","e","f", "g", "h", "i", "j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
detector = HandDetector(maxHands=1)
classifier = Classifier("mainSite/cnn_SignLanguage3.h5", "mainSite/labels.txt")
def teaching(request):
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("mainSite/cnn_SignLanguage3.h5", "mainSite/labels.txt")
    while True:
        success, img = cap.read()
        if not success:
            continue
        
        imgOutput = img.copy()
        
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # Check if ROI is within image bounds
            imgHeight, imgWidth, _ = img.shape
            if x - offset < 0 or y - offset < 0 or x + w + offset > imgWidth or y + h + offset > imgHeight:
                continue
            
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h / w
            
            # Check if aspect ratio of ROI is within a reasonable range
            if aspectRatio < 0.2 or aspectRatio > 5:
                continue
            
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)
                
                cv2.putText(imgOutput, labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
                
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

                cv2.putText(imgOutput, labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)

            
            if len(textArray)==20:
                counter = Counter(textArray)

                most_common = counter.most_common(1)
                most_frequent_element = most_common[0][0] 

                if most_frequent_element == signImageIndex:
                    message = 'TRUE this sign is '+labels[signImageIndex]
                    
                else:
                    message = 'This sign is FALSE, try again.'

                if most_frequent_element == signImageIndex:
                    cv2.destroyWindow('SıgnImage')
                    
                    signImageIndex+=1
                textArray=[]
            else:
                textArray.append(index)
            if signImageIndex<=25:  
                img2 = mpimg.imread('mainSite/SignImage/'+str(signImageIndex)+'.png')
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                
                cv2.imshow("SıgnImage", img2)
            cv2.putText(imgOutput,message,(100,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),1)

        cv2.putText(imgOutput,text,(20,40),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)    
        if signImageIndex==26:
                        response = messagebox.askquestion('Practice Again?','Do you want to practice again?(y,n)')
                        if response == 'yes':
                            signImageIndex=0
                        else:
                            break
        

        cv2.imshow("Image", imgOutput)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows() 
def image(signImageIndex):
    if signImageIndex <= 25:
                img2 = mpimg.imread('mainSite/SignImage/' + str(signImageIndex) + '.png')
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                return render(signImageIndex, 'mainSite/recognize_gesture.html', img2)


class teach_gesture(object):
    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()
        self.reset()
        
    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()
    def reset(self):
        global offset
        global last_frequent
        global most_frequent_element
        global message
        global signImageIndex
        global imgSize
        global text
        global textArray
        global imgOutput
        global labels

        signImageIndex = 0
        offset = 20
        last_frequent=0
        most_frequent_element =27
        imgSize = 300
        text=''
        message = ''
        textArray=[]
        labels = ["a","b","c","d","e","f", "g", "h", "i", "j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]


    def get_frame(self):
        global offset
        global last_frequent
        global most_frequent_element
        global message
        global signImageIndex
        global imgSize 
        global text
        
        global textArray
        global imgOutput
        global labels
        img = self.frame

        imgOutput = img.copy()
        
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # Check if ROI is within image bounds
            imgHeight, imgWidth, _ = img.shape
            
            
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            
            aspectRatio = h / w
            
            
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)
                
                cv2.putText(imgOutput, labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
                
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

                cv2.putText(imgOutput, labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)

            
            if len(textArray)==20:
                counter = Counter(textArray)

                most_common = counter.most_common(1)
                most_frequent_element = most_common[0][0] 

                if most_frequent_element == signImageIndex:
                    message = 'TRUE this sign is '+labels[signImageIndex]
                    
                else:
                    message = 'This sign is FALSE, try again.'

                if most_frequent_element == signImageIndex:
                    
                    
                    signImageIndex+=1
                textArray=[]
            else:
                textArray.append(index)
                
            cv2.putText(imgOutput,message,(100,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),1)

        cv2.putText(imgOutput,text,(20,40),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)

        
        # Display the image in the browser
        _, img_encoded = cv2.imencode('.jpg', imgOutput)
        
        return img_encoded.tobytes()
    def update(self):
        
        while True:
            (self.grabbed, self.frame) = self.video.read()
    

