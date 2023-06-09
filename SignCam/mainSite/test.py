import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from collections import Counter
import threading
def split_string(text):
    return [text[i:i+50] for i in range(0, len(text), 50)]

offset = 20
most_frequent_element = 0
imgSize = 300
text = ''
script = ''
full_text = 'Script:'
text_array = []
textArray = []
imgOutput = 'a'
labels = ["a","b","c","d","e","f", "g", "h", "i", "j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
# labels = ["call_me","fist","i_love_you","ok","one","peace", "rock_n_roll", "stop", "thumbs_down", "thumbs_up"]

detector = HandDetector(maxHands=1)
classifier = Classifier("mainSite/cnn_SignLanguage3.h5", "mainSite/labels.txt")

def recognize_gesture1(request):
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("cnn_SignLanguage3.h5", "labels.txt")

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

            if len(textArray) == 30:
                counter = Counter(textArray)
                most_common = counter.most_common(1)
                most_frequent_element = most_common[0][0]

                text = text + labels[most_frequent_element]

                textArray = []
            else:
                textArray.append(index)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        else:
            full_text = full_text + text
            if full_text[-1] != " ":
                full_text += " "
            if len(full_text) >= 50:
                text_array = split_string(full_text)
            text = ''

        for i, t in enumerate(text_array):
            cv2.putText(imgOutput, t, (20, 300 + 30 * i), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 2)
        if len(full_text) < 50:
            cv2.putText(imgOutput, full_text, (20, 300), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(imgOutput, text, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

        # Display the image in the browser
        _, img_encoded = cv2.imencode('.jpg', imgOutput)
        data = img_encoded.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()



class recognize_gesture(object):
    
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
        global most_frequent_element
        
        global imgSize
        global text
        global textArray
        global imgOutput
        global labels
        global full_text
        global text_array
        global script

        offset = 20

        most_frequent_element = 0

        imgSize = 300
        text = ''
        script = ''
        full_text = 'Script:'
        text_array = []
        textArray = []
        
        labels = ["a","b","c","d","e","f", "g", "h", "i", "j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]


    def get_frame(self):
        global offset

        global most_frequent_element

        global imgSize 
        global text
        global script
        global full_text
        global text_array
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
            imgCropShape = imgCrop.shape
            aspectRatio = h / w
            if aspectRatio < 0.2 or aspectRatio > 5:
                time.sleep(0.5)

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

            if len(textArray) == 50:
                counter = Counter(textArray)
                most_common = counter.most_common(1)
                most_frequent_element = most_common[0][0]

                text = text + labels[most_frequent_element]

                textArray = []
            else:
                textArray.append(index)
        else:
            full_text = full_text + text
            if full_text[-1] != " ":
                full_text += " "
            if len(full_text) >= 50:
                text_array = split_string(full_text)
            text = ''

        for i, t in enumerate(text_array):
            cv2.putText(imgOutput, t, (20, 300 + 30 * i), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 2)
        if len(full_text) < 50:
            cv2.putText(imgOutput, full_text, (20, 300), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(imgOutput, text, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

        # Display the image in the browser
        _, img_encoded = cv2.imencode('.jpg', imgOutput)
        return img_encoded.tobytes()

    def update(self):
        
        while True:
            (self.grabbed, self.frame) = self.video.read()
