import dlib
import cv2
import os
import numpy as np
from imutils import face_utils
import imutils
from scipy.spatial import distance
from time import time, sleep
import threading

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)


def draw_text(img, text, confidence, x, y):
    cv2.putText(img, '%s - %.2f' % (text, confidence), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def draw_shapes(img, shape):
    for (x, y) in shape:
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)


def predict(img, shape, rect):
    face_descriptor1 = facerec.compute_face_descriptor(img, shape)

    min_val = 100
    min_index = -1
    x, y = 0, 0

    for face_descriptor2 in faces_data:
        n = len(face_descriptor2)
        label = int(face_descriptor2[n - 1])
        face_descriptor2 = face_descriptor2[0:n - 1]

        dist = distance.euclidean(face_descriptor1, face_descriptor2)

        if (dist < min_val):
            min_val = dist
            min_index = label
            x, y = rect[0], rect[1]

    if (min_val < threshold):
        print(subjects[min_index], min_val)
        return subjects[min_index], min_val, x, y
        #draw_text(img, subjects[min_index], min_val, x, y)
    else:
        print("Unknown")
        print(subjects[min_index], min_val)
        return 'Unknown', min_val, x, y


subjects = ["Elvina", "Mary", "Kostya", "Asie", "Alex Ch", "Kate", "Nastya",
            "Sergey", "Alina", "Alex P", "Alex B", "Riza", "Denis", "Vlad", "Daniil"]

# files with models
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()
threshold = 0.59

faces_data = np.genfromtxt('train_one.csv', delimiter=",")

class webCamGrabber( threading.Thread ):
    def __init__(self):
        threading.Thread.__init__(self)
        self.image = False
        self.cam = cv2.VideoCapture(0)


    def run(self):
        while True:
            retval, self.image = self.cam.read() #return a True bolean and and the image if all go right


#Start webcam reader thread:
camThread = webCamGrabber()
camThread.start()

while True:
    if camThread.image is not False:
        start = time()
        img = camThread.image
        dets = detector(img)
        print("Number of faces detected: {}".format(len(dets)))

        for k, rect in enumerate(dets):
            shape = sp(img, rect)
            rect_bb = face_utils.rect_to_bb(rect)
            draw_rectangle(img, rect_bb)
            # draw_shapes(img, face_utils.shape_to_np(shape))

            # @todo: need to improve speed
            label, confidence, x, y = predict(img, shape, rect_bb)
            draw_text(img, label, confidence, x, y)

        print("process: " + str(time() - start))
        start = time()
        cv2.imshow("Recognizer", img)
        print("show: " + str(time() - start))
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()