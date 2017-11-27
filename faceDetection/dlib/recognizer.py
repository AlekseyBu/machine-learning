import dlib
import cv2
import os
import numpy as np
from imutils import face_utils
import imutils
from scipy.spatial import distance
from time import time
import threading

def draw_rectangle(img, r):
    (x, y, w, h) = r
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)


def draw_text(img, text, conf, x, y):
    cv2.putText(img, '%s - %.2f' % (text, conf), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def draw_shapes(img, shape):
    for x, y in shape:
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)


def predict(img, shape, rect_coord):
    face_descriptor1 = facerec.compute_face_descriptor(img, shape)

    min_val = 100.0
    min_index = -1
    x, y = 0, 0

    for face_descriptor2 in faces_data:
        n = len(face_descriptor2)
        index = int(face_descriptor2[n - 1])
        face_descriptor2 = face_descriptor2[0:n - 1]

        dist = distance.euclidean(face_descriptor1, face_descriptor2)

        if dist < min_val:
            min_val = dist
            min_index = index
            x, y = rect_coord[0], rect_coord[1]

    if min_val < threshold:
        print(subjects[min_index], min_val)
        return subjects[min_index], min_val, x, y
        #draw_text(img, subjects[min_index], min_val, x, y)
    else:
        print("Unknown")
        print(subjects[min_index], min_val)
        return 'Unknown', min_val, x, y


def calc_face(full_image, face_shapes, rect_coord):
    draw_rectangle(full_image, rect_coord)
    # draw_shapes(image, face_utils.shape_to_np(face_shapes))
    label, confidence, x_val, y_val = predict(full_image, face_shapes, rect_coord)
    draw_text(full_image, label, confidence, x_val, y_val)


subjects = ["Elvina", "Mary", "Kostya", "Asie", "Alex Ch", "Kate", "Nastya",
            "Sergey", "Alina", "Alex P", "Alex B", "Riza", "Denis", "Vlad", "Daniil"]

faces_cnt = [7, 5, 3, 4, 10, 4, 9, 3, 3, 3, 5, 3, 4, 4, 7, 4, 3, 3, 5, 4, 3, 5, 6, 4, 6]
tp = [7, 5, 3, 4, 9, 4, 9, 3, 3, 3, 5, 1, 4, 4, 7, 4, 3, 3, 5, 4, 3, 5, 6, 4, 2]
fp = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
fn = []

# files with models
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()
threshold = 0.59

faces_data = np.genfromtxt('train_full.csv', delimiter=",")

folder = 'test'

precision_img, recall_img = [], []
img_cnt = 0

for filename in os.listdir(folder):
    img_ind = int(filename.split('.')[0])
    img_cnt = img_cnt + 1
    image = cv2.imread(os.path.jo in(folder,filename))
    image = imutils.resize(image, width=1000)
    dets = detector(image, 1)
    print("Number of faces detected: {}".format(len(dets)))

    precision_img.append(tp[img_ind-1]/len(dets))
    recall_img.append(tp[img_ind-1]/faces_cnt[img_ind-1])
    fn.append(faces_cnt[img_ind-1] - tp[img_ind-1])

    start = time()
    threads = []

    for k, rect in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, rect.left(), rect.top(), rect.right(), rect.bottom()))
        shapes = sp(image, rect)
        rect_bb = face_utils.rect_to_bb(rect)

        t = threading.Thread(target=calc_face, name = k, args=(image, shapes, rect_bb))
        threads.append(t)
        t.start()

    for th in threads:
        th.join()

    print(time()-start)
    cv2.imshow("Number of faces detected: {}".format(len(dets)), image)
    cv2.waitKey(0)

precision = sum(precision_img)/img_cnt
print("precision: {}".format(precision))  # 98.7% of positive solutions are correct
recall = sum(recall_img)/img_cnt
print("recall: {}".format(recall))  # 94.27% of all faces were founded
