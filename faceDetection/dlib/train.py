import cv2
import os
import dlib
import numpy as np
import matplotlib.pyplot as plt
import csv
from skimage import io

subjects = ["Elvina", "Mary", "Kostya", "Asie", "Alex Ch", "Kate", "Nastya",
            "Sergey", "Alina", "Alex P", "Alex B", "Riza", "Denis", "Vlad", "Daniil"]

# file = open('train.csv', 'w')
# face_features = csv.writer(file)

# files with models
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()


def prepare_training_data(data_folder_path):

    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    face_features = []
    cnt = 0
    # let's go through each directory and read images within it
    for dir_name in dirs:

        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue

        # build path of directory containin images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        label = int(dir_name[1:len(dir_name)])
        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:
            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            img = io.imread(image_path)
            face = detector(img, 2)

            # we will ignore faces that are not detected
            if face is not None:
                for d in face:
                    print("Detection for {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                        subjects[label], d.left(), d.top(), d.right(), d.bottom()))
                    shape = sp(img, d)

                    # take descriptor
                    face_descriptor1 = facerec.compute_face_descriptor(img, shape)
                    face_descriptor1 = np.array(face_descriptor1)

                    face_features.append([])
                    for j, val in enumerate(face_descriptor1):
                        face_features[cnt].append(face_descriptor1[j])

                    # last cell is for label
                    face_features[cnt].append(label)
                    print(face_features[cnt])
                    cnt = cnt+1

            # break

    np.savetxt("train_full.csv", face_features, delimiter=",")


print("Preparing data...")
prepare_training_data("train")
print("Data prepared")