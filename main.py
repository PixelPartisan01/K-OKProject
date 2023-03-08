# https://www.thepythoncode.com/article/gender-detection-using-opencv-in-python

import cv2
import numpy

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

face_net = cv2.dnn.readNetFromCaffe("caffe\deploy.prototxt.txt", "caffe\\res10_300x300_ssd_iter_140000_fp16.caffemodel")
gender_net= cv2.dnn.readNetFromCaffe("caffe\deploy_gender.prototxt.txt", "caffe\gender_net.caffemodel")


def findFaces (frame, threshold = 0.6):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    face_net.setInput(blob)
    output = face_net.forward()
    faces = []
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > threshold:
            box = output[i, 3:7] * \
                  numpy.array([frame.shape[1], frame.shape[0],
                            frame.shape[1], frame.shape[0]])
            start_x, start_y, end_x, end_y = box.astype(int)
            start_x, start_y, end_x, end_y = start_x - \
                                             10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # append to our list
            faces.append((start_x, start_y, end_x, end_y))
    return faces

def show(img):
    cv2.imshow(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def draw(input):
    img = cv2.imread(input)
    frame = img.copy()


    faces = findFaces(frame)

    for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
        face_img = frame[start_y: end_y, start_x: end_x]
        label = "label"
        yPos = start_y - 15
        while yPos < 15:
            yPos += 15

        font_scale = 0

        for scale in reversed(range(0, 60, 1)):
            textSize = cv2.getTextSize(label, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale / 10, thickness=1)
            new_width = textSize[0][0]

            if (new_width <= (end_x-start_x)+25):
                font_scale = scale/10
                break

        font_scale = 1

        box_color = (255, 0, 0)

        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
        cv2.putTeyt(frame, label, (start_x, yPos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 2)

    draw(frame)
    cv2.destroyAllWindows()

draw("test_1.jpg")