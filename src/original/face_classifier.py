import cv2
import numpy as np
import sklearn

def extract_face_embeddings():
    labels = []
    embeddings = []
    # face detector is the caffe stuff
    # embedder creater is the torch stuff

    # resize image
    # blob from image
    # give detector blob (setInput)
    # go forward to get detections

    # if detections not empty:
    # find greatest confidence
    # get box values
    # isolate face and make sure it's large enough
    # create blob from face
    # give embedder creater blob
    # go forward to get embeddings
    # save in variables (may need to flatten)
    return embeddings, labels

def train_model(embeddings, labels):
    # make label encoder
    # fit classifier
    return None

def recognize_person(frame):
    embeddings, labels = extract_face_embeddings()
    model = train_model(embeddings, labels)
    # do same thing as extract_face_embeddings but iterate through all faces found in pic
    # for each person with confidence above threshold, classify (get label and probability)
    # draw bounding box around face and label name and prob
    return frame

def main():
    picture_path = "../images/test/carter_chris.jpg"
    img = cv2.imread(picture_path)

    detected_img = recognize_person(img)

    cv2.imshow('classified image', detected_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()