import dlib
from imutils import face_utils
import cv2
import numpy as np


class ImageMouthExtractor:
    def __init__(self, shape_predictor_model_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_model_path)

    def extract_mouth(self, img):
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(grey_img, 1)
        # print(rects)
        for (i, rect) in enumerate(rects):
            shape = self.predictor(grey_img, rect)
            shape = face_utils.shape_to_np(shape)
            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                if name == "mouth":
                    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                    processed_frame = img[y:y + h, x:x + w]
                    return processed_frame
