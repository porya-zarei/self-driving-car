import tensorflow_hub as hub
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd


class GeneralDetector():
    def __init__(self, detector_path="./resources/coco17/", labels_path='./resources/coco17/labels.csv', labels_name='OBJECT (2017 REL.)'):
        self.detector = hub.load(detector_path)
        self.labels = pd.read_csv(labels_path, sep=';', index_col='ID')
        self.labels = self.labels[labels_name]

    def process_rgb_tensor(self, img):
        inp = cv2.resize(img, (512, 512))
        rgb = inp.copy()
        rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
        rgb_tensor = tf.expand_dims(rgb_tensor, 0)
        return rgb_tensor, rgb

    def detect(self, img):
        rgb_tensor, rgb = self.process_rgb_tensor(img)
        boxes, scores, classes, num_detections = self.detector(rgb_tensor)
        pred_labels = classes.numpy().astype('int')[0]
        pred_labels = [self.labels[i] for i in pred_labels]
        pred_boxes = boxes.numpy()[0].astype('int')
        pred_scores = scores.numpy()[0]
        img_boxes = rgb.copy()
        for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
            if score < 0.5:
                continue
            score_txt = f'{100 * round(score,0)}'
            img_boxes = cv2.rectangle(
                rgb, (xmin, ymax), (xmax, ymin), (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_boxes, label, (xmin, ymax-10),
                        font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_boxes, score_txt, (xmax, ymax-10),
                        font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        return img_boxes

img = cv2.imread('./resources/graphic-designer.jpeg')
general_detector = GeneralDetector()
cv2.imshow('Result', general_detector.detect(img))
cv2.waitKey(0)
