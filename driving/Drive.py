import flask
import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2


class Drive():
    global sio
    global app
    sio = socketio.Server()
    app = Flask(__name__)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.speed_limit = 10
        self.speed = 0
        self.throttle = 0
        self.steering_angle = 0

    def img_preprocess(self,img):
        img = img[60:135, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img,  (3, 3), 0)
        img = cv2.resize(img, (200, 66))
        img = img/255
        return img

    @staticmethod
    @sio.on('telemetry')
    def telemetry(self,sid, data):
        speed = float(data['speed'])
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        image = self.img_preprocess(image)
        image = np.array([image])
        steering_angle = float(self.model.predict(image))
        throttle = 1.0 - speed/self.speed_limit
        print('{} {} {}'.format(steering_angle, throttle, speed))
        self.send_control(steering_angle, throttle)

    @staticmethod
    @sio.on('connect')
    def connect(self,sid, environ):
        print('Connected')
        self.send_control(0, 0)

    def send_control(self,steering_angle, throttle):
        sio.emit('steer', data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        })
        