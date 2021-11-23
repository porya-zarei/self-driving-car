import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import os

sio = socketio.Server(logger=True)

app = Flask(__name__)  # '__main__'
speed_limit = 15
last_speed = 0
last_angle = 0

def PID_Controller(angle,speed):
    global speed_limit
    global last_speed
    global last_angle
    delta_speed = speed - last_speed
    delta_angle = angle - last_angle
    if speed > speed_limit:
        speed_limit = speed
        speed_limit = min(30, speed_limit)
    else:
        speed_limit = speed_limit * 0.9
    error = angle - 0.5
    steering = error * 0.5
    throttle = speed_limit * 0.2
    last_speed = throttle
    last_angle = steering
    return steering, throttle

def img_preprocess(img):
    cv2.imshow('data-img', img)
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    cv2.imshow("img", img)
    cv2.waitKey(1)
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    try:
        # print("================================================")
        # print(data)
        # print("================================================")
        speed = float(data['speed'])
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        image = img_preprocess(image)
        image = np.array([image])
        steering_angle = float(model.predict(image))
        throttle = 1.0 - speed/speed_limit
        print(f'{throttle} , {steering_angle}')
        # s,t = PID_Controller(steering_angle,speed)
        # print(f'{t}, {s}')
        send_control(steering_angle, throttle)
    except:
        pass

@sio.on('connect')
def connect(sid, environ):
    # why print not working?
    print('Connected')
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    # path = os.path.abspath(os.path.join(os.path.dirname(
    #     __file__), '..', 'behavioral_cloning-model', "model2.h5"))
    path = "./behavioural-cloning-model/model2.h5"
    print(path)
    model = load_model(path)
    application = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), application)

