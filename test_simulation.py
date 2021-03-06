print('Set-up')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import socketio
import eventlet
import numpy as np
import cv2
import base64

from io import BytesIO
from PIL import Image
from flask import Flask
from tensorflow.keras.models import load_model
from autonomous_car_simulation.utils import *
from sklearn.model_selection import train_test_split

sio = socketio.Server()

app = Flask(__name__)
max_speed = 10


def pre_process(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = pre_process(image)
    image = np.array([image])
    steering = float(model.predict(image))
    throttle = 1.0 - speed / max_speed
    print(f'{steering}, {throttle}, {speed}')
    send_control(steering, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)


def send_control(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    ### LISTEN TO PORT 4567
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
