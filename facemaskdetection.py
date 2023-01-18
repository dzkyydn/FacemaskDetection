import keras
import cv2
import numpy as np
import pyglet

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
camera = cv2.VideoCapture(0)

sound = pyglet.media.load("please_use_your_mask.mp3", streaming=False)


model = keras.models.load_model('facemodel.h5')
model.compile(loss=keras.losses.BinaryCrossentropy(),
              optimizer='sgd',
              metrics=[keras.metrics.BinaryCrossentropy()])

while True:
    _, frame = camera.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=3, 
        minSize = (30, 30)
    )

    for(x, y, w, h) in face:
        stroke = 2
        width = x + w
        height = y + h
        img = frame[y:height,x:width]
        img = cv2.resize(img, (75, 75))
        img = np.reshape(img, [1, 75, 75, 3])
        if model.predict(img)[0][0]>0.5:
            cv2.rectangle(frame, (x,y), (width, height), (0, 0, 255), stroke)
            frame = cv2.putText(frame, 'Not wearing mask', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            sound.play()
            #playsound('please_use_your_mask.mp3')
        else:
            cv2.rectangle(frame, (x,y), (width, height), (0, 255, 0), stroke)
            frame = cv2.putText(frame, 'Wearing mask', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('Face Mask Detection Aplication', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()