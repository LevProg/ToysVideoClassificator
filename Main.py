import cv2
import torch
from torchvision import transforms
from transform import Transform
from classificator import classificate

key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)

while True:
    check, frame = webcam.read()
    image=Transform(frame)
    print(classificate(image))
    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)


