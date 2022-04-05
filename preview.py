import cv2

from train import IMG_WIDTH, IMG_HEIGHT

img = cv2.imread("afad/16/111/43905-0.jpg")
img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
cv2.imwrite("face.jpg", img)