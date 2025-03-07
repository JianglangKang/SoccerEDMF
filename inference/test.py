from hsv_classifier import HSVClassifier
from filters import filters
import cv2

img = cv2.imread(r'photo/A6.png')


classfier = HSVClassifier(filters=filters)

results = classfier.predict_img(img)

print(results)
