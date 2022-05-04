import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import urllib.request

#https://stackoverflow.com/a/44604435/7690982
def grab_frame(cap):
    ret,frame = cap.read()
    return frame


def mostrar_imagem(img):
    plt.imshow(img)
    plt.show()

# https://stackoverflow.com/a/14617359/7690982
def segment_on_dt(a, img, img_gray):

    # Added several elliptical structuring element for better morphology process
    struct_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    struct_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    # increase border size
    border = cv2.dilate(img, struct_big, iterations=5)
    border = border - cv2.erode(img, struct_small)
    mostrar_imagem(border)



    dt = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)

    # blur the signal lighty to remove noise
    dt = cv2.GaussianBlur(dt,(7,7),-1)

    # Adaptive threshold to extract local maxima of distance trasnform signal
    dt = cv2.adaptiveThreshold(dt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -9)
    #_ , dt = cv2.threshold(dt, 2, 255, cv2.THRESH_BINARY)


    # Morphology operation to clean the thresholded signal
    dt = cv2.erode(dt,struct_small,iterations = 1)
    dt = cv2.dilate(dt,struct_big,iterations = 1)

    plt.imshow(dt)
    plt.show()

    # Labeling
    lbl, ncc = label(dt)
    lbl = lbl * (255 / (ncc + 1))
    print(ncc)
    # Completing the markers now.
    lbl[border == 255] = 255

    plt.imshow(lbl)
    plt.show()

    lbl = lbl.astype(np.int32)
    cv2.watershed(a, lbl)

    print("[INFO] {} unique segments found".format(len(np.unique(lbl)) - 1))

    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    return 255 - lbl

def atualizar(i):
    atualizar.counter += 1
    print(atualizar.counter)

atualizar.counter = 0



#Inicialização
img = cv2.imread('Screenshot from 20181207_160344.mp4.png')
# mostrar_imagem(img)
h, w = img.shape[:2]
print(h, w)
## Yellow slicer
# blur to remove noise
img = cv2.blur(img, (9,9))

# proper color segmentation

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# mostrar_imagem(hsv)
mask = cv2.inRange(hsv, (10, 25, 105), (50, 150, 200))
# mask = cv2.inRange(hsv, (90, 25, 110), (200, 150, 200))


imask = mask > 0
slicer = np.zeros_like(hsv, np.uint8)
slicer[imask] = img[imask]

# plt.imshow(np.hstack([cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(slicer, cv2.COLOR_BGR2RGB)]))
# plt.show()

# Image Binarization
img_gray = cv2.cvtColor(slicer, cv2.COLOR_BGR2GRAY)
# mostrar_imagem(img_gray)
_, img_bin = cv2.threshold(img_gray, 50, 255,
             cv2.THRESH_BINARY)


# plt.imshow(img_bin)
# plt.show()
# Morphological Gradient
# added
cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),img_bin,(-1,-1), 10)
# mostrar_imagem(img_bin)
cv2.morphologyEx(img_bin, cv2.MORPH_ERODE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),img_bin,(-1,-1),1)

# mostrar_imagem(img_bin)

# Segmentation
result = segment_on_dt(img, img_bin, img_gray)
plt.imshow(np.hstack([result, img_gray]))
plt.show()

# Final Picture
result[result != 255] = 0
result = cv2.dilate(result, None)
img[result == 255] = (0, 0, 255)
plt.imshow(result)
plt.show()