import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import urllib.request


# https://stackoverflow.com/a/14617359/7690982


def segment_on_dt(a, img, img_gray):

    # Added several elliptical structuring element for better morphology process
    struct_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    struct_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    # increase border size
    border = cv2.dilate(img, struct_big, iterations=5)
    border = border - cv2.erode(img, struct_small)




    dt = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)

    # blur the signal lighty to remove noise
    dt = cv2.GaussianBlur(dt,(7,7),-1)

    # Adaptive threshold to extract local maxima of distance trasnform signal
    dt = cv2.adaptiveThreshold(dt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -9)
    #_ , dt = cv2.threshold(dt, 2, 255, cv2.THRESH_BINARY)


    # Morphology operation to clean the thresholded signal
    dt = cv2.erode(dt,struct_small,iterations = 1)
    dt = cv2.dilate(dt,struct_big,iterations = 10)

    plt.imshow(dt)
    plt.show()

    # Labeling
    lbl, ncc = label(dt)
    lbl = lbl * (255 / (ncc + 1))
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

# Open Image
resp = urllib.request.urlopen("https://i.stack.imgur.com/YUgob.jpg")
img = np.asarray(bytearray(resp.read()), dtype="uint8")
img = cv2.imdecode(img, cv2.IMREAD_COLOR)


## Yellow slicer
# blur to remove noise
img = cv2.blur(img, (9,9))

# proper color segmentation
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (0, 140, 160), (35, 255, 255))
#mask = cv2.inRange(img, (0, 0, 0), (55, 255, 255))

imask = mask > 0
slicer = np.zeros_like(img, np.uint8)
slicer[imask] = img[imask]



# Image Binarization
img_gray = cv2.cvtColor(slicer, cv2.COLOR_BGR2GRAY)

_, img_bin = cv2.threshold(img_gray, 140, 255,
             cv2.THRESH_BINARY)


plt.imshow(img_bin)
plt.show()
# Morphological Gradient
# added
cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),img_bin,(-1,-1),10)
cv2.morphologyEx(img_bin, cv2.MORPH_ERODE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),img_bin,(-1,-1),3)

plt.imshow(img_bin)
plt.show()

# Segmentation
result = segment_on_dt(img, img_bin, img_gray)
plt.imshow(np.hstack([result, img_gray]), cmap='Set3')
plt.show()

# Final Picture
result[result != 255] = 0
result = cv2.dilate(result, None)
img[result == 255] = (0, 0, 255)
plt.imshow(result)
plt.show()