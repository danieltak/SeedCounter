import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import label

#https://stackoverflow.com/a/44604435/7690982
def grab_frame(cap):
    ret,frame = cap.read()
    return frame

def atualizar(i):
    atualizar.counter += 1
    print(atualizar.counter)
    img = grab_frame(captura)
    if (atualizar.counter == 754) or ((atualizar.counter -74) % 120 == 0):
        cv2.imwrite(str(atualizar.counter) + '.png', img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im1.set_data(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        im2.set_data(retangulo(img))



def close(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)

def retangulo(img):
    ## Yellow slicer
    # blur to remove noise
    img = cv2.blur(img, (9, 9))

    # proper color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(hsv, (15, 25, 100), (50, 150, 200))
    imask = mask > 0
    slicer = np.zeros_like(hsv, np.uint8)
    slicer[imask] = hsv[imask]

    # Image Binarization
    img_gray = cv2.cvtColor(slicer, cv2.COLOR_BGR2GRAY)
    # mostrar_imagem(img_gray)
    _, img_bin = cv2.threshold(img_gray, 50, 255,
                               cv2.THRESH_BINARY)

    # Morphological Gradient
    # added
    cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), img_bin, (-1, -1),
                     12)
    cv2.morphologyEx(img_bin, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), img_bin, (-1, -1),
                     2)

    # Segmentation
    result = segment_on_dt(img, img_bin, img_gray)
    return result

# https://stackoverflow.com/a/14617359/7690982
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
    dt = cv2.dilate(dt,struct_big,iterations = 1)

    # Labeling
    lbl, ncc = label(dt)
    lbl = lbl * (255 / (ncc + 1))
    print(ncc)
    # Completing the markers now.
    lbl[border == 255] = 255

    lbl = lbl.astype(np.int32)
    cv2.watershed(a, lbl)

    print("[INFO] {} unique segments found".format(len(np.unique(lbl)) - 1))

    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    segment_on_dt.counter += ncc
    print('Contagem: ' + str(segment_on_dt.counter))
    return 255 - lbl

#Inicialização
captura = cv2.VideoCapture('20181207_160344.mp4')
frame = grab_frame(captura)
atualizar.counter = 0
segment_on_dt.counter = 0

n_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))

#Cria os dois subplots
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

#Cria duas imagens nos subplots
im1 = ax1.imshow(frame)
im2 = ax2.imshow(retangulo(frame))

#Animação e atualização
ani = FuncAnimation(plt.gcf(), atualizar, interval=200)

#Fechar
cid = plt.gcf().canvas.mpl_connect("key_press_event", close)

#Mostrar o gráfico
plt.show()