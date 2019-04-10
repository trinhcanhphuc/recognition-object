import cv2
from matplotlib import pyplot as plt
import numpy as np


class Recognition:

    def __init__(self, image_path):
        self.image_path = image_path

    def image2Grey(self):
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        cv2.imwrite('images/watchgray.png',img)

    def loadVideo(self):
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

        while(True):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out.write(frame)
            cv2.imshow('frame',gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def drawAndWriting(self):
        img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        cv2.line(img,(0,0),(200,300),(255,255,255),50)
        cv2.rectangle(img,(50,25),(200,100),(0,0,255),15)
        cv2.circle(img,(447,63), 63, (0,255,0), -1)
        pts = np.array([[100,50],[200,300],[700,200],[500,100]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img, [pts], True, (0,255,255), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'OpenCV Tuts!',(10,500), font, 6, (200,255,155), 13, cv2.LINE_AA)

        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
