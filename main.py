import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0)
# Setting width and height of Video Screen
cap.set(3,1280)
cap.set(4,720)

# HandDetector config
detector = HandDetector(detectionCon=1)

colorR = (0,255,0)
cx, cy, w, h = 100, 100, 200, 200

class DragRect:
    def __init__(self,posCenter, size=[200,200]):
        self.posCenter= posCenter
        self.size = size

    def update(self,cursor):
        cx,cy = self.posCenter
        w,h = self.size

        # If the fingertip is in the rectangle region
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor

rectList = []
for x in range(5):
    rectList.append(DragRect([x*250+150,150]))

while True:
    success, img = cap.read()
    # Flipping the image
    img = cv2.flip(img,1)

    # Detecting the hands
    img = detector.findHands(img)

    # Drawing points over hands
    lmList, _ = detector.findPosition(img)

    if lmList:
        l, _, _ = detector.findDistance(8,12,img,draw=False)
        if l<60:
            cursor = lmList[8] #index finger tip landmark
            #Call the update here
            for rect in rectList:
                rect.update(cursor)

    # Draw Solid Image
    # for rect in rectList:
    #     cx,cy = rect.posCenter
    #     w,h = rect.size
    #     cv2.rectangle(img,(cx-w//2,cy-h//2),(cx+w//2,cy+h//2),colorR,cv2.FILLED)
    #     cvzone.cornerRect(img,(cx-w//2,cy-h//2,w,h),20,rt=0,colorC=[255,255,0])

    # Draw Transparent Image
    imgNew = np.zeros_like(img,np.uint8)
    for rect in rectList:
        cx,cy = rect.posCenter
        w,h = rect.size
        cv2.rectangle(imgNew,(cx-w//2,cy-h//2),(cx+w//2,cy+h//2),colorR,cv2.FILLED)
        cvzone.cornerRect(imgNew,(cx-w//2,cy-h//2,w,h),20,rt=0,colorC=[255,255,255])
    out = img.copy()
    alpha = 0.3
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img,alpha,imgNew,1-alpha,0)[mask]

    # Displaying the Image
    cv2.imshow("Image",out)
    cv2.waitKey(1)