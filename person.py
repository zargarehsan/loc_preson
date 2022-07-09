import numpy as np
import cv2
import PoseModule as pm

detector = pm.poseDetector()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img,2)
    frame = cv2.resize(img, (200, 200))
    img = detector.findPose(img,False)

    black = np.zeros_like(img)
    lmList = detector.getPosition(img,False)

    width, height = img.shape[:2]
    cv2.circle(black, (int(height / 2), int(width / 2)), 10, (0, 255, 0), -1)
    cv2.circle(black, (int(height / 2), int(width / 2)), 50, (0, 255, 0), 1)
    cv2.circle(black, (int(height / 2), int(width / 2)), 150, (0, 255, 0), 1)
    cv2.circle(black, (int(height / 2), int(width / 2)), 250, (0, 255, 0), 1)
    cv2.circle(black, (int(height / 2), int(width / 2)), 350, (0, 255, 0), 1)

    if len(lmList) != 0:
        detector.findPoint(img,11,12)
        w= detector.findDistance(11,12)
        q,e = lmList[0][1:]
        W = 35
        f = 760

        d = int((W*f)/w)

        q1 = np.interp(q,(300,1250),(500,800))
        cv2.putText(black,(f"Depth:{d} CM"),(100,100),cv2.FONT_ITALIC,1,(255,255,255),2)


        # target
        if d < 150 :
            cv2.circle(black,(int(q1),350-d),5,(0,0,255),-1)
        else:
            cv2.circle(black,(int(q1),350-d),5,(0,255,0),-1)




    cv2.imshow('image', frame)
    cv2.imshow('black', black)
    if cv2.waitKey(1) == ord("q"):
      break
