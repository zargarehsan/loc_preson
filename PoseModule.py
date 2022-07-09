import cv2
import time
import mediapipe as mp
import math


class poseDetector():

    def __init__(self, static_image_mode = False,
                 model_complexity = 1,
                 smooth_landmarks = True,
                 enable_segmentation = False,
                 smooth_segmentation = True,
                 min_detection_confidence = 0.5,
                 min_tracking_confidence = 0.5):

        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks,
                                     self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence,
                                     self.min_tracking_confidence)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return self.lmList


    def findPoint(self,img,p1,p2,draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        if draw:
            cv2.circle(img,(x1,y1),10,(200,0,100),-1)
            cv2.circle(img,(x2,y2),10,(200,0,100),-1)

    def findDistance(self, p1, p2):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = math.hypot(x2 - x1, y2 - y1)

        return length


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (960, 540))
        img = detector.findPose(img)
        lmList = detector.getPosition(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()