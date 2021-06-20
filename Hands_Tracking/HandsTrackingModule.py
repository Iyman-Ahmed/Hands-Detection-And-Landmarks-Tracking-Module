import cv2
import mediapipe as mp
import time
from Pose_Estimation import pose_estimation_module as pe

class handstracking():
    def __init__(self, mode=False, maxhands=2, min_det_conf=0.5, min_track_conf=0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.DetCon = min_det_conf
        self.TraCon = min_track_conf

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.maxhands,
                                        self.DetCon, self.TraCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findhands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mphands.HAND_CONNECTIONS, )

        return img

    def findposition(self, img, handNo = 0, Draw=True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y*h)
                lmlist.append([id,cx,cy])
                if Draw:
                    cv2.circle(img, (cx,cy), 5, (255,255,0),cv2.FILLED )
        return lmlist

def main():
    cap = cv2.VideoCapture(0)
    detector = handstracking()
    ctime = 0
    ptime = 0
    #es = pe.pose_estimator()
    while True:
        success, img = cap.read()
        img = detector.findhands(img)
        lmlist = detector.findposition(img=img)

        if len(lmlist) != 0:
             print(lmlist)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, "FPS =" + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
        cv2.imshow('Image', img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
