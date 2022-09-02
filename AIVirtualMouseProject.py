# -*- coding:utf-8 -*- 
# 用户：Ghostraveler
# 开发时间：2022/8/12 9:30 上午
import cv2
import time
import csv
import mediapipe as mp
import pyautogui
from cvzone.HandTrackingModule import HandDetector

class FaceMeshDetector():
    def __init__(self,staticMode = False,maxFaces=2,minDetectionCon = 0.5,minTrackCon = 0.5):
                 self.staticMode = staticMode
                 self.maxFaces = maxFaces
                 self.minDetectionCon = minDetectionCon
                 self.minTrackCon = minTrackCon

                 self.mpDraw = mp.solutions.drawing_utils
                 self.mpFaceMesh = mp.solutions.face_mesh
                 self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,self.minDetectionCon
                                                          ,self.minTrackCon)

                 self.drawSpec = self.mpDraw.DrawingSpec(thickness=1,circle_radius=1)


    def findFaceMesh(self,img,draw=True):
        self.header=['序号','横坐标','纵坐标']

        self.imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        with open('data2.csv','w',encoding='utf-8') as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(self.header)

            if self.results.multi_face_landmarks:

                for faceLms in self.results.multi_face_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACE_CONNECTIONS,
                                          self.drawSpec,self.drawSpec)

                    face = []
                    for id,lm in enumerate(faceLms.landmark):
                        #print(lm)
                        ih,iw,ic = img.shape
                        x,y= int(lm.x*iw),int(lm.y*ih)
                        #cv2.putText(img,str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

                        #print(id,x,y)
                        face.append([id,x,y])
                        writer.writerow([id,x,y])
                    faces.append(face)

        return img,faces



def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector= FaceMeshDetector(maxFaces=2)
    detector1 = HandDetector(maxHands=1)

    while True:
        success, img = cap.read()
        img,faces= detector.findFaceMesh(img)

        if len(faces)!= 0:
            x1,y1,z1=faces[0][10]; x2,y2,z2=faces[0][9]; x3,y3,z3=faces[0][4];
            x4,y4,z4=faces[0][152]; x5,y5,z5=faces[0][70]; x6,y6,z6=faces[0][107];
            x7,y7,z7=faces[0][336]; x8,y8,z8=faces[0][300];

            x1=(z3-z2)/(z2-z1)
            x2=(z4-z3)/(z3-z2)
            x3=(z4-z3)/(z2-z1)
            x4=(y6-y5)/(z2-z1)
            x5=(y8-y7)/(z3-z2)
            x6=(y6-y5)/(z3-z2)

            print(x6)

            if (x1>0.5 and x1<3.0) and (x2>0 and x2<2.0) and(x3>2.0 and x3<2.8) and (x4>0.35 and x4<2.6)\
                    and (x5>0.6 and x5<0.75) and (x6>0.7 and x6<0.85):
                print('hjm')
                flag=True
            else:
                print('no hjm')
                flag=False


            if flag==True:
                hands, img1 = detector1.findHands(img)
                if hands:
                    # 打印出手上所有节点的坐标
                    lmList = hands[0]['lmList']
                    x1, y1, z1 = lmList[8]
                    pyautogui.moveTo(x1, y1)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow('Image', img)

        cv2.waitKey(1)



if __name__ == "__main__":
    main()