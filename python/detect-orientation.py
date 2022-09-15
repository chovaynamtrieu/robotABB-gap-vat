import cv2
import cv2 as cv
import numpy as np
import numpy
import time
import random
import os
import math
from matplotlib import pyplot as plt

def midpoint(p1, p2):
    return np.array((int((p1[0] + p2[0])/2),
                     int((p1[1] + p2[1])/2)))

def get_corner(pathImage, sttPhoi):
    # reading image
    img = cv2.imread(pathImage)
    img2 = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 3)
    _, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours( threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    k1 = []
    for contour in contours:
        '''
        x,y,w,h = cv.boundingRect(contour)
        cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
        '''
        if cv2.contourArea(contour) < 50:
            continue

        if cv2.contourArea(contour) > 350:
            continue
        
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        a1 = np.array(box[0])
        a2 = np.array(box[1])
        a3 = np.array(box[2])
        t1 = numpy.linalg.norm(a1-a2)
        t2 = numpy.linalg.norm(a2-a3)
        #print(t1, t2)
        if t1<40 and t2<40:
            box = np.int0(box)
            cv2.drawContours(img2, [box], 0, (0, 0, 255), 2)

            # finding center point of shape
            M = cv2.moments(box)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
                k1.append(np.array((x,y)))

    line_thickness = 2
    if len(k1) == 4 and sttPhoi == 3:
        khoangCach = [np.linalg.norm(k1[0]-k1[3]),
                      np.linalg.norm(k1[1]-k1[3]),
                      np.linalg.norm(k1[2]-k1[3])]
        diem_tuong_ung_3 = khoangCach.index(max(khoangCach))

        center = midpoint(k1[3], k1[diem_tuong_ung_3])
        for i in range(0, len(k1)):
            if k1[i][0]>=center[0] and k1[i][1]>=center[1]:
                #cv2.line(img2, center, k1[i], (255, 255, 255), thickness=line_thickness)
                detal_x = k1[i][0]-center[0]
                detal_y = k1[i][1]-center[1]
                corner = np.arctan(detal_y/detal_x)*180/3.14
                break
            #    print('1')
            #else:
            #    print('2')

        cv2.line(img2, k1[3], k1[diem_tuong_ung_3], (0, 255, 0), thickness=line_thickness)
        del(k1[3])
        del(k1[diem_tuong_ung_3])
        cv2.line(img2, k1[0], k1[1], (0, 255, 0), thickness=line_thickness)
        cv2.line(img2, center, np.array([center[0]+70, center[1]]), (255, 0, 0), thickness=line_thickness)
        cv2.line(img2, center, np.array([center[0], center[1]-70]), (255, 0, 0), thickness=line_thickness)
        
    elif len(k1) == 4 and (sttPhoi == 4 or sttPhoi == 5):
        khoangCach = [np.linalg.norm(k1[0]-k1[3]),
                      np.linalg.norm(k1[1]-k1[3]),
                      np.linalg.norm(k1[2]-k1[3])]
        diem_tuong_ung_3 = khoangCach.index(max(khoangCach))
        center = midpoint(k1[3], k1[diem_tuong_ung_3])
                
        a11 = k1[diem_tuong_ung_3]
        a12 = k1[3]
        del(k1[3])
        del(k1[diem_tuong_ung_3])
        a21 = k1[0]
        a22 = k1[1]

        t11 = midpoint(a11, a21)
        t12 = midpoint(a11, a22)#
        t21 = midpoint(a12, a21)#
        t22 = midpoint(a12, a22)

        if np.linalg.norm(a11-a22) < np.linalg.norm(a11-a21):
            k2 = [t12, t21, t22, t11]
        else:
            k2 = [t11, t22, t21, t12]

        for i in range(0, len(k2)):
            if k2[i][0]>=center[0] and k2[i][1]>=center[1]:
                #cv2.line(img2, center, k1[i], (255, 255, 255), thickness=line_thickness)
                detal_x = k2[i][0]-center[0]
                detal_y = k2[i][1]-center[1]
                corner = np.arctan(detal_y/detal_x)*180/3.14
                if i>2:
                    corner = corner + 90
                break
            #    print('1')
            #else:
            #    print('2')
        
        cv2.line(img2, t12, t21, (0, 255, 0), thickness=line_thickness)
        cv2.line(img2, t11, t22, (0, 255, 0), thickness=line_thickness)
        cv2.line(img2, center, np.array([center[0]+70, center[1]]), (255, 0, 0), thickness=line_thickness)
        cv2.line(img2, center, np.array([center[0], center[1]-70]), (255, 0, 0), thickness=line_thickness)
    elif len(k1) == 3 and sttPhoi == 1:
        M = cv2.moments(np.array(k1))
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
            center=np.array([x, y])

        corner = 0
        for i in range(0, len(k1)):
            if k1[i][0]>=center[0] and k1[i][1]>=center[1]:
                detal_x = k1[i][0]-center[0]
                detal_y = k1[i][1]-center[1]
                corner = np.arctan(detal_y/detal_x)*180/3.14
                #print('31')
                break
        if corner == 0:
            for i in range(0, len(k1)):
                if k1[i][0]<=center[0] and k1[i][1]>=center[1]:
                    detal_x = center[0]-k1[i][0]
                    detal_y = k1[i][1]-center[1]
                    corner = np.arctan(detal_x/detal_y)*180/3.14 + 90
                    #print('32')
                    break

        cv2.line(img2, center, k1[0], (0, 255, 0), thickness=line_thickness)
        cv2.line(img2, center, k1[1], (0, 255, 0), thickness=line_thickness)
        cv2.line(img2, center, k1[2], (0, 255, 0), thickness=line_thickness)
        cv2.line(img2, center, np.array([center[0]+70, center[1]]), (255, 0, 0), thickness=line_thickness)
        cv2.line(img2, center, np.array([center[0], center[1]-70]), (255, 0, 0), thickness=line_thickness)
    elif len(k1) == 2 and sttPhoi == 2:
        center = midpoint(k1[0], k1[1])
        corner = 0
        for i in range(0, len(k1)):
            if k1[i][0]>=center[0] and k1[i][1]>=center[1]:
                detal_x = k1[i][0]-center[0]
                detal_y = k1[i][1]-center[1]
                corner = np.arctan(detal_y/detal_x)*180/3.14
                #print('31')
                break
        if corner == 0:
            for i in range(0, len(k1)):
                if k1[i][0]<=center[0] and k1[i][1]>=center[1]:
                    detal_x = center[0]-k1[i][0]
                    detal_y = k1[i][1]-center[1]
                    corner = np.arctan(detal_x/detal_y)*180/3.14 + 90
                    #print('32')
                    break
        cv2.line(img2, center, k1[0], (0, 255, 0), thickness=line_thickness)
        cv2.line(img2, center, k1[1], (0, 255, 0), thickness=line_thickness)

        cv2.line(img2, center, np.array([center[0]+70, center[1]]), (255, 0, 0), thickness=line_thickness)
        cv2.line(img2, center, np.array([center[0], center[1]-70]), (255, 0, 0), thickness=line_thickness)
    # displaying the image copy after drawing contours
    cv2.imshow('shapes 2', img2)
    return corner

def test_corner(a = 0, b = 0):
    if a == 0:
        a = random.randint(1, 5)
    if b == 0:
        b = random.randint(1, 40)
    t1 = time.time()
    print('hinh mau %s chi tiet %s'%(b, a))
    corner = get_corner(pathImage = "..\\classification-data\\phoi_%s\\image_%s.png"%(a, b), sttPhoi = a)
    print('thoi gian chay: %.3f'%(time.time()-t1))
    print('goc lech: %.2f'%corner)
    
#test_corner()
corner = get_corner(pathImage = "D:/New folder (5)/20212/do-an-20212/New folder (6)/runModelCpp/im1.png", sttPhoi = 2)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
