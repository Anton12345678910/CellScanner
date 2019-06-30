import numpy as np
import cv2
from matplotlib import pyplot as plt

imagen = cv2.imread('bacterias1.png',0)
imagen8 = cv2.imread('bacterias1.png',1)
imagen9 = cv2.imread('bacterias1.png',1)


kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((3,3),np.uint8)
kernel3 = np.ones((6,6),np.uint8)
kernel4 = np.ones((7,7),np.uint8)
kernel5 = np.ones((2,2),np.uint8)
kernel6 = np.ones((4,4),np.uint8)


hist = cv2.calcHist([imagen],[0],None,[256],[0,256])

argmax = hist[:240].argmax()
argmin = hist[argmax:].argmin()

equ2=cv2.equalizeHist(imagen)
#ima2=cv2.bitwise_not(equ2)
#ret, thresh2= cv2.threshold(ima2,30,100,cv2.THRESH_BINARY)
ret, thresh3= cv2.threshold(equ2,205,255,cv2.THRESH_BINARY)
#canny3=cv2.Canny(thresh3,240,255,1)
#canny2=cv2.Canny(thresh2,30,120,1)


(img, contornos, jerarquia) = cv2.findContours(thresh3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(imagen8, contornos, -1, (0,0,255), 1)

X = []
Y = []
Ma = []
mA = []
Angle = []
#print(contornos[1][1]*0.1)
for i in range(0, len(contornos)): #un corrimiento de los contornos (quedan mas centrados)
	for j in range(0, len(contornos[i])):
		contornos[i][j] = contornos[i][j]*1.001#*0.999

for i in range(0, len(contornos)):#se guardan los valores de las elipses que mas se asemejan a las bacterias
	if(len(contornos[i])>4):
		(x,y),(MA,ma),angle = cv2.fitEllipse(contornos[i])
		#x, y son el centro, MA es el ancho y ma el alto 

		if ((MA < 6) and (MA >0) and (ma <20) and (ma >0)):
			ellipse = cv2.fitEllipse(contornos[i])
			X.append(x)
			Y.append(y)
			Ma.append(MA)
			mA.append(ma)
			Angle.append(angle)

for i in range(0, len(X)):#se aplica un amplificador para agrandar los margenes de los bordes
	Ma[i] = Ma[i] * 1.10
	mA[i] = mA[i] * 1.08


for i in range(0, len(X)):#se dibujan las elipses con bordes amplificados
	ellipse = (X[i], Y[i]), (Ma[i], mA[i]), Angle[i]
	im = cv2.ellipse(imagen9,ellipse,(0,255,0),1)



print(len(contornos))
print(len(X))
cv2.imshow('original',imagen)
#cv2.imshow('thresh2',thresh2)
cv2.imshow('thresh3',thresh3)
#cv2.imshow('ima2',ima2)
cv2.imshow('equ2',equ2)
#cv2.imshow('canny2',canny2)
#cv2.imshow('canny3',canny3)
cv2.imshow('imagen_con_contornos',imagen8)
cv2.imshow('imagen_con_elipses',imagen9)
cv2.waitKey(0)