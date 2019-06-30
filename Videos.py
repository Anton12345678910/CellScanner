import cv2
import numpy as np
import pandas as pd

def filtrado(x,y,X1, Y1):#acota la region de busqueda de la bacteria senialada con el punto
	indices = []
	for i in range(0, len(X1)):
		if (X1[i] <= x + 10) and (Y1[i] <= y + 10):
			if (X1[i] >= x - 10) and (Y1[i] >= y - 10):
				indices.append(i)
	return indices

def encontrar(x, y, indices, cont):#encuentra el indice de la bacteria señalada con el punto (se supone XD)
	minimo = 999.0
	encontrado = 0
	distancias = []
	for i in range(0, len(indices)):
		#print("indices", indices)
		for j in range(0, len(cont[indices[i]])):
			x1 = cont[indices[i]][j][0][0]
			y1 = cont[indices[i]][j][0][1]
			tmp = np.sqrt(((x-x1)**2)+((y-y1)**2))
			distancias.append(np.sqrt(((x-x1)**2)+((y-y1)**2)))

		tmp = min(distancias)
		
		if tmp < minimo:
			minimo = tmp
			encontrado = i
		
		del distancias[:]

	return indices[encontrado]

cap = cv2.VideoCapture("gro1.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')#para guardar videos
frame_width = int(cap.get(3))# //
frame_height = int(cap.get(4))# //
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))# //

# Parametros para la funcion de Lucas Kanade
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01))
 
#Capturamos una imagen y la convertimos de RGB -> HSV
_, imagen = cap.read()
frame = imagen.copy()
# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_verde = np.array([50,61,15], dtype=np.uint8)
upper_verde = np.array([166,255,255], dtype=np.uint8)

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_verde, upper_verde)
mask = cv2.bitwise_not(mask)
# Bitwise-AND mask and original image
bitwise = cv2.bitwise_and(frame,frame, mask= mask)
gris = cv2.cvtColor(bitwise, cv2.COLOR_BGR2GRAY)
################################################
imagen3 = gris.copy()
#kernel = np.ones((4,4),np.uint8)
kernel2 = np.ones((3,3),np.uint8)
hist = cv2.calcHist([imagen3],[0],None,[256],[0,256])
maximo=max(hist[:240])
argmax = hist[:240].argmax()
argmin = hist[argmax:].argmin()
#aplica el algoritmo de canny a la imagen 'gaussiana', con umbrales 100 y 110 (minimo y maximo respectivamente)
ima=cv2.dilate(imagen3, kernel2, iterations = 1)
canny = cv2.Canny(ima, argmax-argmin-3, argmax+argmin+3,3)

(img, contornos, jerarquia) = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	#comprobacion de bordes detectados

X = []
Y = []
Ma = []
mA = []
Angle = []
contornos2 = []

for i in range(0, len(contornos)):#se guardan los valores de las elipses que mas se asemejan a las bacterias
	if(len(contornos[i])>4):
		(x,y),(MA,ma),angle = cv2.fitEllipse(contornos[i])
		#x, y son el centro, MA es el ancho y ma el alto 
		if ((MA < 10) and (MA >4) and (ma <35) and (ma > 5)):
			contornos2.append(contornos[i])
			ellipse = cv2.fitEllipse(contornos[i])
			X.append(x)
			Y.append(y)
			Ma.append(MA)
			mA.append(ma)
			Angle.append(angle)
#son 131 :v
#2, 3 = 4, {7, 8(salto)}, 9, {10 = 11, 12,13,14,15} 
#2, 3 = 4, {salto: 7, 8, 12, 19, 20, 21, 22, 26, 53},15 = 16, 54 = x+8,y+5; x-5,y-8, 65, 123 salto, 129
punto_a_encontrar = 2
momentos = cv2.moments(contornos2[punto_a_encontrar])
cx = float(momentos['m10']/momentos['m00'])
cy = float(momentos['m01']/momentos['m00'])

#print("punto a encontrar: ", X[punto_a_encontrar], Y[punto_a_encontrar])

#Convertimos la imagen a gris para poder introducirla en el bucle principal
frame_anterior = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

#Convertimos el punto elegido a un array de numpy que se pueda pasar como parametro
#a la funcion cv2.calcOpticalFlowPyrLK()
punto_elegido = np.array([[[X[punto_a_encontrar], Y[punto_a_encontrar]]]],np.float32)

divisiones = 0
nframes = 0
largo1 = mA[punto_a_encontrar]
largo2 = 1
div_text = " 0"
seg_text = " 0"
datos_largo = {}
datos_ancho = {}

ancho = []
largo = []
ancho_numpy = []
largo_numpy = []
count = 0
while(True):
	ret, frame = cap.read()

	if ret==True:
		nframes = nframes + 1
		imagen = frame.copy()
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(imagen,'Divitions: '+div_text,(40, 50), font, 1,(255,255,255),1,cv2.LINE_AA)
		cv2.putText(imagen,'Time of the last divition: '+seg_text,(40, 90), font, 1,(255,255,255),1,cv2.LINE_AA)
		# Convert BGR to HSV
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		gris1 = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

		#se aplica el metodo de Lucas Kanade
		punto_elegido, st, err = cv2.calcOpticalFlowPyrLK(frame_anterior, gris1, punto_elegido,None, **lk_params)
		#print("punto elegido: ", punto_elegido[0][0])

		#Se guarda el frame de la iteracion anterior del bucle
		frame_anterior = gris1.copy()

		# define range of blue color in HSV
		lower_verde = np.array([50,61,15], dtype=np.uint8)
		upper_verde = np.array([166,255,255], dtype=np.uint8)

		# Threshold the HSV image to get only blue colors
		mask = cv2.inRange(hsv, lower_verde, upper_verde)
		mask = cv2.bitwise_not(mask)
		# Bitwise-AND mask and original image
		bitwise = cv2.bitwise_and(frame,frame, mask= mask)
		gris = cv2.cvtColor(bitwise, cv2.COLOR_BGR2GRAY)
		################################################
		imagen3 = gris.copy()
		#kernel = np.ones((4,4),np.uint8)
		kernel2 = np.ones((3,3),np.uint8)
		hist = cv2.calcHist([imagen3],[0],None,[256],[0,256])
		maximo=max(hist[:240])
		argmax = hist[:240].argmax()
		argmin = hist[argmax:].argmin()
		#aplica el algoritmo de canny a la imagen 'gaussiana', con umbrales 100 y 110 (minimo y maximo respectivamente)
		ima=cv2.dilate(imagen3, kernel2, iterations = 1)
		canny = cv2.Canny(ima, argmax-argmin-3, argmax+argmin+3,3)

		(img, contornos, jerarquia) = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		#comprobacion de bordes detectados
		X = []
		Y = []
		Ma = []
		mA = []
		Angle = []
		contornos2 = []

		for i in range(0, len(contornos)):#se guardan los valores de las elipses que mas se asemejan a las bacterias
			if(len(contornos[i])>4):
				(x,y),(MA,ma),angle = cv2.fitEllipse(contornos[i])
				#x, y son el centro, MA es el ancho y ma el alto 
				if ((MA < 10) and (MA >2) and (ma <40) and (ma > 8)):
					ellipse = cv2.fitEllipse(contornos[i])
					X.append(x)#centro x
					Y.append(y)#centro y
					Ma.append(MA)#ancho
					mA.append(ma)#largo
					Angle.append(angle)#angulo :V
					contornos2.append(contornos[i])
					ancho_numpy.append(MA)
					largo_numpy.append(ma)
		"""			
		for i in range(0, len(mA)):
			ancho.append(str(Ma[i]))
			largo.append(str(mA[i]))
		"""

		ancho.append("frame_ancho"+str(nframes))
		largo.append("frame_largo"+str(nframes))
		datos_ancho["frame_ancho"+str(nframes)] = pd.Series(Ma)
		datos_largo["frame_largo"+str(nframes)] = pd.Series(mA)
		c = 0
		puntos = filtrado(punto_elegido[0][0][0], punto_elegido[0][0][1],X, Y)
		punto = encontrar(punto_elegido[0][0][0], punto_elegido[0][0][1], puntos, contornos2)
		punto_elegido = np.array([[[X[punto], Y[punto]]]],np.float32)
		#print("puntos", puntos)
		largo2 = mA[punto]
		#print(largo2)
		if ((largo2*100)/largo1) <= 50:
			print("DIVISION!!!")
			divisiones = divisiones + 1
			div_text = str(divisiones)
			seg_text = str(nframes/10)
			print("En el segundo: ", nframes/10)

		for i in range(0, len(puntos)):
			c = puntos[i]
			ellipse = (X[c], Y[c]), (Ma[c], mA[c]), Angle[c]
			im = cv2.ellipse(imagen,ellipse,(255,0,0),1, cv2.LINE_AA)

		#print("punto encontrado: ", X[c], Y[c])
		for i in punto_elegido:
			#cv2.circle(imagen,tuple(i[0]), 3, (255,0,0), -1)
			ellipse = (X[punto], Y[punto]), (Ma[punto], mA[punto]), Angle[punto]
			im = cv2.ellipse(imagen,ellipse,(0,0,255),1, cv2.LINE_AA)

		for i in range(0, len(X)):#se aplica un amplificador para agrandar los margenes de los bordes
			Ma[i] = Ma[i] * 1.2
			mA[i] = mA[i] * 1.2

		for i in range(0, len(X)):#se dibujan las elipses con bordes amplificados
			ellipse = (X[i], Y[i]), (Ma[i], mA[i]), Angle[i]
			im = cv2.ellipse(bitwise,ellipse,(0,0,255),1, cv2.LINE_AA)

		ellipse = (X[0], Y[0]), (Ma[0], mA[0]), Angle[0]
		im = cv2.ellipse(bitwise,ellipse,(255,0,0),-1, cv2.LINE_AA)
		cv2.drawContours(frame, contornos, -1, (0,255,0),1)
		cv2.imshow('Frame',frame)
		cv2.imshow('bitwise',bitwise)
		cv2.imwrite("imagen%d.jpeg" %count, imagen)
		count = count +1
		out.write(imagen) # escribe el frame actual(para el video)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		largo1 = largo2
		largo2 = 0
		
	else:
		break
print(len(datos_ancho))

writer = pd.ExcelWriter('Datos_Bacterias.xlsx')
dflargo = pd.DataFrame(datos_largo, columns = largo)
dfancho = pd.DataFrame(datos_ancho, columns = ancho)

dflargo.to_excel(writer,'largo')
dfancho.to_excel(writer,'ancho')
writer.save()
print(type(dflargo.describe()))
print(pd.DataFrame.mean(dflargo))
print(type(pd.DataFrame.mean(dflargo)))
print("································3")
promedio = 0
print(len(largo_numpy))
largo_numpy = np.array(largo_numpy)
ancho_numpy = np.array(ancho_numpy)
print("promedio largos: ", np.mean(largo_numpy))
print("promedio anchos: ", np.mean(ancho_numpy))
print("desviacion estandar largos: ", np.std(largo_numpy))
print("desviacion estandar anchos: ", np.std(ancho_numpy))
print("mediana largos: ", np.median(largo_numpy))
print("mediana anchos: ", np.median(ancho_numpy))
print("varianza largos: ", np.var(largo_numpy))
print("varianza anchos: ", np.var(ancho_numpy))
# Cuando todo está listo, se libera la captura 
print("Frames del video: ", nframes)
print("Total de divisiones: ", divisiones)
cap.release()
out.release()#suelta el "guardado" del video
cv2.destroyAllWindows()
