
from tkinter import *
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import numpy.linalg as linalg
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pandas import ExcelWriter
from mayavi.api import Engine
from mayavi.sources.api import ParametricSurface
from mayavi.modules.api import Surface
from mayavi import mlab
from tvtk.tools import visual
# loading Python Imaging Library 
from PIL import ImageTk, Image 
  
# To get the dialog box to open when required  
from tkinter import filedialog 

kernel = np.ones((5,5),np.uint8)
kernel5 = np.ones((2,2),np.uint8)
kernel6 = np.ones((4,4),np.uint8)
a = "garbanzo"
class Imagen(object):
	def __init__(self):
		self.img = None
		self.img_procesada = None
		self.img_3deizada = None
		self.contornos = []
		self.centroX = None
		self.centroY = []
		self.ancho = []
		self.largo = []
		self.angulos = []

	# SET
	def set_img(self, img_inp):
		self.img = img_inp
		#print(type(self.img))

	def set_img_procesada(self, img_inp):
		self.img_procesada = img_inp

	def set_img_3deizada(self, img_inp):
		self.img_3deizada = img_inp
	
	def set_contornos(self, contornos_inp):
		self.contornos = contornos_inp
	
	def set_centroX(self, centrosX_inp):
		self.centrosX = centrosX_inp

	def set_centroY(self, centrosY_inp):
		self.centroY = centrosY_inp

	def set_ancho(self, anchos_inp):
		self.ancho = anchos_inp
		#print(CellScann.get_ancho())

	def set_largo(self, largos_inp):
		self.largo = largos_inp

	def set_angulos(self, angulos_inp):
		self.angulo = angulos_inp

	#GET 
	def get_img(self):
		return self.img
	
	def get_img_procesada(self):
		return self.img_procesada

	def get_img_3deizada(self):
		return self.img_3deizada

	def get_contornos(self):
		return self.contornos
	
	def get_centroX(self):
		return self.centrosX

	def get_centroY(self):
		return self.centroY

	def get_ancho(self):
		return self.ancho

	def get_largo(self):
		return self.largo

	def get_angulos(self):
		return self.angulo
	
CellScann = Imagen()

def procesamiento(frame):
	gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	imagen2=cv2.bitwise_not(gris)
	_, thresh= cv2.threshold(imagen2,149,255,cv2.THRESH_BINARY)
	_, thresh2= cv2.threshold(imagen2,220,255,cv2.THRESH_BINARY)

	thresh=cv2.bitwise_not(thresh)

	DE = cv2.erode(thresh,kernel, iterations=1)
	EF = cv2.dilate(DE,kernel6,iterations=1)
	FG = cv2.erode(thresh2,kernel5,iterations=1)

	or4=cv2.bitwise_or(EF,FG)
	not4=cv2.bitwise_not(or4)
	return not4

def contornear(imagen_canny, ancho_max, ancho_min, largo_max, largo_min):
	(img, contornos, jerarquias) = cv2.findContours(imagen_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#modificada D: (img, contornos, jerarquias)
	#se elimina jerarquias en opencv 4.1
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
			#x, y son el centro, MA es el ancho y ma el largo ;;;;10;4;35;5 // 10 2 40 8
			if ((MA < ancho_max) and (MA > ancho_min) and (ma <largo_max) and (ma > largo_min)):
				contornos2.append(contornos[i])
				ellipse = cv2.fitEllipse(contornos[i])
				X.append(x)
				Y.append(y)
				Ma.append(MA)
				mA.append(ma)
				Angle.append(angle)
	return contornos2, X, Y, Ma, mA, Angle

def tresdeizar(X,Y,anchox, anchoy, angulo):
	engine = Engine()
	engine.start()
	scene = engine.new_scene()
	scene.scene.disable_render = True # for speed

	visual.set_viewer(scene)

	surfaces = []
	for k in range(0,len(X)):
		source = ParametricSurface()
		source.function = 'ellipsoid'
		engine.add_source(source)

		surface = Surface()
		source.add_module(surface)
		
		actor = surface.actor # mayavi actor, actor.actor is tvtk actor

		actor.property.opacity = 0.7
		actor.property.color = (0,0,1) # tuple(np.random.rand(3))
		actor.mapper.scalar_visibility = False # don't colour ellipses by their scalar indices into colour map

		actor.actor.orientation = np.array([90,angulo[k],0]) #* 90 #(angulo[k]) # in degrees

		actor.actor.position = np.array([X[k],Y[k],0])
		actor.actor.scale = np.array([anchox[k]/2, anchox[k]/2, anchoy[k]/2] )

		surfaces.append(surface)

		source.scene.background = (1.0,1.0,1.0)

	CellScann.set_img_3deizada(mlab)
	return mlab.show()

def open_img(): 

	# Select the Imagename  from a folder  
	x = openfilename() 

	# opens the image 
	img = Image.open(x) 
	opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
	CellScann.set_img(opencvImage)
	# PhotoImage class is used to add image to widgets, icons etc 
	img = ImageTk.PhotoImage(img) 
	
	#CellScann.set_ancho([10,20])
	# create a label 
	panel = Label(root, image = img) 
		
	# set the image as img  
	panel.image = img 
	panel.grid(row = 2, column= 0) 

def openfilename(): 
  
    # open file dialog box to select image 
    # The dialogue box has a title "Open" 
    filename = filedialog.askopenfilename(title ='"pen') 
    return filename 

def procesar():

	imagen = CellScann.get_img()
	cannyzada = procesamiento(imagen)

	contornos,  X, Y, Ma, mA, Angle = contornear(cannyzada, 10, 4, 35, 5)
	CellScann.set_contornos(np.array(contornos, dtype=object))
	CellScann.set_centroX(np.array(X))
	CellScann.set_centroY(np.array(Y))
	CellScann.set_ancho(np.array(Ma))
	CellScann.set_largo(np.array(mA))
	CellScann.set_angulos(np.array(Angle))


	#Datos Anchos
	decimales = 4
	Label(root, text="""
	Promedio anchos: {} \t\t|\t Promedio largos: {} \n
	Desviacion anchos: {} \t\t|\t Desviacion largos: {} \n
	Mediana anchos: {} \t\t|\t Mediana largos: {} \n
	Varianza anchos: {} \t\t|\t Varianza largos: {} \n
	""".format(str(round(np.mean(CellScann.get_ancho()),decimales)), str(round(np.mean(CellScann.get_largo()),decimales)), str(round(np.std(CellScann.get_ancho()),decimales)), str(round(np.std(CellScann.get_largo()),decimales)), str(round(np.median(CellScann.get_ancho()),decimales)), str(round(np.median(CellScann.get_largo()),decimales)), str(round(np.var(CellScann.get_ancho()),decimales)), str(round(np.var(CellScann.get_largo()),decimales)))).grid(row = 2, column = 3)


	"""
	Label(root, text="Promedio anchos: {}".format(str(round(np.mean(CellScann.get_ancho()),decimales)))).grid(row = 3, column = 3)
	Label(root, text="Desviacion anchos: {}".format(str(round(np.std(CellScann.get_ancho()),decimales)))).grid(row = 4, column = 3)
	Label(root, text="Mediana anchos: {}".format(str(round(np.median(CellScann.get_ancho()),decimales)))).grid(row = 5, column = 3)
	Label(root, text="Varianza anchos: {}".format(str(round(np.var(CellScann.get_ancho()),decimales)))).grid(row = 6, column = 3)

	#Datos Largos
	Label(root, text="\t|\t Promedio largos: {}".format(str(round(np.mean(CellScann.get_largo()),decimales)))).grid(row = 3, column = 4)
	Label(root, text="\t|\t Desviacion largos: {}".format(str(round(np.std(CellScann.get_largo()),decimales)))).grid(row = 4, column = 4)
	Label(root, text="\t|\t Mediana largos: {}".format(str(round(np.median(CellScann.get_largo()),decimales)))).grid(row = 5, column = 4)
	Label(root, text="\t|\t Varianza largos: {}".format(str(round(np.var(CellScann.get_largo()),decimales)))).grid(row = 6, column = 4)
	"""
	for i in range(0, len(X)):#se dibujan las elipses con bordes amplificados de todas las elipses
		ellipse = (X[i], Y[i]), (Ma[i], mA[i]), Angle[i]
		ima = cv2.ellipse(imagen,ellipse,(0,0,255),1, cv2.LINE_AA)

	ima_pil = Image.fromarray(ima)
	CellScann.set_img_procesada(ima)
	# set the image as img  
	procesada = cv2.cvtColor(ima, cv2.COLOR_BGR2RGB)
	img = Image.fromarray(procesada)
	img = ImageTk.PhotoImage(img) 
	panel = Label(root, image = img) 
	panel.image = img
	panel.grid(row = 2, column = 0)

def tresdeizacion(): 
	tresdeizar(CellScann.get_centroX(), CellScann.get_centroY(), CellScann.get_ancho(), CellScann.get_largo(), CellScann.get_angulos())
	#CellScann.get_img_3deizada().show()

# Se crea una ventana
root = Tk() 


# Titulo de la ventana
root.title("CellScanner") 
  
# Resolucion de la ventana "inicial"
root.geometry("900x500") 
  
# Ventana ajustable
root.resizable(width = True, height = True) 

# Botones en el grid
open_ima = Button(root, text ='Abrir imagen', command = open_img).grid(row = 1, column=0, columnspan = 4) 

Procesar = Button(root, text ='Procesar', command = procesar).grid(row = 1, column=7, columnspan = 9) 

tresde = Button(root, text ='3deizar', command = tresdeizacion).grid(row = 1, column=20,columnspan = 15)

root.mainloop() 

