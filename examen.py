import cv2
import numpy as np
from scipy import ndimage
import math
import cv2
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

def calculate(x,y,sigma):
  return((1 / (1 * (3.1416) * (pow(sigma, 2))) *(1-(pow(x, 2) + (pow(y, 2))) / (2 * (pow(sigma, 2)))))* (math.exp(-(pow(x, 2) + (pow(y, 2))) / (2 * (pow(sigma, 2))))))

def create_log_filter(size,sigma):
  matrix=[[0]*size for i in range(size)]
  for i in range(size):
    for j in range(size):
      matrix[i][j]=calculate(i-(size-1)/2,j-(size-1)/2,float(sigma))
  return matrix

def calculateGauss(x,y,sigma):
  return(1 / (2 * (3.1416) * (pow(sigma, 2))) * (math.exp(-(pow(x, 2) + (pow(y, 2))) / (2 * (pow(sigma, 2))))))

def create_gauss_filter(size,sigma):
  matrix=[[0]*size for i in range(size)]
  for i in range(size):
    for j in range(size):
      matrix[i][j]=calculateGauss(i-(size-1)/2,j-(size-1)/2,float(sigma))
  return matrix

def rgb2gray(image):
  rows, cols, dimensions= image.shape
  matriz= np.zeros((rows,cols),np.uint8)
  for i in range(rows):
    for j in range(cols):
      rp=image[i][j][0]*0.299
      gp=image[i][j][1]*0.587 
      bp=image[i][j][1]*0.114 
      #bp=image[i][j][2]*0.114
      pixel=rp+gp+bp
      matriz[i][j]=pixel
  return matriz

def imagenBordes(img_gray, img_copia, filas_lim, columnas_lim, filas, columnas):
    for y in range(filas):
        for x in range(columnas):
            img_copia[y+filas_lim][x+columnas_lim] = img_gray[y][x]
    return img_copia

def Convol2(img_gray, extremos, kernel, x, y):
    kernel = kernel/np.sum(kernel)
    valorPixel= 0
    for i in range(-extremos,extremos+1,1):
        for j in range(-extremos,extremos+1,1):
            valorKernel= kernel[i+extremos][j+extremos]
            vecinoX= x+i
            vecinoY= y+j
            valorImgGray=0
            valorImgGray= img_gray[vecinoX+extremos][vecinoY+extremos]
            valorPixel= valorPixel + (valorKernel * valorImgGray)
    return valorPixel

def Convol1(img_gray, img_gauss, kernel, size_kernel, filas, columnas):
    extremos= math.floor(size_kernel / 2)
    l=0
    for i in range(filas):
        k=0
        for j in range(columnas):
            img_gauss[l][k]= Convol2(img_gray, extremos, kernel, i, j)
            k= k+1
        l=l+1
    return img_gauss


def promedioHistograma(histograma, comienzo, fin):
    acumulado = 0
    sumaHist = 0
    promedio = 0
    for i in range(comienzo, fin):
        acumulado += histograma[i]*i
        sumaHist += histograma[i]  
    if (sumaHist != 0):
        promedio = acumulado / sumaHist
    return promedio

def miu(histograma, comienzo, fin):
    acumuladoHist = 0
    sumatoria = 0
    promedio = promedioHistograma(histograma, comienzo, fin)
    var = 0
    for i in range(comienzo, fin): 
        sumatoria += pow((i - promedio), 2) * histograma[i]
        acumuladoHist += histograma[i]
    if (acumuladoHist != 0) :
        var = sumatoria / acumuladoHist
    return var

def umbralOTSU(histograma):
    sigmaMinima = 10000000
    umbralOTSU = 0
    for Totsu in range(256):
        acumulado=0
        total=0
        Wb=0
        for i in range(0, Totsu+1):
            acumulado += histograma[i]
        for j in range(0, len(histograma)):
            total += histograma[j]
        if (total != 0):
            Wb = acumulado / total

        ub = miu(histograma, 0, Totsu)

        acumulado2=0
        total2=0
        Wf=0
        for i in range(Totsu+1,256):
            acumulado2 += histograma[i]
        for j in range(0, len(histograma)):
            total2 += histograma[j]
        if (total2 != 0):
            Wf = acumulado2 / total2
            
        uf = miu(histograma, Totsu+1, 256)

        sigmaw = (Wb * ub) + (Wf * uf)
        
        if (sigmaw < sigmaMinima) :
            sigmaMinima = sigmaw
            umbralOTSU = Totsu

    return umbralOTSU

def buscarMinimo(arreglo):
    minimo=float(100000)
    pos=0
    for i in range(256):
        if arreglo[i] > 0 and arreglo[i] < minimo:
            minimo= arreglo[i]
            pos= i
    return pos

def umbralizar(img_gauss, img_binarizada, umbral, filas, columnas):
    pixel=0
    for i in range(filas):
        for j in range(columnas):
            pixel=img_gauss[i][j]
            if pixel > umbral:
                img_binarizada[i][j]= 0
            else:
                img_binarizada[i][j]= 255
    return img_binarizada


imagen = cv2.imread("Jit1.JPG")
imagen=cv2.medianBlur(imagen,13)
scale_percent=25
width=int(imagen.shape[1]*scale_percent/100)
height=int(imagen.shape[0]*scale_percent/100)
dim=(width,height)
imagen=cv2.resize(imagen,dim,interpolation=cv2.INTER_AREA)

filas,columnas,dimensiones=imagen.shape
imagen_rojo=np.zeros((filas,columnas),np.uint8)
imagen_verde=np.zeros((filas,columnas),np.uint8)
imagen_azul=np.zeros((filas,columnas),np.uint8)


pixel_vals = imagen.reshape((-1,3)) 
pixel_vals = np.float32(pixel_vals)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 

k = 4
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 

centers = np.uint8(centers) 
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape((imagen.shape)) 


for i in range(filas):
    for j in range(columnas):
        for k in range(dimensiones):
            imagen_rojo[i][j]=segmented_image[i][j][2]
            imagen_verde[i][j]=segmented_image[i][j][1]
            imagen_azul[i][j]=segmented_image[i][j][0]

gaussiana_rojo=cv2.medianBlur(imagen_rojo,13)
gaussiana_verde=cv2.medianBlur(imagen_verde,13)
gaussiana_azul=cv2.medianBlur(imagen_azul,13)

#Obtenemos el histograma
histograma_rojo= [0] * 256
for i in range(filas):
    for j in range(columnas):
        pixel= gaussiana_rojo[i][j]
        histograma_rojo[pixel]= histograma_rojo[pixel] + 1

histograma_verde= [0] * 256
for i in range(filas):
    for j in range(columnas):
        pixel= gaussiana_verde[i][j]
        histograma_verde[pixel]= histograma_verde[pixel] + 1

histograma_azul= [0] * 256
for i in range(filas):
    for j in range(columnas):
        pixel= gaussiana_azul[i][j]
        histograma_azul[pixel]= histograma_azul[pixel] + 1


umbralOTSU_rojo=umbralOTSU(histograma_rojo)
umbralOTSU_verde=umbralOTSU(histograma_verde)
umbralOTSU_azul=umbralOTSU(histograma_azul)

rojo_binarizado= np.zeros((filas,columnas),np.uint8)
rojo_binarizado= umbralizar(gaussiana_rojo, rojo_binarizado, umbralOTSU_rojo, filas, columnas)

verde_binarizado= np.zeros((filas,columnas),np.uint8)
verde_binarizado= umbralizar(gaussiana_verde, verde_binarizado, umbralOTSU_verde, filas, columnas)

azul_binarizado= np.zeros((filas,columnas),np.uint8)
azul_binarizado= umbralizar(gaussiana_azul, azul_binarizado, umbralOTSU_azul, filas, columnas)


ret, im = cv2.threshold(rojo_binarizado, 100, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_rojo = cv2.drawContours(im, contours, -1, (0,255,75), 2)


ret, im = cv2.threshold(verde_binarizado, 100, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_verde = cv2.drawContours(im, contours, -1, (0,255,75), 2)

ret, im = cv2.threshold(azul_binarizado, 100, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_azul = cv2.drawContours(im, contours, -1, (0,255,75), 2)



deteccion= np.zeros((filas,columnas,3),np.uint8)

for i in range(filas):
    for j in range(columnas):
        deteccion[i][j][2]=img_rojo[i][j]
        deteccion[i][j][1]=img_verde[i][j]
        deteccion[i][j][0]=img_azul[i][j]
        
deteccion_grises= cv2.cvtColor(deteccion, cv2.COLOR_BGR2GRAY)
deteccion_grises_suavizada=cv2.GaussianBlur(deteccion_grises, (5,5), .4)
binarizada_2=np.zeros((filas,columnas),np.uint8)

for i in range(filas):
    for j in range(columnas):
        if deteccion_grises[i][j]==255 or deteccion_grises[i][j]==0:
            binarizada_2[i][j]=0
        else:
            binarizada_2[i][j]=255

binarizada_2=cv2.medianBlur(binarizada_2,17)

binarizada_2=cv2.Canny(binarizada_2,30,200)
contours, hierarchy  = cv2.findContours(binarizada_2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


contours_2=[]
colores=[[255,0,0],[0,255,0],[0,0,255],[255,0,255]]
anterior=0
indice_color=0
for c in contours:
    if cv2.contourArea(c) < 10000 or abs(anterior-cv2.contourArea(c))<500:
        continue
    rect=cv2.minAreaRect(c)
    box=cv2.boxPoints(rect)
    box=np.int0(box)
    #cv2.drawContours(imagen,[box],0,colores[indice_color],4)
    anterior=cv2.contourArea(c)
    contours_2.append(box)
    indice_color+=1


punto_inicial_azul_x=contours_2[0][0][0]
punto_inicial_azul_y=contours_2[0][0][1]

punto_final_azul_x=contours_2[0][2][0]
punto_final_azul_y=contours_2[0][2][1]

cv2.line(imagen,(punto_inicial_azul_x,punto_inicial_azul_y),(punto_final_azul_x,punto_final_azul_y),(255,0,0),2)

punto_inicial_rojo_x=contours_2[2][0][0]
punto_inicial_rojo_y=contours_2[2][0][1]

punto_final_rojo_x=contours_2[2][2][0]
punto_final_rojo_y=contours_2[2][2][1]

cv2.line(imagen,(punto_inicial_rojo_x,punto_inicial_rojo_y),(punto_final_rojo_x,punto_final_rojo_y),(0,0,255),2)

x1=contours_2[2][0][0]
y1=contours_2[2][0][1]
x2=contours_2[2][2][0]
y2=contours_2[2][2][1]
distancia=np.sqrt(pow(x2-x1,2)+pow(y2-y1,2))

print("LONGITUD LINEA ROJA:{}".format(int(distancia)))

x1=contours_2[0][0][0]
y1=contours_2[0][0][1]
x2=contours_2[0][2][0]
y2=contours_2[0][2][1]
distancia=np.sqrt(pow(x2-x1,2)+pow(y2-y1,2))
print("LONGITUD LINEA AZUL:{}".format(int(distancia)))


cv2.imshow("kmeans",segmented_image)
cv2.imshow("gaussiana_rojo",gaussiana_rojo)
cv2.imshow("gaussiana_verde",gaussiana_verde)
cv2.imshow("gaussiana_azul",gaussiana_azul)
cv2.imshow("rojo_binarizado",rojo_binarizado)
cv2.imshow("verde_binarizado",verde_binarizado)
cv2.imshow("azul_binarizado",azul_binarizado)
cv2.imshow("deteccion",deteccion)
cv2.imshow("deteccion_grises",deteccion_grises)
cv2.imshow("binarizada_2",binarizada_2)
cv2.imshow("imagen",imagen)

print("Ubicacion bordes:\n")
print(contours_2)
cv2.imwrite('imagen_final.png',imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()