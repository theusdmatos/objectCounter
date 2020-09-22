import cv2
import numpy as np
import imutils

img = cv2.imread('images/1.png', 1)
kernel = np.ones((5,5),np.uint8)

#A remoção de ruído é feita para remover ruídos indesejados da imagem para analisá-la da melhor forma
denoising = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 5, 21)

#Transformações morfologicas: são transformações que resultam na alteração da forma da imagem

#Objetos finos ou pequenos tendem a ser eliminados
erosion = cv2.erode(denoising, kernel, iterations = 1)
#Buracos finos ou pequenos serão eliminados, unindo os objetos, a imagem original é “engordada”
dilation = cv2.dilate(erosion, kernel, iterations = 1)

#convertendo a imagem para tons de cinza
gray = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
cv2.imshow("Cinza", gray)
cv2.waitKey(0)

#aplicando a detecção de bordas na imagem
edged = cv2.Canny(gray, 30, 150)

cv2.imshow("Bordas", edged)
cv2.waitKey(0)
#encontrando os contornos usando as bordas que foram encontradas
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = img.copy()

for c in cnts:
	cv2.drawContours(output, [c], -1, (139, 0, 0), 2)
	cv2.imshow("Contornos", output)
	cv2.waitKey(0)
text = " {} labels encontradas!".format(len(cnts))
cv2.putText(output, text, (5, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (139, 0, 0), 2)
cv2.imshow("Labels", output)
cv2.imwrite("out.png", output)
cv2.waitKey(0)
