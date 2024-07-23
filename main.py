import cv2
from cvzone.FaceDetectionModule import FaceDetector
import time

# Carregar o classificador Haar Cascade para sorriso
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Inicializar a captura de vídeo e o detector de rostos
video = cv2.VideoCapture(0)
detector = FaceDetector()
running = True  # Variável para controlar o loop

while running:
    _, img = video.read()
    original_img = img.copy()  # Fazer uma cópia da imagem original
    img, bboxes = detector.findFaces(img, draw=True)

    if bboxes:
        for bbox in bboxes:
            x, y, w, h = bbox['bbox']
            roi_gray = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)

            if len(smiles) > 0:
                # Salvar a imagem quando um sorriso é detectado
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(f"sorriso_{timestamp}.jpg", original_img)
                print(f"Foto salva como: sorriso_{timestamp}.jpg")

    cv2.imshow('Resultado', img)
    key = cv2.waitKey(1)
    if key == 27:  # Pressione 'ESC' para sair
        running = False

video.release()
cv2.destroyAllWindows()
