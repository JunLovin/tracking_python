import cv2
from cvzone.HandTrackingModule import HandDetector
# from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector as FaceDetector

# . Creo un objeto de la clase VideoCapture y lo asigno a la variable capture
capture = cv2.VideoCapture(0)

# . Creo un objeto de la clase HandDetector y lo asigno a la variable detector
detector = HandDetector(detectionCon=0.8, maxHands=2)
# . Creo un objeto de la clase FaceDetector y lo asigno a la variable face_detector
face_detector = FaceDetector(maxFaces=1)

# - Creo un ciclo infinito
while True:
    # ? Leo la camara y la asigno a la variable img
    success, img = capture.read()
    # ? Encuentro las manos y las asigno a la variable hands
    hands, hand_images = detector.findHands(img)
    # ? Encuentro la cara y la asigno a la variable face
    face, faces = face_detector.findFaceMesh(img, draw=True)

    # . Si hay cara en la imagen
    if faces:
        # ? Obtengo la cara
        face = faces[0]
        # ? Imprimo la longitud de la cara en la consola para saber cuantos puntos tiene la cara en la imagen
        print(len(face))

    # ? Hago la imagen mas grande
    img = cv2.resize(img, (1280, 720))
    # ? Muestro la imagen
    cv2.imshow("Tracking Preview", img)

    # . Si se presiona una tecla se cierra la venta
    if cv2.waitKey(1) != -1:
        break

# - Cierro la camara y destruyo todas las ventanas
capture.release()
cv2.destroyAllWindows() 