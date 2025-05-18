import cv2
from cvzone.HandTrackingModule import HandDetector
# from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector as FaceDetector

capture = cv2.VideoCapture(0)

detector = HandDetector(detectionCon=0.8, maxHands=2)
face_detector = FaceDetector(maxFaces=1)

while True:
    success, img = capture.read()
    hands, hand_images = detector.findHands(img)
    face, faces = face_detector.findFaceMesh(img, draw=True)

    if faces:
        face = faces[0]
        print(len(face))

    img = cv2.resize(img, (1280, 720))
    cv2.imshow("Preview bacano", img)

    if cv2.waitKey(1) != -1:
        break

capture.release()
cv2.destroyAllWindows() 