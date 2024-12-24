import cv2
import mediapipe as mp
import math


class handDetector:
    def __init__(self, detectionCon=0.5, maxHands=1):
        self.detectionCon = detectionCon
        self.maxHands = maxHands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(min_detection_confidence=self.detectionCon, min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img):
        # Convertir l'image en RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        bbox = None

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            # Calculer la boîte englobante manuellement
            xList = [lm[1] for lm in lmList]
            yList = [lm[2] for lm in lmList]
            xmin, ymin = min(xList), min(yList)
            xmax, ymax = max(xList), max(yList)

            bbox = (xmin, ymin, xmax, ymax)  # Box de la main

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0),
                              2)  # Dessiner la boîte englobante
        return lmList, bbox

    def findDistance(self, p1, p2, img, draw=True):
        """
        Trouver la distance entre deux points de repère.
        p1: index du premier point de repère
        p2: index du second point de repère
        """
        lmList, _ = self.findPosition(img, draw=False)  # Ne pas dessiner les repères ici

        if len(lmList) != 0:
            x1, y1 = lmList[p1][1], lmList[p1][2]
            x2, y2 = lmList[p2][1], lmList[p2][2]

            # Calculer la distance euclidienne
            length = math.hypot(x2 - x1, y2 - y1)

            if draw:
                # Dessiner une ligne entre les deux points et un cercle à chaque extrémité
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)

            return length, img, [x1, y1, x2,
                                 y2]  # Retourner la distance, l'image modifiée, et les coordonnées des points
        return 0, img, []

    def fingersUp(self):
        """
        Déterminer quels doigts sont levés.
        Retourne une liste de 0 ou 1, chaque index représentant un doigt (0 = baissé, 1 = levé).
        """
        fingers = []
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                # Pouce
                if handLms.landmark[self.mpHands.HandLandmark.THUMB_TIP].y < handLms.landmark[
                    self.mpHands.HandLandmark.THUMB_IP].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Index
                if handLms.landmark[self.mpHands.HandLandmark.INDEX_FINGER_TIP].y < handLms.landmark[
                    self.mpHands.HandLandmark.INDEX_FINGER_DIP].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Majeur
                if handLms.landmark[self.mpHands.HandLandmark.MIDDLE_FINGER_TIP].y < handLms.landmark[
                    self.mpHands.HandLandmark.MIDDLE_FINGER_DIP].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Annulaire
                if handLms.landmark[self.mpHands.HandLandmark.RING_FINGER_TIP].y < handLms.landmark[
                    self.mpHands.HandLandmark.RING_FINGER_DIP].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Petit doigt
                if handLms.landmark[self.mpHands.HandLandmark.PINKY_TIP].y < handLms.landmark[
                    self.mpHands.HandLandmark.PINKY_DIP].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers
