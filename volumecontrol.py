import cv2
from handtrackingmodule import handDetector
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


# Classe pour le contrôle du volume
class VolumeControl:
    def __init__(self):
        # Obtenir l'instance de volume du périphérique audio
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(
            IAudioEndpointVolume._iid_,
            1,
            None
        )
        self.volume = self.interface.QueryInterface(IAudioEndpointVolume)

    def set_volume(self, level):
        # Le niveau doit être entre 0.0 et 1.0
        level = max(0.0, min(level, 1.0))
        self.volume.SetMasterVolumeLevelScalar(level, None)  # Ajuste le volume

    def get_volume(self):
        # Retourner le niveau actuel du volume
        return self.volume.GetMasterVolumeLevelScalar()


# Initialiser la détection des mains
detector = handDetector()

# Initialiser le contrôle du volume
volume_control = VolumeControl()

# Démarrer la capture vidéo
cap = cv2.VideoCapture(0)

# Largeur de la barre en pixels (2 cm ~ 76 pixels)
bar_width = 45  # Largeur de la barre
bar_height = 302  # Hauteur de la barre (8 cm ~ 302 pixels)
bar_x = 50  # Position x de la barre
bar_y = 100  # Position y de la barre (base de la barre)

while True:
    ret, img = cap.read()

    # Trouver les mains dans l'image
    img = detector.findHands(img)

    # Récupérer la position des points clés de la main
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        # Exemple : utiliser la distance entre les points 4 (pouce) et 8 (index) pour ajuster le volume
        length, img, _ = detector.findDistance(4, 8, img)

        # Calculer le volume en fonction de la distance (volume inversé)
        volume_level = min(length / 300, 1)  # Limiter la valeur entre 0 et 1 (distance inversée)

        # Ajuster le volume
        volume_control.set_volume(volume_level)

        # Afficher le volume à l'écran
        cv2.putText(img, f"Volume: {int(volume_level * 100)}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Afficher la barre de progression pour le volume (barre verticale)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 3)  # Cadre de la barre
        cv2.rectangle(img, (bar_x, bar_y + int((1 - volume_level) * bar_height)),
                      (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), -1)  # Remplissage de la barre

    # Afficher l'image avec le volume
    cv2.imshow("Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
