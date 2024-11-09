import cv2

# Assurez-vous que vous utilisez le bon index de caméra
cap = cv2.VideoCapture(0)  # Essayez avec 0, 1, 2, etc.

if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la caméra")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur: Impossible de lire la caméra")
        break

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
