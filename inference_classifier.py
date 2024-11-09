import pickle
import cv2
import mediapipe as mp
import numpy as np

# Charger le modèle et max_length
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
max_length = model_dict['max_length']

# Initialiser la capture vidéo
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'Zero', 1: 'Un', 2: 'Deux', 3: 'Trois', 4: 'Quatre', 5: 'Cinq', 6: 'Six', 7: 'Sept', 8: 'Huit', 9: 'Neuf',
               10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'K', 20: 'L', 21: 'M',
               22: 'N', 23: 'O', 24: 'P', 25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y'}

# Position de la zone de texte
text_position = (50, 50)
text_font = cv2.FONT_HERSHEY_SIMPLEX
text_scale = 1.3
text_thickness = 3
text_color = (0, 0, 0)
text_baseline = 5  # Marge de base sous le texte

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Assurez-vous que data_aux a la bonne longueur
        if len(data_aux) != max_length:
            data_aux = data_aux + [0] * (max_length - len(data_aux))

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        # Position de la zone de texte à côté de la caméra
        text_x = W + 50
        text_y = H - 50  # Position Y à la fin de la fenêtre

        # Affichez le texte dans la nouvelle zone de texte
        (text_width, text_height), baseline = cv2.getTextSize(predicted_character, text_font, text_scale, text_thickness)
        text_y -= text_height + text_baseline  # Ajustez la position Y pour centrer le texte
        cv2.putText(frame, predicted_character, (text_x, text_y), text_font, text_scale, text_color, text_thickness, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
