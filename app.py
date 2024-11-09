import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, Canvas
from PIL import Image, ImageTk
import pickle
import time

class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition App")

        self.cap = cv2.VideoCapture(0)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        self.labels_dict_chiffres = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}
        self.labels_dict_lettres = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}
        self.labels_dict = self.labels_dict_lettres

        self.current_model_path = './model_lettres.p'

        # Load the model and max_length
        self.load_model()

        self.video_frame = ttk.LabelFrame(root, text="Hand Gesture Detection")
        self.video_frame.pack(padx=10, pady=10, side=tk.LEFT)

        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()

        self.text_frame = ttk.LabelFrame(root, text="Detected Text")
        self.text_frame.pack(padx=10, pady=10, side=tk.RIGHT)

        self.text_label = tk.Text(self.text_frame, font=('Helvetica', 18), height=10, width=30)
        self.text_label.pack(expand=True, fill=tk.BOTH)

        self.clear_button = ttk.Button(self.text_frame, text="Clear Text", command=self.clear_text)
        self.clear_button.pack()

        self.numbers_button = ttk.Button(self.text_frame, text="Numbers", command=self.switch_to_numbers)
        self.numbers_button.pack()

        self.letters_button = ttk.Button(self.text_frame, text="Letters", command=self.switch_to_letters)
        self.letters_button.pack()

        self.list_numbers_button = ttk.Button(self.text_frame, text="List of Numbers", command=self.show_numbers_list)
        self.list_numbers_button.pack()

        self.list_letters_button = ttk.Button(self.text_frame, text="List of Letters", command=self.show_letters_list)
        self.list_letters_button.pack()

        self.previous_prediction = None
        self.current_prediction = None
        self.prediction_start_time = None
        self.prediction_duration = 1.5  # in seconds

        self.update()

    def clear_text(self):
        self.text_label.delete('1.0', tk.END)
        self.previous_prediction = None
        self.current_prediction = None
        self.prediction_start_time = None

    def switch_to_numbers(self):
        self.labels_dict = self.labels_dict_chiffres
        self.current_model_path = './model_chiffres.p'
        self.load_model()

    def switch_to_letters(self):
        self.labels_dict = self.labels_dict_lettres
        self.current_model_path = './model_lettres.p'
        self.load_model()

    def load_model(self):
        model_dict = pickle.load(open(self.current_model_path, 'rb'))
        self.model = model_dict['model']
        self.max_length = model_dict['max_length']

    def update(self):
        
        ret, frame = self.cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the video frame
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.video_label.photo = photo
        self.video_label.config(image=photo)

        self.root.after(0, self.process_frame, frame_rgb)

    def process_frame(self, frame_rgb):
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_rgb,  # image to draw
                    hand_landmarks,  # model output
                    self.mp_hands.HAND_CONNECTIONS,  # hand connections
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []
            x_ = []
            y_ = []

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

            if len(data_aux) != self.max_length:
                data_aux = data_aux + [0] * (self.max_length - len(data_aux))

            prediction = self.model.predict([np.asarray(data_aux)])
            predicted_character = self.labels_dict[int(prediction[0])]

            current_time = time.time()
            if self.current_prediction == predicted_character:
                if self.prediction_start_time and (current_time - self.prediction_start_time >= self.prediction_duration):
                    if predicted_character != self.previous_prediction:
                        self.text_label.insert(tk.END, predicted_character + '\n')
                        self.previous_prediction = predicted_character
                    self.current_prediction = None
                    self.prediction_start_time = None
            else:
                self.current_prediction = predicted_character
                self.prediction_start_time = current_time

        # Schedule the next frame processing
        self.root.after(10, self.update)

    def show_numbers_list(self):
        numbers_window = tk.Toplevel(self.root)
        numbers_window.title("List of Number Signs")
        canvas = Canvas(numbers_window)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(numbers_window, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor='nw')
        
        for number, sign in self.labels_dict_chiffres.items():
            frame_item = ttk.Frame(frame)
            frame_item.pack(padx=5, pady=5)
            img_path = fr"c:\Users\pybas\Documents\projet\projet\sign-language-detector-python\images\numbers\{number}.jpg"  # Update with correct path to images
            img = Image.open(img_path)
            img = img.resize((200, 200), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            img_label = ttk.Label(frame_item, image=photo)
            img_label.image = photo  # Keep a reference to avoid garbage collection
            img_label.pack(side=tk.LEFT)
            text_label = ttk.Label(frame_item, text=sign, font=('Helvetica', 18))
            text_label.pack(side=tk.LEFT)

        frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

    def show_letters_list(self):
        letters_window = tk.Toplevel(self.root)
        letters_window.title("List of Letter Signs")
        canvas = Canvas(letters_window)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(letters_window, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor='nw')
        
        for letter, sign in self.labels_dict_lettres.items():
            frame_item = ttk.Frame(frame)
            frame_item.pack(padx=5, pady=5)
            img_path = fr"c:\Users\pybas\Documents\projet\projet\sign-language-detector-python\images\letters\{sign}.jpg"  # Update with correct path to images
            img = Image.open(img_path)
            img = img.resize((200, 200), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            img_label = ttk.Label(frame_item, image=photo)
            img_label.image = photo  # Keep a reference to avoid garbage collection
            img_label.pack(side=tk.LEFT)
            text_label = ttk.Label(frame_item, text=sign, font=('Helvetica', 18))
            text_label.pack(side=tk.LEFT)

        frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

root = tk.Tk()
app = HandGestureApp(root)
root.mainloop()

