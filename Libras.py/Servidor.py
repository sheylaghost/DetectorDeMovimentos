import cv2
import mediapipe as mp
import socket

# Cria servidor socket
HOST = '127.0.0.1' 
PORT = 5001

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()

print("Servidor Python rodando...")

conn, addr = server.accept()
print(f"Conectado em {addr}")

# MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands 
mp_gestures = mp.solutions.gesture_recognition

cap = cv2.VideoCapture(0) # Tente cv2.VideoCapture(1) se não funcionar

# Adiciona o modelo de reconhecimento de gestos do MediaPipe
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    with mp_gestures.GestureRecognizer() as recognizer:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Processa a mão e o gesto
            hand_results = hands.process(rgb)
            gesture_results = recognizer.recognize(rgb)

            gesto_detectado = ""

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Usa o resultado do modelo de gestos
                    if gesture_results.gestures:
                        gesto_detectado = gesture_results.gestures[0].category_name
                        
                        # Exibe a tradução na tela
                        cv2.putText(frame, gesto_detectado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Se um gesto foi detectado, envia para o cliente
            if gesto_detectado != "":
                try:
                    conn.send((gesto_detectado + "\n").encode('utf-8'))
                    print(f"Enviado: {gesto_detectado}")
                except BrokenPipeError:
                    print("Cliente desconectou.")
                    break

            cv2.imshow("Detecção de Mãos", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()
conn.close()