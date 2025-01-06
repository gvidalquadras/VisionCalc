import cv2
import numpy as np

lower_color = np.array([100, 150, 50], dtype=np.uint8)  
upper_color = np.array([140, 255, 255], dtype=np.uint8)

def initialize_camera():
    '''
    Inicialización (y verificación) de la cámara para capturar imágenes en tiempo real.
    Return: Imagen capturada por la cámara en cada instante (en timepo real)
    '''

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return None
    return cap

def detect_pen(frame, lower_color, upper_color):
    '''
    Detección del bolígrafo por color en el espacio HSV.
    Args: 
        - frame (ndarray): Fotograma en formato BGR
        - lower_color (int): Límite inferior del color del bolígrafo en el esapcio HSV.
        - upper_color (int): Límite superior del color del bolígrafo en el esapcio HSV

    Return: None

    '''

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 500:
            (x, y), radius = cv2.minEnclosingCircle(max_contour)
            center = (int(x), int(y))
            if radius > 5:
                return center
            
    return None

# Función para resetear la secuencia
def reset_sequence():
    global user_sequence
    user_sequence = []
    print("Secuencia reseteada.")


def detect_shape(contour):
    # Aproximar el contorno
    contour = np.array(contour, dtype=np.int32)
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    if len(approx) == 3:
        return "triangle"
    elif len(approx) == 4:
        return "square"
    elif len(approx) > 5:
        return "circle"
    return "unknown"

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    cv2.imwrite("img.png", threshold)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    if contours:
        for contour in contours:
            # contour = np.array(contour, dtype=np.int32)
            if cv2.contourArea(contour) > 500:  # Filtrar contornos pequeños
                try:
                    shape = detect_shape(contour)
                    return shape
                except cv2.error as e:
                    print(f"Error procesando contorno: {e}")
    return "unknown"

def validate_sequence(user_sequence, sequence):
    
    if user_sequence == sequence:
        print("Secuencia correcta. Acceso permitido.")
        return True
    else:
        print("Secuencia incorrecta. Acceso denegado.")
        return False

def main():
    cap = initialize_camera()
    if not cap:
        return

    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el fotograma inicial.")
        cap.release()
        return

    height, width, _ = frame.shape
    drawing_canvas = np.zeros((height, width, 3), dtype=np.uint8)
    drawing = False
    sequence = ["circle"]
    user_sequence = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        pen_tip = detect_pen(frame, lower_color, upper_color)

        # Agregar el texto centrado en la parte superior
        text = "Introduzca el patron de desbloqueo"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (width - text_size[0]) // 2  # Posición horizontal centrada
        text_y = 50  # Posición vertical (ajusta si es necesario)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        if user_sequence:
            # Mostrar la secuencia introducida debajo
            sequence_text = "Secuencia introducida: " + '+'.join(user_sequence)
            sequence_size = cv2.getTextSize(sequence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            sequence_x = (width - sequence_size[0]) // 2  # Posición horizontal centrada
            sequence_y = text_y + 40  # 40 píxeles más abajo que el texto de la contraseña
            cv2.putText(frame, sequence_text, (sequence_x, sequence_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


        if pen_tip:
            cv2.circle(frame, pen_tip, 10, (0, 255, 0), -1)
            if drawing:
                cv2.circle(drawing_canvas, pen_tip, 5, (0, 0, 255), -1)

        combined = cv2.addWeighted(frame, 0.5, drawing_canvas, 0.5, 0)
        cv2.imshow("Dibujo Virtual", combined)

        
        key = cv2.waitKey(1)
        if key == ord('d'):  # Alternar dibujo
            drawing = not drawing
        elif key == ord('c'):  # Limpiar el lienzo
            drawing_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        elif key == ord('a'):
            contour = process_frame(drawing_canvas)
            if contour != "unknown":
                user_sequence.append(contour)
                count += 1
            else:
                print("Repite")

        elif key == ord("r"):
            print(user_sequence)
            if user_sequence == sequence:
                print("Desbloqueado")
                return True  
            else:
                print("Inténtalo de nuevo")
                user_sequence = []
                return False 

        elif key == 27:  # Salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

