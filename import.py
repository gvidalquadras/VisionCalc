import cv2
import numpy as np

lower_color = np.array([100, 150, 50], dtype=np.uint8)  
upper_color = np.array([140, 255, 255], dtype=np.uint8)

def initialize_camera():
    '''
    Inicialización (y verificación) de la cámara para capturar imágenes en tiempo real.
    Return: Imagen capturada por la cámara en cada instante (en tiempo real)
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

def detect_shape(contour):
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
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                try:
                    shape = detect_shape(contour)
                    return shape
                except cv2.error as e:
                    print(f"Error procesando contorno: {e}")
    return "unknown"

def draw_red(frame, width, height):
    # Definir áreas de los botones
    button_width = 150
    button_height = 50
    button_padding = 10
    button_y = height - 60

    # Botón rojo ("Borrar todo")
    red_button = (width // 2 - button_width // 2, button_y)
    cv2.rectangle(frame, red_button, (red_button[0] + button_width, red_button[1] + button_height), (0, 0, 255), -1)
    cv2.putText(frame, "Borrar", (red_button[0] + 20, red_button[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    return red_button

def check_button_press(pen_tip, red_button):
    if pen_tip:
        x, y = pen_tip
        # Verificar si el bolígrafo está dentro del área del botón rojo (Borrar todo)
        if red_button[0] <= x <= red_button[0] + 200 and red_button[1] <= y <= red_button[1] + 50:
            return "red"  # Borrar todo
        
    return None

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
    sequence = ["circle", "circle", "circle"]
    user_sequence = []

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

        # Mostrar la secuencia introducida
        if user_sequence:
            sequence_text = "Secuencia introducida: " + '+'.join(user_sequence)
            sequence_size = cv2.getTextSize(sequence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            sequence_x = (width - sequence_size[0]) // 2
            sequence_y = text_y + 40
            cv2.putText(frame, sequence_text, (sequence_x, sequence_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Dibuja los botones
        red_button = draw_red(frame, width, height)

        if pen_tip:
            cv2.circle(frame, pen_tip, 10, (0, 255, 0), -1)
            if drawing:
                cv2.circle(drawing_canvas, pen_tip, 5, (0, 0, 255), -1)

        combined = cv2.addWeighted(frame, 0.5, drawing_canvas, 0.5, 0)
        cv2.imshow("Dibujo Virtual", combined)

        button_press = check_button_press(pen_tip, red_button)
        if button_press == "red":
            # Borrar toda la secuencia
            user_sequence = []

        key = cv2.waitKey(1)
        if key == ord('d'):  # Alternar dibujo
            drawing = not drawing
        elif key == ord('c'):  # Limpiar el lienzo
            drawing_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        elif key == ord('a'):
            contour = process_frame(drawing_canvas)
            if contour != "unknown":
                user_sequence.append(contour)
            else:
                print("Repite")

        elif key==ord("r"):
            if user_sequence == sequence:
                print("Desbloqueado")
            else:
                print("Inténtalo de nuevo")
                user_sequence = []
        elif key == 27:  # Salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
