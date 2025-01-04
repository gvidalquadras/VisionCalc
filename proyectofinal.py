import cv2
import numpy as np

# Definir los límites del color del bolígrafo en HSV
# Cambia estos valores según el color de tu bolígrafo
lower_color = np.array([156, 106, 103], dtype=np.uint8)  
upper_color = np.array([200, 200, 200], dtype=np.uint8)  

# Inicializar la cámara
def initialize_camera():
    cap = cv2.VideoCapture(0)  # Usar la cámara en tiempo real
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return None
    return cap

# Detectar el bolígrafo
def detect_pen(frame, lower_color, upper_color):
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

# Programa principal
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.rectangle(frame,(0,0),(50,50),(0,0,255), 2)
        cv2.line(frame, (0, 0), (50, 50), (0, 0,255), 2)  # Línea de color rojo
        cv2.line(frame, (50, 0), (0, 50), (0, 0, 255), 2)  # Línea de color rojo
        
        frame = cv2.flip(frame, 1)
        pen_tip = detect_pen(frame, lower_color, upper_color)

        if pen_tip:
            cv2.circle(frame, pen_tip, 10, (0, 255, 0), -1)
            if drawing:
                cv2.circle(drawing_canvas, pen_tip, 5, (255, 0, 0), -1)
        
        combined = cv2.addWeighted(frame, 0.5, drawing_canvas, 0.5, 0)
        cv2.imshow("Dibujo Virtual", combined)

        key = cv2.waitKey(1)
        if key == ord('d'):  # Alternar dibujo
            drawing = not drawing
        elif key == ord('c'):  # Limpiar el lienzo
            drawing_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        elif key == 27:  # Salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
