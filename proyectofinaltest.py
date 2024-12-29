import cv2
import numpy as np
import pytesseract
from utils import non_max_suppression, get_hsv_color_ranges

# Definir los límites del color del bolígrafo en HSV
lower_color = np.array([100, 150, 50], dtype=np.uint8)  # Cambia estos valores según el color de tu bolígrafo
upper_color = np.array([140, 255, 255], dtype=np.uint8)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


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

# Preprocesar la imagen para el OCR
def preprocess_image(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray.png", gray)
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)  # Fondo blanco, trazos negros
    cv2.imwrite("imagen_procesada.png", binary)
    return binary

# Reconocer texto del dibujo
def recognize_text(canvas):

    binary = preprocess_image(canvas)
    # text = pytesseract.image_to_string(binary, config='--psm 7')  # Reconocer como una línea de texto
    text = pytesseract.image_to_string(binary, config='--psm 7 -c tessedit_char_whitelist=0123456789+-*/()')
    return text.strip()

# Evaluar la expresión matemática
def evaluate_expression(expression):
    try:
        result = eval(expression)  # Evalúa la expresión
        return result
    except Exception as e:
        print(f"Error evaluando la expresión: {e}")
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

        frame = cv2.flip(frame, 1)
        pen_tip = detect_pen(frame, lower_color, upper_color)

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
        elif key == ord('r'):  # Reconocer y evaluar
            expression = recognize_text(drawing_canvas)
            cv2.imwrite("imagen.png", drawing_canvas)
            print(f"Expresión reconocida: {expression}")
            result = evaluate_expression(expression)
            print(f"Resultado: {result}")
            if result is not None:
                cv2.putText(drawing_canvas, f"= {result}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif key == 27:  # Salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

