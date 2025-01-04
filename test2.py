import cv2
import numpy as np
import pytesseract
import random

# Configurar Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Definir los límites del color del bolígrafo en HSV
lower_color = np.array([100, 150, 50], dtype=np.uint8)  # Cambia estos valores según el color de tu bolígrafo
upper_color = np.array([140, 255, 255], dtype=np.uint8)

# Inicializar la cámara
def initialize_camera():
    cap = cv2.VideoCapture(0)
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
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    return binary

# Reconocer texto del dibujo
def recognize_text(canvas):
    binary = preprocess_image(canvas)
    text = pytesseract.image_to_string(binary, config='--psm 7 -c tessedit_char_whitelist=0123456789+-*/()=')
    return text.strip()

# Evaluar la expresión matemática
def evaluate_expression(expression):
    try:
        result = eval(expression)
        return result
    except Exception as e:
        print(f"Error evaluando la expresión: {e}")
        return None

# Generar una operación matemática aleatoria
def generate_operation():
    operators = ['+', '-', '*', '/']
    num1 = random.randint(1, 20)
    num2 = random.randint(1, 20)
    operator = random.choice(operators)
    if operator == '/':
        num1 = num1 * num2  # Asegurar divisiones exactas
    return f"{num1} {operator} {num2}", eval(f"{num1}{operator}{num2}")

# Dibujar menú principal
def draw_menu(frame):
    height, width, _ = frame.shape
    menu_frame = frame.copy()

    # Dibujar botones
    cv2.rectangle(menu_frame, (width // 4 - 100, height // 2 - 50), (width // 4 + 100, height // 2 + 50), (255, 0, 0), -1)
    cv2.putText(menu_frame, "Juego", (width // 4 - 60, height // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.rectangle(menu_frame, (3 * width // 4 - 100, height // 2 - 50), (3 * width // 4 + 100, height // 2 + 50), (0, 255, 0), -1)
    cv2.putText(menu_frame, "Calculadora", (3 * width // 4 - 90, height // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return menu_frame

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
    mode = "menu"  # Modo inicial
    target_expression = ""
    target_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        pen_tip = detect_pen(frame, lower_color, upper_color)

        if mode == "menu":
            menu_frame = draw_menu(frame)
            # if pen_tip:
            #     x, y = pen_tip
            #     if width // 4 - 100 < x < width // 4 + 100 and height // 2 - 50 < y < height // 2 + 50:
            #         mode = "game"
            #         target_expression, target_result = generate_operation()
            #         drawing_canvas = np.zeros((height, width, 3), dtype=np.uint8)
            #     elif 3 * width // 4 - 100 < x < 3 * width // 4 + 100 and height // 2 - 50 < y < height // 2 + 50:
            #         mode = "calculator"
            #         drawing_canvas = np.zeros((height, width, 3), dtype=np.uint8)
            # cv2.imshow("Dibujo Virtual", menu_frame)
        else:
            if pen_tip:
                x, y = pen_tip
                if 10 < x < 60 and 10 < y < 60:  # Icono de casa
                    mode = "menu"
                    continue
                cv2.circle(frame, pen_tip, 10, (0, 255, 0), -1)
                if drawing:
                    cv2.circle(drawing_canvas, pen_tip, 5, (0, 0, 255), -1)

            # Dibujar icono de casa
            cv2.rectangle(frame, (10, 10), (60, 60), (0, 255, 255), -1)
            cv2.putText(frame, "Casa", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            combined = cv2.addWeighted(frame, 0.5, drawing_canvas, 0.5, 0)

            if mode == "game" and target_expression:
                cv2.putText(combined, f"Resuelve: {target_expression}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Dibujo Virtual", combined)

            key = cv2.waitKey(1)
            if key == ord('d'):  # Alternar dibujo
                drawing = not drawing
            elif key == ord('c'):  # Limpiar el lienzo
                drawing_canvas = np.zeros((height, width, 3), dtype=np.uint8)
            elif key == ord('r'):  # Reconocer y evaluar
                expression = recognize_text(drawing_canvas)
                print(f"Expresión reconocida: {expression}")
                result = evaluate_expression(expression)
                print(f"Resultado: {result}")
                if mode == "calculator":
                    if result is not None:
                        cv2.putText(drawing_canvas, f"{expression} = {result}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif mode == "game":
                    if result == target_result:
                        print("Correcto!")
                        target_expression, target_result = generate_operation()
                        drawing_canvas = np.zeros((height, width, 3), dtype=np.uint8)
                    else:
                        print("Incorrecto. Intenta de nuevo.")
            elif key == 27:  # Salir
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
