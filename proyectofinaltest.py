import cv2
import numpy as np
import pytesseract
from utils import non_max_suppression, get_hsv_color_ranges
import random

lower_color = np.array([100, 150, 50], dtype=np.uint8)  
upper_color = np.array([140, 255, 255], dtype=np.uint8)


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



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


def preprocess_image(canvas, rect_start, rect_end):
    '''
    Preprocesamiento de la imagen 
    CAMBIAR PARA NUEVO MÉTODO?????
    '''

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray.png", gray)
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV) 
    cv2.imwrite("imagen_procesada.png", binary)
    return binary


def recognize_text(canvas, rect_start, rect_end):
    '''
    Reconocimiento del texto
    '''
    binary = preprocess_image(canvas, rect_start, rect_end)
    text = pytesseract.image_to_string(binary, config='--psm 7 -c tessedit_char_whitelist=0123456789+-*/()')
    return text.strip()


def evaluate_expression(expression):
    '''
    Detección del bolígrafo por color en el espacio HSV.
    Args: 
        - expression (string): Expresión matemática a resolver.

    Return: El resultado de la expresión si no existe ningún error, 
    en caso de ocurra un error, devuelve None.

    '''
    try:
        result = eval(expression)  
        return result
    except Exception as e:
        print(f"Error evaluando la expresión: {e}")
        return None
    

def generate_operation():
    '''
    Generación de una operación matemática simple aleatoria para el modo "juego" del programa.
    Return: (string) Expresión generada.
    '''
    operators = ['+', '-', '*', '/']
    num1 = random.randint(1, 20)
    num2 = random.randint(1, 20)
    operator = random.choice(operators)
    if operator == '/':
        num1 = num1 * num2  # Asegurar divisiones exactas para facilidad de los cálculos
    return f"{num1} {operator} {num2}", eval(f"{num1}{operator}{num2}")


def draw_menu(frame):
    '''
    Dibuja el menú principal con dos botones para elegir el modo del programa: "Juego" y "Calculadora"
    Args:
        - frame (ndarray): Fotograma donde dibujar el menú.
    Return: 
        - menu_frame (ndarray): Imagen con el menú dibujado.
    '''

    height, width, _ = frame.shape
    menu_frame = frame.copy()

    # Dibujar botones
    cv2.rectangle(menu_frame, (width // 4 - 100, height // 2 - 50), (width // 4 + 100, height // 2 + 50), (255, 0, 0), -1)
    cv2.putText(menu_frame, "Juego", (width // 4 - 60, height // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.rectangle(menu_frame, (3 * width // 4 - 100, height // 2 - 50), (3 * width // 4 + 100, height // 2 + 50), (0, 255, 0), -1)
    cv2.putText(menu_frame, "Calculadora", (3 * width // 4 - 90, height // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return menu_frame


def main():
    '''
    Programa principal.
    '''

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
    mode = "menu"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        pen_tip = detect_pen(frame, lower_color, upper_color)
        
        height, width, _ = frame.shape

        # Tamaño recuadro
        rect_width = width - (width // 4)  
        rect_height = height // 2 

        # Calcular las coordenadas con el desplazamiento en Y
        rect_start = (width // 2 - rect_width // 2, height // 2 - rect_height // 2 + 100)  # Esquina superior izquierda
        rect_end = (width // 2 + rect_width // 2, height // 2 + rect_height // 2 + 100)  # Esquina inferior derecha

        if mode == "menu":

            drawing_canvas = np.zeros((height, width, 3), dtype=np.uint8)
            # Dibujar botones
            cv2.rectangle(frame, (width // 4 - 100, height // 2 - 50), (width // 4 + 100, height // 2 + 50), (255, 0, 0), -1)
            cv2.putText(frame, "Juego", (width // 4 - 60, height // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.rectangle(frame, (3 * width // 4 - 100, height // 2 - 50), (3 * width // 4 + 100, height // 2 + 50), (0, 255, 0), -1)
            cv2.putText(frame, "Calculadora", (3 * width // 4 - 90, height // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if pen_tip:
                x, y = pen_tip
                if width // 4 - 100 < x < width // 4 + 100 and height // 2 - 50 < y < height // 2 + 50:
                    mode = "game"
                    print("mode=game")
                    resolved = True
                    drawing_canvas = np.zeros((height, width, 3), dtype=np.uint8)
                elif 3 * width // 4 - 100 < x < 3 * width // 4 + 100 and height // 2 - 50 < y < height // 2 + 50:
                    mode = "calculator"
                    drawing_canvas = np.zeros((height, width, 3), dtype=np.uint8)


        elif mode == "calculator":

            cv2.rectangle(frame, (width - 150, 20), (width - 20, 70), (255, 0, 255), -1)
            cv2.putText(frame, "Menu", (width - 130, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if pen_tip:
                x, y = pen_tip
                if width - 150 < x < width - 20 and 20 < y < 70:  # Recuadro Menú
                    mode = "menu"
                    continue
                cv2.circle(frame, pen_tip, 10, (0, 255, 0), -1)
                
        elif mode == "game":

            cv2.rectangle(frame, (width - 150, 20), (width - 20, 70), (255, 0, 255), -1)
            cv2.putText(frame, "Menu", (width - 130, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if resolved:
                target_expression, target_result = generate_operation()
                resolved = False
    

            if pen_tip:
                x, y = pen_tip
                if width - 150 < x < width - 20 and 20 < y < 70:  # Recuadro Menú
                    mode = "menu"
                    continue    
                cv2.circle(frame, pen_tip, 10, (0, 255, 0), -1)

            cv2.putText(drawing_canvas, f"{target_expression} = ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.rectangle(frame, rect_start, rect_end, (255, 255, 255), 2)

            text = "Escriba su respuesta dentro del rectangulo"
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 0.5  # Tamaño del texto
            thickness = 1  # Grosor del texto

            # Calcular el tamaño del texto para centrarlo
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = rect_start[0] + (rect_width - text_width) // 2  # Centrado horizontalmente dentro del rectángulo
            text_y = rect_height - text_height*3  # Centrado verticalmente dentro del rectángulo
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255,255,255), thickness)
            

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
        elif key == ord("u"):
            target_expression, target_result = generate_operation()
            drawing_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        elif key == ord('r'):  # Reconocer y evaluar
            
            if mode == "calculator":
                expression = recognize_text(drawing_canvas, rect_start, rect_end)
                result = evaluate_expression(expression)
            
                print(f"Expresión reconocida: {expression}")
                print(f"Resultado: {result}")
                if result is not None:
                    cv2.putText(drawing_canvas, f"{expression} = {result}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif mode == "game":

                canvas = drawing_canvas[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]]
                expression = recognize_text(canvas, rect_start, rect_end)
                result = evaluate_expression(expression)

                if result == target_result:
                    resolved = True
                    drawing_canvas = np.zeros((height, width, 3), dtype=np.uint8)

                    
                else:
                    print(f"Expresión reconocida: {expression}")
                    print("Incorrecto. Intenta de nuevo.")

        elif key == 27:  # Salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

