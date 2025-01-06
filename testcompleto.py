import sistemaSeguridad
import proyectofinaltest 

def main():
    print("Iniciando el sistema de seguridad...")
    desbloqueado = False
    # Iniciar el sistema de seguridad
    while not desbloqueado: 
        desbloqueado = sistemaSeguridad.main()

        if desbloqueado:
            print("Acceso permitido. Iniciando la calculadora virtual...")
        else:
            print("Acceso denegado. Sistema bloqueado.")
    proyectofinaltest.main()
if __name__ == "__main__":

    main()
