import cv2
import numpy as np
import PyKDL
from Lib.EGM.Core import *
import threading
import time
import math

chain = PyKDL.Chain()

H_0_1 = PyKDL.Frame.DH(0, np.deg2rad(-90), 265, np.deg2rad(0))
H_1_2 = PyKDL.Frame.DH(444, np.deg2rad(0), 0, np.deg2rad(-90))
H_2_3 = PyKDL.Frame.DH(110, np.deg2rad(-90), 0, np.deg2rad(0))
H_3_4 = PyKDL.Frame.DH(0, np.deg2rad(90), 470, np.deg2rad(0))
H_4_5 = PyKDL.Frame.DH(80, np.deg2rad(-90), 0, np.deg2rad(0))
H_5_6 = PyKDL.Frame.DH(0, np.deg2rad(0), 101, np.deg2rad(180))


axis = PyKDL.Joint(PyKDL.Joint.RotZ)

chain.addSegment(PyKDL.Segment(axis, H_0_1))
chain.addSegment(PyKDL.Segment(axis, H_1_2))
chain.addSegment(PyKDL.Segment(axis, H_2_3))
chain.addSegment(PyKDL.Segment(axis, H_3_4))
chain.addSegment(PyKDL.Segment(axis, H_4_5))
chain.addSegment(PyKDL.Segment(axis, H_5_6))

#solvers de cinemática inversa y directa construidos
ik_solver_vel = PyKDL.ChainIkSolverVel_pinv(chain)
fk_solver_pos = PyKDL.ChainFkSolverPos_recursive(chain)


#establezco conexión
Cls = EGM_Control_Cls('127.0.0.1', 6511)

def move_robot(ik_solver_vel, fk_solver_pos, Cls, v, q, delta_t):
    # Variables definidas en la función

    q_dot = PyKDL.JntArray(6)
    H_base_tcp = PyKDL.Frame()

    # hago cinemática inversa de velocidad
    ret = ik_solver_vel.CartToJnt(q, v, q_dot)

    if ret < 0:
        print('error de singularidad')
    else:
        # convierto la velocidad angular a posición de articulaciones
        for i in range(6):
            q[i] += q_dot[i] * delta_t

        # Extraigo los valores de q en una lista
        q_arr = [np.rad2deg(q[i]) for i in range(6)]

        # Limites articulares
        limits = np.array([[-178, 178],  # Joint 1
                           [-178, 178],  # Joint 2
                           [-223, 83],  # Joint 3
                           [-178, 178],  # Joint 4
                           [-178, 178],  # Joint 5
                           [-268, 268]])  # Joint 6

        # Limito los valores de q dentro de los límites articulares
        q_lim = np.clip(q_arr, limits[:, 0], limits[:, 1])
        flag = True
        for i in range(6):
            if q_lim[i] != q_arr[i]:
                print(f"La articulación {i + 1} ha llegado al límite en {q_lim[i]} grados")
                flag = False

        # doy la orden de posición

        if flag:
            Cls.Set_Absolute_Joint_Position(np.array(q_lim), False)

        # devuelvo los valores limitados a q
        for i in range(6):
            q[i] = np.deg2rad(q_lim[i])

        # hallo nueva posición
        fk_solver_pos.JntToCart(q, H_base_tcp)
    return H_base_tcp, q


# defino variables para el bucle
V_FACTOR = 0.3  # [mm]  (0.1, 5)
Avance = 50
delta_t = 0.01  # [s]


q = PyKDL.JntArray(6)
q_dot = PyKDL.JntArray(6)
H_base_tcp = PyKDL.Frame()

ret = fk_solver_pos.JntToCart(q, H_base_tcp)


# Crea un objeto de captura de video desde la cámara
cap = cv2.VideoCapture(0)  # 0 para la cámara predeterminada
# utiliza los parámetros de distorsión para quitarla de la cámara
mtx = np.array([[472.44819903, 0., 323.81612906],
                [0., 472.65005792, 247.23094772],
                [0., 0., 1.]])
dist = np.array([0.22522434, -0.20108896, 0.00134874, -0.00602057, -0.08248137])

x_n = 0
y_n = 0
z_n = 0

def do_detect():
    global x_n, y_n, z_n

    while True:
        ret, frame1 = cap.read()  # Lee un fotograma
        frame = cv2.undistort(frame1, mtx, dist, None, mtx)  # quita la distorsión

        # Convierte el fotograma de BGR a HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Especificación del rango de color rojo
        lower_red = np.array([159, 100, 100])
        upper_red = np.array([180, 255, 255])

        # Crea una máscara para el color deseado
        mask = cv2.inRange(hsv_frame, lower_red, upper_red)

        # Encuentra los contornos en la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Inicializa variables para el objeto más cercano
        min_distance = float('inf')
        closest_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filtra contornos pequeños
                x, y, w, h = cv2.boundingRect(contour)
                z = area  # Usamos el área como una estimación de la coordenada z

                # Calculo de la distancia al objeto
                distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)

                # Actualiza el objeto más cercano si es necesario
                if distance < min_distance:
                    min_distance = distance
                    closest_contour = contour

        if closest_contour is not None:
            # Dibujo de un rectángulo alrededor del objeto más cercano
            x, y, w, h = cv2.boundingRect(closest_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"({x},{y},{z})", (x - 10, y - 10), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

            x_n = y - 240
            y_n = 320 - x
            z_n = Avance
        else:
            x_n = 0
            y_n = 0
            z_n = 0

        # Saca el fotograma con el rectángulo
        cv2.imshow("Objeto Rojo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Presiona 'q' para salir
            break


thread = threading.Thread(target=do_detect)
thread.start()

print('starting')

while True:
    pos = PyKDL.Vector(x_n, y_n, z_n)

    if pos.Norm() != 0.0:
        # saco posición deseada respecto a la base
        p_d = H_base_tcp * (pos*V_FACTOR)

        # creo el twist y muevo el robot
        v = PyKDL.Twist(p_d - H_base_tcp.p, PyKDL.Vector(0, 0, 0))
        H_base_tcp, q = move_robot(ik_solver_vel, fk_solver_pos, Cls, v, q, delta_t)

    time.sleep(delta_t)

# Liberaq recursos
cap.release()
cv2.destroyAllWindows()

# color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],
#   'white': [[180, 18, 255], [0, 0, 231]],
#  'red1': [[180, 255, 255], [159, 50, 70]],
# 'red2': [[9, 255, 255], [0, 50, 70]],
# 'green': [[89, 255, 255], [36, 50, 70]],
#  'blue': [[128, 255, 255], [90, 50, 70]],
# 'yellow': [[35, 255, 255], [25, 50, 70]],
# 'purple': [[158, 255, 255], [129, 50, 70]],
#   'orange': [[24, 255, 255], [10, 50, 70]],
#  'gray': [[180, 18, 230], [0, 0, 40]]}
