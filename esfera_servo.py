import cv2
import numpy as np
import PyKDL
from Lib.EGM.Core import *
import threading
import time
import math
from scipy.spatial.transform import Rotation as R
from orientado import CentroidTransform

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

# solvers de cinemática inversa y directa construidos
ik_solver_vel = PyKDL.ChainIkSolverVel_pinv(chain)
fk_solver_pos = PyKDL.ChainFkSolverPos_recursive(chain)

# establezco conexión
Cls = EGM_Control_Cls('127.0.0.1', 6511)

# saco posición inicial del robot
q = PyKDL.JntArray(6)
q_a = (0, 24.83, 37.83, 0, 27.34, 0)
for i in range(6):
    q[i] = np.deg2rad(q_a[i])
H_base_tcp = PyKDL.Frame()
ret = fk_solver_pos.JntToCart(q, H_base_tcp)

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

def cart2sph(H_base_tcp, center_sphere):
    # posición del TCP respecto al centro de la esfera
    x = H_base_tcp.p.x() - center_sphere.x()
    y = H_base_tcp.p.y() - center_sphere.y()
    z = H_base_tcp.p.z() - center_sphere.z()

    # Calcula las coordenadas esféricas
    xy2 = x ** 2 + y ** 2
    r = np.sqrt(xy2 + z ** 2)
    theta = np.arccos(z / r)  # Ángulo polar
    phi = np.arctan2(y, x) # Ángulo azimutal

    #tcp_centro = (x, y, z)
    #print('tcp objetivo respecto a la base', H_base_tcp.p)
    #print('tcp objetivo respecto a la esfera', tcp_centro)

    return r, theta, phi

def sph2cart(r, theta, phi, center_sphere):
    # posición relativa del tcp repecto al centro de la esfera
    x_rel = r * math.sin(theta) * math.cos(phi)
    y_rel = r * math.sin(theta) * math.sin(phi)
    z_rel = r * math.cos(theta)

    # posición del tcp respecto a la base
    x = x_rel + center_sphere.x()
    y = y_rel + center_sphere.y()
    z = z_rel + center_sphere.z()

    #cart_center = (x_rel, y_rel, z_rel)
    cart_base = PyKDL.Vector(x, y, z)

    return cart_base


# Crea un objeto de captura de video desde la cámara
cap = cv2.VideoCapture(0)  # 0 para la cámara predeterminada
# utiliza los parámetros de distorsión para quitarla de la cámara
mtx = np.array([[472.44819903, 0., 323.81612906],
                [0., 472.65005792, 247.23094772],
                [0., 0., 1.]])
dist = np.array([0.22522434, -0.20108896, 0.00134874, -0.00602057, -0.08248137])

x_n = 0
y_n = 0


def do_detect():
    global x_n, y_n

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

        # Inicializa variables para el objeto más pequeño
        min_area = float('inf')
        des_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filtra contornos demasiado pequeños

                # Actualiza el objeto más pequeño si es necesario
                if area < min_area:
                    min_area = area
                    des_contour = contour

        if des_contour is not None:
            # Dibujo de un rectángulo alrededor del objeto
            x, y, w, h = cv2.boundingRect(des_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"({x},{y})", (x - 10, y - 10), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

            # coordenadas en píxeles respecto al tcp (desde el centro de frame
            # x hacia abajo e y hacia la izquierda)
            x_n = y - 240
            y_n = 320 - x

        else:
            x_n = 0
            y_n = 0

        # Saca el fotograma con el rectángulo
        cv2.imshow("Objeto Rojo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Presiona 'q' para salir
            break


thread = threading.Thread(target=do_detect)
thread.start()

print('starting')

# defino VARIABLES para el bucle

delta_t = 0.01  # [s]
radio = 200

V_FACTOR = 0.00001

# factor de rotación
f_rot = -0.002  # el - es porque se debe orientar hacia la esfera no hacia la dirección de movimiento


based_center = H_base_tcp * PyKDL.Vector(0, 0, radio)  # expreso la posición  de la esfera respecto a la base
#print('centro respecto a la base', based_center)

while True:
    pos = PyKDL.Vector(x_n, y_n, 0)
    if pos.Norm() != 0.0:  # si no está ya centrado

        # TRANSLACIÓN
        H = H_base_tcp.__copy__()

        H.p = H * pos

        r, theta, phi = cart2sph(H, based_center)

        if phi < np.deg2rad(-179):
            print(np.rad2deg(phi))
            phi += 2 * np.pi
            print('cambio')

        # PROYECTO LA POSICIÓN DESEADA
        pos_d = sph2cart(radio, theta, phi, based_center)

        # ORIENTACIÓN
        tcp_axis = PyKDL.Rotation.RotZ(np.pi / 2) * pos
        base_axis = H_base_tcp.M * tcp_axis * f_rot

        # comando la posición
        v_pos = PyKDL.Twist(pos_d - H_base_tcp.p, base_axis)

        H_base_tcp, q = move_robot(ik_solver_vel, fk_solver_pos, Cls, v_pos, q, delta_t)

        print('theta, phi', np.rad2deg(theta), ',', np.rad2deg(phi))

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Presiona 'q' para salir
        break
    time.sleep(delta_t)

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
