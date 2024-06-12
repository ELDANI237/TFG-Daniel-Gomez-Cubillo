#
#
# Este código no forma parte del tfg pero se creó para estudiar las coordenadas esféricas de cara a preparar la aplicación práctica
#
#


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

Cls = EGM_Control_Cls('127.0.0.1', 6511)
q = PyKDL.JntArray(6)
delta_t = 0.01


H_base_tcp = PyKDL.Frame()
ret = fk_solver_pos.JntToCart(q, H_base_tcp)

v = PyKDL.Twist(PyKDL.Vector(500, 0, 0), PyKDL.Vector(0, 0, 0))


def move_robot(ik_solver_vel, fk_solver_pos, Cls, v, q, delta_t):
    # Variables definidas en la función

    q_dot = PyKDL.JntArray(6)
    H_base_tcp = PyKDL.Frame()

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

        for i in range(6):
            if q_lim[i] != q_arr[i]:
                print(f"La articulación {i + 1} ha llegado al límite en {q_lim[i]} grados")

        # doy la orden de posición
        Cls.Set_Absolute_Joint_Position(np.array(q_lim), False)

        # hallo nueva posición
        fk_solver_pos.JntToCart(q, H_base_tcp)
    return H_base_tcp, q


def cart2sph(H_base_tcp, center_sphere):
    # posición del TCP respecto al centro de la esfera
    x = H_base_tcp.p.x() - center_sphere.x()
    y = H_base_tcp.p.y() - center_sphere.y()
    z = H_base_tcp.p.z() - center_sphere.z()

    # Calcular las coordenadas esféricas
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)  # Ángulo polar
    #phi = np.arctan2(y, x)  # Ángulo azimutal
    #print(x, y)
    phi = np.arctan2(y, x)

    return r, theta, phi


def sgn(value):
    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 0


def cartesian_to_spherical(H_base_tcp, center_sphere):
    # posición del TCP respecto al centro de la esfera
    x = H_base_tcp.p.x() - center_sphere.x()
    y = H_base_tcp.p.y() - center_sphere.y()
    z = H_base_tcp.p.z() - center_sphere.z()

    #conversión
    r = math.sqrt(x ** 2 + y ** 2 + z ** 2)

    if r == 0:
        theta = 0
        phi = 0
    else:
        if z > 0:
            theta = math.atan(math.sqrt(x ** 2 + y ** 2) / z)
        elif z == 0:
            theta = math.pi / 2
        else:
            theta = math.pi + math.atan(math.sqrt(x ** 2 + y ** 2) / z)

        if x > 0 and y >= 0:  # 1er cuadrante
            phi = math.atan(y / x)
        elif x > 0 and y < 0:  # 4º cuadrante
            phi = 2 * math.pi + math.atan(y / x)
        elif x == 0:
            phi = math.pi / 2 * sgn(y)
        else:  # x < 0 (2º y 3er cuadrante)
            phi = math.pi + math.atan(y / x)

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

    cart_center = (x_rel, y_rel, z_rel)
    cart_base = PyKDL.Vector(x, y, z)

    return cart_base


def tcp_direction_center(tcp_base, center_sphere):
    direction = PyKDL.Vector(
        center_sphere.x() - tcp_base.x(),
        center_sphere.y() - tcp_base.y(),
        center_sphere.z() - tcp_base.z()
    )
    direction.Normalize()
    return direction
#J1: 0,00
#J2: 43,14
#J3: 6,66
#J4: 0,00
#J5: 40,20
#J6: 0,00
#Cfg: (0,0,0,0)

# ESFERA
inc = 5
rot_factor = 0
center = PyKDL.Vector(580, 0, 0)  # posición inicial robot = 571, 0, 900
radio = 200
porciones = 4

# movimiento inicial: polar
polar = True
azimutal = False

onzero = False
count = 0
funcionando = True

part = math.pi / porciones
while funcionando:
    #saco coordenadas esféricas
    r, theta, phi = cart2sph(H_base_tcp, center)
    # r, theta, phi = cartesian_to_spherical(H_base_tcp, center)

    # aumento el angulo polar de tcp respecto al centro de la esfera
    if polar:
        theta += np.deg2rad(inc)
        print('Theta', np.rad2deg(theta), 'phi', np.rad2deg(phi), 'r', r)

        #si llega a los límites de la semiesfera se pasa a azimutal
        if np.rad2deg(theta) >= 85 or np.rad2deg(theta) <= 0.001:
            polar = False
            phi_d = phi + part

            # en 183 grados expresa phi en negativo
            if np.rad2deg(phi_d) > 183:
                print('phi_d antes',np.rad2deg(phi_d))
                phi_d -= 2 * math.pi


            print("")
            print('--------------------------------------------------------iteración', count, 'phi deseada', np.rad2deg(phi_d))
            print("polar off")
            print("")
            count += 1
            #si llega a 0 se cambia el incremento para que siga en la misma dirección azimutal
            if np.rad2deg(theta) <=0.001:

                onzero = True
                inc = -inc
            elif np.rad2deg(theta) >= 3:
                onzero = False

            azimutal = True

    if azimutal:
        phi += np.deg2rad(inc)
        print('Theta', np.rad2deg(theta), 'phi:', np.rad2deg(phi), 'de',np.rad2deg(phi_d) )

        #si llega al valor deseado se pasa al polar
        if phi >= phi_d and phi <= (phi_d + np.deg2rad(10)):
            azimutal = False
            print('azimutal off por phi= ',np.deg2rad(phi))
            if not onzero:
                inc = -inc
            print('inc',inc)
            polar = True

    #número de repeticiones
    if count >= (2 * porciones) + porciones/2:
        print(f"se han completado las {porciones} porciones de esfera")
        funcionando = 0

    #radio de la esfera
    r = radio
    # saco coordenadas del tcp respecto a la base
    tcp_base = sph2cart(r, theta, phi, center)

    # vector hacia el centro desde el tcp
    c = tcp_direction_center(tcp_base, center)

    # giro el tcp en el eje perpendicular a la dirección al centro
    tcp_axis = PyKDL.Rotation.RotZ(np.pi / 2) * c
    base_axis = H_base_tcp.M * tcp_axis * rot_factor

    #print('perpendicular a tcp', tcp_axis, 'centro respecto a tcp', c)

    # mando la posición a Rapid
    v_rot = PyKDL.Twist(PyKDL.Vector(tcp_base-H_base_tcp.p), base_axis)
    H_base_tcp, q = move_robot(ik_solver_vel, fk_solver_pos, Cls, v_rot, q, delta_t)

