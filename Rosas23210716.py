"""
Práctica 2: Sistema Cardiovascular

Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México

Nombre del alumno: Aaron Raul Rosas Montoya
Número de control: 23210716
Correo institucional: l23210716@tectijuana.edu.mx

Asignatura: Modelado de Sistemas Fisiológicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""
#Instalar librerias en consola
#!pip install control
#!pip install slycot

# Librerías para cálculo numérico y generación de gráficas
import control as ctrl
import numpy as np
import matplotlib.pyplot as plt # noqa
from scipy import signal
import pandas as pd

u = np.array(pd.read_excel('signal.xlsx',header=None))
x0,t0,tend,dt,w,h = 0,0,15,1e-3,10,5
N = round((tend-t0)/dt) + 1
t = np.linspace(t0,tend,N)
u = np.reshape(signal.resample(u,len(t)),-1)

def cardio(Z,C,R,L):
    num = [L*R,R*Z]
    den = [C*L*R*Z,L*R+L*Z,R*Z]
    sys = ctrl.tf(num,den)
    return sys

#Función de transferencia: Normotenso
Z,C,R,L = 0.033,1.5,0.95,0.01
sysnormo = cardio(Z,C,R,L)
print(f'Función de transferencia del normotenso (control): {sysnormo} \n')

#Función de transferencia: Hipotenso
Z,C,R,L = 0.02,0.25,0.6,0.005
syshipo = cardio(Z,C,R,L)
print(f'Función de transferencia del hipotenso (caso 1): {syshipo} \n')

#Función de transferencia: Hipertenso
Z,C,R,L = 0.05,2.5,1.4,0.02
syshiper = cardio(Z,C,R,L)
print(f'Función de transferencia del hipertenso (caso 2): {syshiper} \n')

#Respuestas en lazo abierto
_,Pp0 = ctrl.forced_response(sysnormo,t,u,x0)
_,Pp1 = ctrl.forced_response(syshipo,t,u,x0)
_,Pp2 = ctrl.forced_response(syshiper,t,u,x0)

fg1 = plt.figure()
plt.plot(t, Pp0, '-', linewidth=1, color=[0.2, 0.7, 0.3], label='Pp(t): Normotenso')
plt.plot(t, Pp1, '-', linewidth=1, color=[0.2, 0.4, 0.9], label='Pp(t): Hipotenso')
plt.plot(t, Pp2, '-', linewidth=1, color=[0.9, 0.2, 0.2], label='Pp(t): Hipertenso')
plt.grid(False)
plt.xlim(0,15); plt.xticks(np.arange(0,16,1))
plt.ylim(-0.6,1.4); plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('t [s]')
plt.ylabel('Pp(t) [V]')
plt.legend(bbox_to_anchor=(0.5,-0.2),loc='center',ncol=3)
plt.show()
fg1.set_size_inches(w,h)
fg1.tight_layout()
fg1.savefig('Cardiovascular Lazo abierto Python.pdf')

#Controlador PID
def controlador(kP,kI,kD,sys):
        Cr = 1e-6
        Re = 1/(kI*Cr)
        Rr = kP*Re
        Ce = kD/Rr
        numPID = [Re*Rr*Ce*Cr,(Re*Ce + Rr*Cr),1]
        denPID = [Re*Cr,0]
        PID = ctrl.tf(numPID,denPID)
        X = ctrl.series(PID,sys)
        sysPID = ctrl.feedback(X,1,sign=-1)
        return sysPID

hipoPID = controlador(0.158,326.322,0.000491,syshipo)
print(f'Función de transferencia del hipotenso en lazo cerrado: {hipoPID}')
hiperPID = controlador(154.0495,35150.1401,0.02208,syshiper)
print(f'Función de transferencia del hipotenso en lazo cerrado: {hiperPID}')

#Respuestas del sistema de control en lazo cerrado
_,PID1 = ctrl.forced_response(hipoPID,t,Pp0,x0)
_,PID2 = ctrl.forced_response(hiperPID,t,Pp0,x0)

#Comparación Normotenso vs Hipotenso y Normotenso vs Hipertenso
mycolors = [[0.45, 0.22, 0.05],[0.72, 0.07, 0.07],[0.08, 0.25, 0.65]]

fg2,axs = plt.subplots(2,1)
fg2.set_size_inches(w,h*2)

axs[0].plot(t, Pp0, '-', linewidth=1, color=mycolors[0], label='$Pp(t):Normotenso$')
axs[0].plot(t, Pp1, '-', linewidth=1, color=mycolors[1], label='$Pp(t):Hipotenso$')
axs[0].plot(t, PID1, ':',linewidth=2, color=mycolors[2], label='$PID(t):Hipotenso$')
axs[0].set_xlim(0,10); axs[0].set_xticks(np.arange(0,11,1))
y0_min = float(min(Pp0.min(), Pp1.min(), PID1.min()))
y0_max = float(max(Pp0.max(), Pp1.max(), PID1.max()))
axs[0].set_ylim(y0_min - 0.1, y0_max + 0.1)
axs[0].set_yticks(np.arange(round(y0_min-0.1,1), round(y0_max+0.2,1), 0.2))
axs[0].set_xlabel('$t$ $[s]$', fontsize=11)
axs[0].set_ylabel('$Pp_i(t)$ $[V]$', fontsize=11)
axs[0].legend(bbox_to_anchor=(0.5,1.12),loc='lower center',ncol=3,frameon=False,fontsize=10)
axs[0].set_title('Normotenso vs Hipotenso', fontsize=10, pad=30)
axs[0].tick_params(labelsize=11)

axs[1].plot(t, Pp0, '-', linewidth=1, color=mycolors[0], label='$Pp(t):Normotenso$')
axs[1].plot(t, Pp2, '-', linewidth=1, color=mycolors[1], label='$Pp(t):Hipertenso$')
axs[1].plot(t, PID2, ':',linewidth=2, color=mycolors[2], label='$PID(t):Hipertenso$')
axs[1].set_xlim(0,10); axs[1].set_xticks(np.arange(0,11,1))
y1_min = float(min(Pp0.min(), Pp2.min(), PID2.min()))
y1_max = float(max(Pp0.max(), Pp2.max(), PID2.max()))
axs[1].set_ylim(y1_min - 0.1, y1_max + 0.1)
axs[1].set_yticks(np.arange(round(y1_min-0.1,1), round(y1_max+0.2,1), 0.2))
axs[1].set_xlabel('$t$ $[s]$', fontsize=11)
axs[1].set_ylabel('$Pp_i(t)$ $[V]$', fontsize=11)
axs[1].legend(bbox_to_anchor=(0.5,1.12),loc='lower center',ncol=3,frameon=False,fontsize=10)
axs[1].set_title('Normotenso vs Hipertenso', fontsize=10, pad=30)
axs[1].tick_params(labelsize=11)

fg2.tight_layout()
fg2.subplots_adjust(hspace=0.6)
plt.show()
fg2.savefig('cardiovascular.pdf')











