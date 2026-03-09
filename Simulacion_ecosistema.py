import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo
T_total = 365 #Días
dt = 1 #Paso de tiempo en DIAS.
N = int(T_total/dt)
t = np.linspace(0,T_total,N)
eps=1e-6#Valor pequeño para evitar división por cero


Lobos_inicial = 20
Ciervos_inicial = 100
Area_in = 40e6 #m^2
Conejos_inicial = 150
Pasto_in = 1e6
#Parametros de crecimiento: (Cuanto crece por día cada poblacion)

r_ciervos = (1/60) / 2 #Una cria al mes por cada 2 ciervos. [1/día]
r_lobos = (1/120) / 2 #Una cria cada 4 meses por cada 2 lobos. [1/día]
r_pasto = 1/7 # 1 m^2 de pasto crece un 100% cada semana, o un metro cuadrado de pasto se regenera en 7 días. [1/día]
r_conejos = (1/7) / 2 #Una cria cada 7 días por cada 2 conejos. [1/día]


# Parametros de depredación:

d_ciervos_lobos = 2 /30 # Cada lobo come 2 ciervos al mes. [ciervo/dia*lobo]
d_conejo_lobos = 1  # Cada lobo come 5 conejos al dia. [conejo/dia*lobo]
d_ciervo_pasto = 10 # Cada ciervo come 10 metros de pasto al dia. [m^2/dia*ciervo]
d_conejo_pasto = 0.5 # Cada conejo come 0.5 metros de pasto al dia. [m^2/dia*conejo]




#Tiempo que aguanta sin comer cada uno (días)
t_sinc_ciervo = 7
t_sinc_lobo = 7
t_sinc_conejo = 1



#Parametros muerte natural:
rm_lobo = 1/365 #Un lobo vive de media (salvo hambre) un año.
rm_ciervo = 1/180 #Un ciervo vive de media medio año.
rm_pasto = 1/240 
rm_conejo = 1/60 #Un conejo vive de media 2 meses.


#Parametros estadísticos (suerte en la caza):
p_lobo_ciervo=5 #Si hay 5 ciervos para cada lobo este cazará siempre.
P_lobo_conejo = 20 #Si hay 20 conejos para cada lobo este cazará siempre.
P_ciervo = 10 #Si hay 100 m2 de pasto para cada ciervo, este comerá siempre.
P_conejo = 1 #Si hay 1 m2 de pasto para cada conejo, este comerá siempre.


Lobos = np.zeros(N)
Ciervos = np.zeros(N)
Pasto = np.zeros(N)
conejos = np.zeros(N)
Lobos[0] = Lobos_inicial
Ciervos[0] = Ciervos_inicial
Pasto[0] = Pasto_in
conejos[0] = Conejos_inicial


#Funciones de evolución: 

def muerte_natural(poblacion, rm, dt):
    return poblacion * rm * dt

def nacimientos(poblacion, r, dt):
    return poblacion * r * dt

def depredacion(predador, d1, dt, presa1, factor_p1,presa2,factor_p2):#Un depredador con 2 presas.
    prob1 = min(1, presa1/(factor_p1*predador + eps) )
    prob2 = min(1, presa2/(factor_p2*predador + eps) )
    df = d1  * prob1/(prob1 + prob2 +eps)

    return predador * df * dt

def depredacion2(predador1, d1,factor_p1,predador2,d2,factor_p2, dt, presa):#Una presa con 2 depredadores. (Es lo mismo que sumar dos veces la funcion anterior)
    prob1 = min(1, presa/(factor_p1*predador1 + eps) )
    prob2 = min(1, presa/(factor_p2*predador2 + eps) )
    df1 = d1 * prob1
    df2 = d2 * prob2 

    return (predador1 * df1  + predador2 * df2 )* dt

def consumo(presa,  dt, predador, factor_p,t_sinc): #Determina cuantos mueren de hambre.

    if presa/predador > factor_p:
        pob_pred = 0
    else:
        prob = 1 - (presa/(factor_p*predador + eps))
        pob_pred = predador * prob * (1/t_sinc) * dt
        
    return pob_pred

def consumo2(presa1,  dt, predador, factor_p1,t_sinc, presa2, factor_p2): #Determina cuantos mueren de hambre si tiene el depredador dos presas distintas.
    prob1 = min(1, presa1/(factor_p1*predador + eps) )
    prob2 = min(1, presa2/(factor_p2*predador + eps) )
    if prob1 + prob2 > 1:
        pob_pred = 0
    else:
        P = 1 - (prob1 + prob2)
        pob_pred = predador * P * (1/t_sinc) * dt 
    return pob_pred


for i in range(1, int(N)):

    # Crecimiento pasto
    if Pasto[i-1] >= Area_in:
        Pasto[i] = Area_in - depredacion2(Ciervos[i-1], d_ciervo_pasto,P_ciervo,conejos[i-1], d_conejo_pasto, P_conejo, dt, Area_in) - rm_pasto * dt * Area_in
    else:
        Pasto[i] = nacimientos(Pasto[i-1], r_pasto, dt) - muerte_natural(Pasto[i-1], rm_pasto, dt) - depredacion2(Ciervos[i-1], d_ciervo_pasto,P_ciervo,conejos[i-1], d_conejo_pasto, P_conejo, dt, Pasto[i-1]) + Pasto[i-1]

    if Pasto[i] >= Area_in:
        Pasto[i] = Area_in

    Ciervos[i] = nacimientos(Ciervos[i-1], r_ciervos, dt) - muerte_natural(Ciervos[i-1], rm_ciervo, dt) - depredacion(Lobos[i-1], d_ciervos_lobos, dt, Ciervos[i-1], p_lobo_ciervo,conejos[i-1],P_lobo_conejo) - consumo(Pasto[i-1],  dt, Ciervos[i-1], P_ciervo,t_sinc_ciervo) + Ciervos[i-1]
    Lobos[i] = nacimientos(Lobos[i-1], r_lobos, dt) - muerte_natural(Lobos[i-1], rm_lobo, dt) - consumo2(Ciervos[i-1], dt, Lobos[i-1], p_lobo_ciervo,t_sinc_lobo,conejos[i-1],P_lobo_conejo) + Lobos[i-1]
    conejos[i] = nacimientos(conejos[i-1], r_conejos, dt) - muerte_natural(conejos[i-1], rm_conejo, dt) - depredacion(Lobos[i-1], d_conejo_lobos, dt, conejos[i-1], P_lobo_conejo,Ciervos[i-1],p_lobo_ciervo) - consumo(Pasto[i-1],  dt, conejos[i-1], P_conejo,t_sinc_conejo) + conejos[i-1]



    if Ciervos[i] < 1 :
        Ciervos[i] = 0

    if Lobos[i] < 1:
        Lobos[i] = 0
        
    if Pasto[i] < 1:
        Pasto[i] = 0
            
    if conejos[i] < 1:
        conejos[i] = 0


# Graficar resultados
Lobos= np.round(Lobos)
Ciervos = np.round(Ciervos)
Pasto = np.round(Pasto)
#print(Lobos)
#print(Ciervos)
#print(Pasto)
plt.figure(figsize=(12, 6))
plt.plot(t, Lobos, label='Lobos', color='red')
plt.plot(t, Ciervos, label='Ciervos', color='brown')
plt.plot(t, conejos, label='Conejos', color='gray')
plt.plot(t, Pasto/1e6, label='Pasto [km^2]', color='green')
plt.xlabel('Tiempo (días)')
plt.ylabel('Número de individuos')
plt.title('Evolución de las poblaciones en el ecosistema')
plt.legend()
plt.grid(True)
plt.show()







    # Crecimiento de los lobos
    #if (Lobos[i-1]/(Ciervos[i-1]+eps)) > m_lobos:
     #   Lobos[i] = (Lobos[i-1]*(1 + r_lobos*dt - rm_lobo * dt) - (((((Lobos[i-1] )) - (m_lobos*Ciervos[i-1]))* (1/t_sinc_lobo)) *dt)) 
      #  if Ciervos[i-1] < 1:
       #     Lobos[i] = (Lobos[i-1]*(1 + r_lobos*dt - rm_lobo * dt) - (((Lobos[i-1]/(1)))* (1/t_sinc_lobo)) *dt)
    #else:
     #   Lobos[i] = (Lobos[i-1]*(1 + r_lobos*dt - rm_lobo * dt)) 