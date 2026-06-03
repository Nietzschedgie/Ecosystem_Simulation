# ============================================================
# SIMULACIÓN DE UN ECOSISTEMA
# Modelo con 5 especies:
# lobos, ciervos, pasto, conejos y zorros.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PARÁMETROS DEL MODELO
# ============================================================

# Tiempo total de simulación: 10 años
T_total = 3650 #Días

# Paso de tiempo en días
dt = 0.1 #Paso de tiempo en DIAS.

# Número total de pasos temporales
N = int(T_total/dt)

# Vector temporal para representar los resultados
t = np.linspace(0,T_total,N)

# Valor pequeño para evitar divisiones por cero
eps=1e-6

# Número de especies del modelo
n_a = 5 


# ============================================================
# POBLACIONES INICIALES
# ============================================================

Lobos_inicial = 2
Ciervos_inicial = 810
Area_in = 8e5 #m^2
Pasto_in =Area_in
Conejos_inicial = 7690
Zorros_in = 198


# ============================================================
# PARÁMETROS DE CRECIMIENTO
# r_i representa la tasa de reproducción de cada especie.
# ============================================================

r_ciervos = (1/120) / 2 #Una cria cada 4 meses por cada 2 ciervos. [1/día]
r_lobos = (1/180) / 2 #Una cria cada 6 meses por cada 2 lobos. [1/día]
r_pasto = 1/7 # 1 m^2 de pasto crece un 100% cada semana, o un metro cuadrado de pasto se regenera en 7 días. [1/día]
r_conejos = (1/30) / 2 #Una cria cada 30 días por cada 2 conejos. [1/día]
r_zorros = (1/60) /2 #Una cria cada 2 meses por cada dos zorros [1/día].

# Vector de tasas de crecimiento en el orden:
# [lobos, ciervos, pasto, conejos, zorros]
R = np.array([r_lobos,r_ciervos,r_pasto,r_conejos,r_zorros])


# ============================================================
# PARÁMETROS DE DEPREDACIÓN
# D[j,i] = capacidad máxima de consumo del depredador j sobre la presa i
# ============================================================

d_ciervos_lobos = 1 /30 # Cada lobo come 2 ciervos al mes. [ciervo/dia*lobo]
d_conejo_lobos = 1/3  # Cada lobo come 1 conejo cada 3 días. [conejo/dia*lobo]
d_ciervo_pasto = 20 # Cada ciervo come 20 metros de pasto al dia. [m^2/dia*ciervo]
d_conejo_pasto = 1 # Cada conejo come 1 metro de pasto al dia. [m^2/dia*conejo]
d_zorros_lobos = 1/20 # Cada lobo puede comer un zorro cada 20 días.
d_conejo_zorro = 1/5 # Cada zorro se come un conejo cada 5 días.

# Matriz de depredación
D = np.zeros((n_a,n_a))
D[0,1] = d_ciervos_lobos #D[predador, presa]
D[0,3] = d_conejo_lobos
D[1,2] = d_ciervo_pasto
D[3,2] = d_conejo_pasto
D[4,3] = d_conejo_zorro
D[0,4] = d_zorros_lobos

# Unidades: D_ji = [presa_i/dia * depredador_j]



# ============================================================
# PARÁMETROS DE HAMBRE / TOLERANCIA SIN COMER
# T_sinc indica cuántos días puede aguantar una especie sin alimento.
# ============================================================

t_sinc_ciervo = 15
t_sinc_lobo = 30
t_sinc_conejo = 7
t_sinc_zorro = 20
t_sinc_pasto = 1e9


T_sinc = np.array([t_sinc_lobo,t_sinc_ciervo,t_sinc_pasto,t_sinc_conejo,t_sinc_zorro])


# ============================================================
# PARÁMETROS DE MUERTE NATURAL
# rm_i representa la tasa de mortalidad natural de cada especie.
# ============================================================

rm_lobo = 1/(365 *5) #Un lobo vive de media (salvo hambre) 5 años. [1/día]
rm_ciervo = 1/(365*5) #Un ciervo vive de media 5 años. [1/día]
rm_pasto = 1/365 #Un pasto vive de media 1 año. [1/día]
rm_conejo = 1/365 #Un conejo vive de media 1 año. [1/día]
rm_zorro = 1/(365*2) #Un zorro vive de media 2 años. [1/día]

# Vector de mortalidad natural en el orden:

Rm = np.array([rm_lobo,rm_ciervo,rm_pasto,rm_conejo,rm_zorro])


# ============================================================
# PARÁMETROS ESTADÍSTICOS DE CAZA
# P[i,j] controla la facilidad relativa con la que el depredador i caza a la presa j.(Número aproximado de presas necesarias para 50% de éxito).
# ============================================================

p_lobo_ciervo=10 #Un lobo necesita 10 ciervos para tener un 50% de éxito en la caza, si solo cazase ciervos.
P_lobo_conejo = 10 
P_ciervo = 50 
P_conejo = 5 
P_zorro_conejo = 5
P_lobo_zorro = 10

# Matriz de probabilidad / disponibilidad de caza
P = np.zeros((n_a,n_a))
P[0,1] = 1/p_lobo_ciervo #P[predador, presa]
P[0,3] = 1/P_lobo_conejo
P[1,2] = 1/P_ciervo
P[3,2] = 1/P_conejo
P[4,3] = 1/P_zorro_conejo
P[0,4] = 1/P_lobo_zorro

# Unidades de P_ji: [individuos_depredador_j/individuos_presa_i]



# ============================================================
# MATRIZ DE ESTADO DEL SISTEMA
# Cada fila representa una especie y cada columna un instante de tiempo.
# ============================================================

M = np.zeros((n_a,N)) 
M[0,0] = Lobos_inicial
M[1,0] = Ciervos_inicial
M[2,0] = Pasto_in
M[3,0] = Conejos_inicial
M[4,0] = Zorros_in


# ============================================================
# FUNCIONES DE EVOLUCIÓN DEL MODELO
# ============================================================

#   FILA=DEPREDADOR     COLUMNA=PRESA    (En P y D).

def muerte_natural(poblacion, rm, dt):
    return poblacion * rm * dt

def nacimientos(poblacion, r, dt,b):
    return poblacion * r * dt * b


def depredacion_matriz(M, D, dt, P, n_a,b,T_sinc):
    consumo = np.zeros(n_a)
    alpha = np.zeros(n_a)

    for i in range(n_a):
        
        for j in range(n_a):
            S = 0

            for k in range(n_a):

                S += M[k] * P[j,k]
            
            alpha[i] += D[j,i] * (P[j,i]/(S + eps)) * M[j] * b[j] 

        consumo[i] = alpha[i] * M[i]
        
    return consumo * dt





def consumo_matriz(M, dt, P, n_a, T_sinc,m):

    muertes_hambre = np.zeros(n_a)
    b = np.zeros(n_a)

    for i in range(n_a):
        S = 0

        for j in range(n_a):
            S += M[j]/(M[i]+eps) * P[i,j]
        b[i] = (S**m)/(1+S**m)
        

        muertes_hambre[i] = (1/T_sinc[i]) * (1-b[i]) * M[i]

    return muertes_hambre * dt , b


# ============================================================
# FUNCIÓN PARA LA DISMINUCIÓN ESCALONADA DEL ÁREA
# Cada año el área se multiplica por l.
# (Si l=1, el área se mantiene constante. Si l<1, el área disminuye cada año. Si l>1, el área aumenta cada año.)
# ============================================================

def escalon_func(A,dt,l,escalon):
    escalon += dt/365
    if escalon  > 1: #Cada año se reduce el area.
        A = A * l
        escalon = 0
    else:
        A = A
    return A, escalon


# ============================================================
# FUNCIÓN PRINCIPAL DE SIMULACIÓN
# Integra el sistema en el tiempo y devuelve la matriz de poblaciones.
# ============================================================

def simular_ecosistema(M, R, Rm, D, P, T_sinc, dt, Area_in,M0,m,l):
    # l es el factor de diminución del area.
    
    M = np.zeros((n_a,N)) #Matriz de estado del sistema, donde las filas representan cada especie y las columnas el paso del tiempo.
    A = np.zeros(N)
    A[0] = Area_in
    M[:,0] = M0
    escalon =0
    for i in range(1, int(N)):

        #A[i] = A[0] * (l **(i*dt/365)) #Disminucion gradual del area.

        A[i], escalon = escalon_func(A[i-1],dt,l,escalon) #Disminucion escalonada del area.
        
        muertes_hambre, b = consumo_matriz(M[:,i-1], dt, P, n_a, T_sinc, m)
        b[2] = 1 #El pasto no muere de hambre.
        M[:,i] = M[:,i-1] + nacimientos(M[:,i-1],R,dt,b) - muerte_natural(M[:,i-1],Rm,dt) - depredacion_matriz(M[:,i-1],D,dt,P,n_a,b,T_sinc) - muertes_hambre
        for j in range(n_a):
            if  M[j,i] < 2: #Si quedan menos de 2 individuos, se considera que la especie se ha extinguido.: 
                M[j,i] = 0
        if M[2,i] > A[i]: #El pasto no puede superar el area total.
            M[2,i] = A[i]
        M[:,i] = (M[:,i])
    return M


# ============================================================
# EJECUCIÓN DE LA SIMULACIÓN
# ============================================================

m=1
M = (simular_ecosistema(M, R, Rm, D, P, T_sinc, dt, Area_in,M[:,0],m,l=1))
print(np.round(M[:,N-1]).astype(int)) #Población final.





# ============================================================
# BÚSQUEDA DE EQUILIBRIOS POR MÉTODO NUMÉRICO (FSOLVE = MÉTODO HIBRIDO DE POWELL (NEWTON))
# Se usan semillas aleatorias y se filtran soluciones repetidas.
# ============================================================

def encontrar_equilibrios_rango(N_min, N_max, n_starts, tol=1., seed=42,dt=dt,T_sinc=T_sinc,m=m): 
    from scipy.optimize import fsolve #fsolve encuentra raíces de funciones, en este caso puntos donde F(N)=0, es decir, equilibrios.

    N_min = np.asarray(N_min, dtype=int)
    N_max = np.asarray(N_max, dtype=int)
    rng   = np.random.default_rng(seed)

    def F(N):
        N = np.maximum(N, 0)
        muertes_hambre, b = consumo_matriz(N, dt, P, n_a, T_sinc, m)
        N = nacimientos(N, R, dt,b) - muerte_natural(N, Rm, dt) - depredacion_matriz(N, D, dt, P, n_a,b,T_sinc) - muertes_hambre
        return N

    equilibrios = []
    for _ in range(n_starts):
        x0  = rng.uniform(N_min, N_max)           # punto de partida aleatorio en [Nmin, Nmax]
        sol = np.maximum(np.round(fsolve(F, x0)), 0).astype(int)

        # Filtros: residuo pequeño, dentro del rango, y no duplicado
        if np.linalg.norm(F(sol)) > tol:              continue
        if np.any(sol < N_min) or np.any(sol > N_max): continue
        if any(np.linalg.norm(sol - e) < 5 for e in equilibrios): continue

        equilibrios.append(sol)
        print(f"Equilibrio encontrado: {sol}")

    return np.array(equilibrios) if equilibrios else np.empty((0, n_a), dtype=int)






# ============================================================
# FUNCIONES DE REPRESENTACIÓN GRÁFICA
# ============================================================

def plot(t,M):
    plt.figure(figsize=(12, 6))
    plt.plot(t, M[0,:], label='Lobos', color='red')
    plt.plot(t, M[1,:], label='Ciervos', color='brown')
    plt.plot(t, M[2,:], label='Pasto [km^2]', color='green')
    #plt.plot(t, np.log(M[2,:]), label='Pasto [km^2]', color='green')

    plt.plot(t, M[3,:]/1e2, label='Conejos', color='gray')
    plt.plot(t, M[4,:], label='Zorros', color='orange')
    plt.xlabel('Tiempo (días)', fontsize=14)
    plt.ylabel('Número de individuos', fontsize=14)
    plt.title('Evolución de las poblaciones en el ecosistema', fontsize=16)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=12)
    plt.show()

def plot_log(t,M):
    plt.figure(figsize=(12, 6))
    plt.plot(t, np.log10(M[0,:]), label='Lobos', color='red')
    plt.plot(t, np.log10(M[1,:]), label='Ciervos', color='brown')
    plt.plot(t, np.log10(M[2,:]), label='Pasto [m^2]', color='green')
    plt.plot(t, np.log10(M[3,:]), label='Conejos', color='gray')
    plt.plot(t, np.log10(M[4,:]), label='Zorros', color='orange')
    plt.xlabel('Tiempo (días)', fontsize=18)
    plt.ylabel('Número de individuos (log10)', fontsize=20)
    plt.title('Evolución de las poblaciones en el ecosistema', fontsize=22)
    plt.legend(fontsize='xx-large')
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=12)
    plt.show()



# ============================================================
# GRÁFICA FINAL
# ============================================================

plot_log(t,M)


# ============================================================
# BÚSQUEDA DE EQUILIBRIOS (EN EL MÓDELO BASE NO ENCONTRARÁ NINGUNO, PERO SE PUEDE PROBAR CON VARIACIONES DE PARÁMETROS)
# ============================================================
eq=encontrar_equilibrios_rango(N_min=[0,0,0,0,0], N_max=[200,1000,Area_in,10000,500], n_starts=1000,tol=1e-3,seed=42,dt=dt,T_sinc=T_sinc,m=m)
print(eq)

