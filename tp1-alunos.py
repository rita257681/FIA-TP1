import gymnasium as gym
import numpy as np
import pygame

ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
#RENDER_MODE = 'human'
RENDER_MODE = None #seleccione esta opção para não visualizar o ambiente (testes mais rápidos)
EPISODES = 1000

env = gym.make("LunarLander-v3", render_mode =RENDER_MODE, 
    continuous=True, gravity=GRAVITY, 
    enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
    turbulence_power=TURBULENCE_POWER)


def check_successful_landing(observation):
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    legs_touching = contact_left == 1 and contact_right == 1

    on_landing_pad = abs(x) <= 0.2

    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)
    stable = stable_velocity and stable_orientation
 
    if legs_touching and on_landing_pad and stable:
        print("Aterragem bem sucedida!")
        return True

    print("Aterragem falhada!")        
    return False
        
def simulate(steps=1000,seed=None, policy = None): 
    observ, _ = env.reset(seed=seed)        # reset do ambiente para obter uma observacão inicial   
    for step in range(steps):               # ciclo de simulacao
        action = policy(observ)             # chamada a funcao do agente: a partir da observação, produz a acao a executar

        observ, _, term, trunc, _ = env.step(action)    # executa a acao, retorna nova observacao do ambiente e info sobre termina da simulacao

        if term or trunc:
            break

    success = check_successful_landing(observ)  # verifica o sucesso da aterragem; retorna tambem o numero de passos utilizados
    return step, success



#Perceptions
##TODO: Defina as suas perceções aqui

def get_perceptions(observation):
    x, y, vx, vy, theta, v_theta, leg_l, leg_r = observation

    theta_deg = np.rad2deg(theta)

    perceptions = {
        #posição Horizontal 
        "X_left": x < -0.1,
        "X_right": x > 0.1,
        "X_center": -0.1 <= x <= 0.1,
        
        #posição Vertical 
        "Y_high": y > 0.3, # Ajustado para definir "longe" do solo
        "Y_low": y <= 0.3,
        
        #velocidade 
        "Vx_positive": vx > 0.05,
        "Vx_negative": vx < -0.05,
        "Vx_very_positive": vx > 0.15,
        "Vx_very_negative": vx < -0.15,
        "Vy_unstable": vy < -0.1,
        "Vy_stable": vy >= -0.1,
        "Vθ_clockwise": v_theta < -0.05,
        "Vθ_anti_clockwise": v_theta > 0.05,
        
        #orientação 
        "Theta_positive": theta_deg > 8, # Margem de segurança 
        "Theta_negative": theta_deg < -8,
        
        #tocar no chão 
        "contact_left": bool(leg_l),
        "contact_right": bool(leg_r),
        "legs_touching": bool(leg_l and leg_r)
    }
    return perceptions

#Actions
##TODO: Defina as suas ações aqui

def action_rotate_right(): return np.array([0.0, 1.0])  # R_right 
def action_rotate_left():  return np.array([0.0, -1.0]) # R_left 
def action_main_motor():   return np.array([1.0, 0.0])  # Main_Motor 
def action_do_nothing():   return np.array([0.0, 0.0])  # Do_nothing 



def reactive_agent(observation):
    p = get_perceptions(observation)
    action = np.array([0.0, 0.0])

    # 1. Se ambas as pernas tocam, desliga motores 
    if p["legs_touching"]:
        return action
    
    # 2. Estabilização de contacto lateral
    if p["contact_right"] or p["contact_left"]:
        if p["contact_right"] and not p["contact_left"]:
            action += action_rotate_left()
        if p["contact_left"] and not p["contact_right"]:
            action += action_rotate_right()
        return np.clip(action, -1.0, 1.0)   # não deixa descer para Vy_unstable
    
    # 3. Controlo de inclinação crítica 
    if p["Theta_positive"]:
        action += action_rotate_right()
    elif p["Theta_negative"]:
        action += action_rotate_left()
    elif p["Vθ_clockwise"]:      
        action += action_rotate_left()
    elif p["Vθ_anti_clockwise"]: 
        action += action_rotate_right()
    
    # 4. Correção de derrapagem e deriva horizontal
    if p["Vx_very_positive"]:
        action += action_rotate_left()
    elif p["Vx_very_negative"]:
        action += action_rotate_right()
    elif p["X_left"] and p["Vx_negative"]:
        action += action_rotate_right()
    elif p["X_right"] and p["Vx_positive"]:
        action += action_rotate_left()

    # 5. Queda demasiado rápida 
    if p["Vy_unstable"]:
        action += action_main_motor()
        
    # 6. Descida estável no centro
    if p["Y_high"] and p["X_center"] and p["Vy_stable"]:
        pass # action já é [0.0, 0.0]

    action = np.clip(action, -1.0, 1.0)
    return action
    
def keyboard_agent(observation):
    action = [0,0] 
    keys = pygame.key.get_pressed()
    
    print('observação:',observation)

    if keys[pygame.K_UP]:  
        action =+ np.array([1,0])
    if keys[pygame.K_LEFT]:  
        action =+ np.array( [0,-1])
    if keys[pygame.K_RIGHT]: 
        action =+ np.array([0,1])

    return action
    

success = 0.0               # taxa de sucesso
steps = 0.0                 # passos médios por aterragem bem sucedida
for i in range(EPISODES):
    st, su = simulate(steps=1000000, policy=reactive_agent)

    if su:
        steps += st
    success += su
    
    if su>0:
        print('Média de passos das aterragens bem sucedidas:', steps/success*100)
    print('Taxa de sucesso:', success/(i+1)*100)
    
