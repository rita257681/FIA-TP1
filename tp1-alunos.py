import gymnasium as gym
import numpy as np
import pygame

ENABLE_WIND = True
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

    # Identificar a direção da correção necessária
    # Se x > 0.05 e vx > 0, a nave está a fugir para a direita
    moving_away_right = x > 0.05 and vx > 0
    moving_away_left = x < -0.05 and vx < 0
    
    # Se x > 0.1 e vx < -0.05, a nave já está a recuperar
    correcting_to_center = (x > 0.1 and vx < -0.05) or (x < -0.1 and vx > 0.05)

    if ENABLE_WIND:
        perceptions = {
            "X_left": x < -0.1,
            "X_right": x > 0.1,
            "Y_high": y > 0.5, # Melhor valor encontrado
            "Y_low": y <= 0.5,
            
            # Vento exige reação a velocidades muito baixas para evitar inércia
            "Vx_positive": vx > 0.05, # Melhor valor encontrado
            "Vx_negative": vx < -0.05,
            "Vx_very_fast": abs(vx) > 0.2,
            
            "Vy_unstable": vy < -0.4, # Melhor valor encontrado
            "Vy_stable": vy >= -0.4,
            
            "Vθ_clockwise": v_theta < -0.07, # Melhor valor encontrado
            "Vθ_anti_clockwise": v_theta > 0.07,
            
            "Theta_positive": theta_deg > 5, # Melhor valor encontrado
            "Theta_negative": theta_deg < -5,
            
            "contact_left": bool(leg_l),
            "contact_right": bool(leg_r),
            "legs_touching": bool(leg_l or leg_r),
            "moving_away": moving_away_right or moving_away_left,
            "correcting": correcting_to_center
        }
    else:
        perceptions = {
            "X_left": x < -0.1, # Melhor valor encontrado
            "X_right": x > 0.1,

            "Y_high": y > 0.5, # Melhor valor encontrado
            "Y_low": y <= 0.5,

            "Vx_positive": vx > 0.06, # Melhor valor encontrado
            "Vx_negative": vx < -0.06,
            "Vx_very_fast": abs(vx) > 0.2, # Melhor valor encontrado

            "Vy_unstable": vy < -0.1, # Melhor valor encontrado
            "Vy_stable": vy >= -0.1,

            "Vθ_clockwise": v_theta < -0.07, # Melhor valor encontrado
            "Vθ_anti_clockwise": v_theta > 0.07,

            "Theta_positive": theta_deg > 7, # Melhor valor encontrado
            "Theta_negative": theta_deg < -7,

            "contact_left": bool(leg_l),
            "contact_right": bool(leg_r),
            
            "legs_touching": bool(leg_l and leg_r),
            "moving_away": moving_away_right or moving_away_left,
            "correcting": correcting_to_center
        }

    return perceptions



#Actions
##TODO: Defina as suas ações aqui

def action_rotate_right(): return np.array([0.0, 1.0])  # R_right 
def action_rotate_left():  return np.array([0.0, -1.0]) # R_left 
def action_main_motor():   return np.array([1.0, 0.0])  # Main_Motor 
def action_do_nothing():   return np.array([0.0, 0.0])  # Idle  


def reactive_agent(observation):
    p = get_perceptions(observation)
    action = np.array([0.0, 0.0])

    # Condição de paragem (Aterragem) 
    if p["legs_touching"]:
        return action_do_nothing()

    # Prioridade Máxima: Estabilização de Ângulo e Velocidade Angular 
    # Se a nave estiver a rodar ou inclinada, os motores laterais corrigem primeiro
    if p["Theta_positive"]:
        action += action_rotate_right()
    elif p["Theta_negative"]:
        action += action_rotate_left()
    
    # Se não estiver inclinada mas tiver velocidade angular, estabiliza
    elif p["Vθ_clockwise"]:      
        action += action_rotate_left()
    elif p["Vθ_anti_clockwise"]: 
        action += action_rotate_right()

    # Controlo Horizontal (Combate ao Vento e Deriva)
    # Se estiver a fugir do centro ou com velocidade lateral excessiva
    if not p["correcting"]:
        if p["Vx_positive"]:
            action += action_rotate_left() # Inclina para a esquerda para travar
        elif p["Vx_negative"]:
            action += action_rotate_right() # Inclina para a direita para travar

    # Controlo Vertical (Motor Principal) 
    # Ativa o motor se a queda for instável OU se estivermos a usar a inclinação para travar lateralmente
    if p["Vy_unstable"]:
        action += action_main_motor()
    elif p["Vx_very_fast"] and p["Y_low"]:
        action += action_main_motor()
    # Segurança de Contacto (Garantir verticalidade no toque final) 
    if p["contact_right"] and not p["contact_left"]:
        action += action_rotate_left()
    elif p["contact_left"] and not p["contact_right"]:
        action += action_rotate_right()

    return np.clip(action, -1.0, 1.0)


    
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