# Lunar Lander: Agente Reativo (Gymnasium)

Este repositório contém o Trabalho Prático da disciplina de Fundamentos de Inteligência Artificial, desenvolvido na Universidade de Coimbra no ano letivo 2025/2026. O projeto foca-se na implementação de um agente reativo para controlar a aterragem de uma nave no ambiente `LunarLander-v3` da biblioteca Gymnasium.

## Equipa

* **João Oliveira**: Nº de estudante 2023214320 (Turma Prática: PL8, Email: uc2023214320@student.uc.pt).


* **Rita Ramos**: Nº de estudante 2022257681 (Turma Prática: PL8, Email: uc2022257681@student.uc.pt).



## Descrição do Projeto

O objetivo deste trabalho é criar um agente baseado num Sistema de Produção para controlar as ações de aterragem de uma nave num ambiente físico simulado. O script avalia a performance do agente ao longo de 1000 episódios, calculando a taxa de sucesso e a média de passos necessários para uma aterragem bem-sucedida, suportando ainda testes com simulação de vento e turbulência.

## Percepções do Agente

O agente monitoriza o ambiente e traduz o estado contínuo num conjunto de variáveis booleanas (Percepções). Os *thresholds* ajustam-se consoante a presença de vento (`ENABLE_WIND`):

* **Posição Horizontal:** Verifica se a nave está à esquerda (`X_left`, $x < -0.1$) ou à direita (`X_right`, $x > 0.1$) da base.


* **Posição Vertical:** O agente percepciona se está longe do solo (`Y_high`, $y > 0.5$) ou já perto do solo (`Y_low`, $y \le 0.5$).


* **Velocidade Horizontal:** Positiva ao deslocar-se para a direita (`Vx_positive`) ou negativa para a esquerda (`Vx_negative`). Existe também um alerta para velocidades críticas (`Vx_very_fast`, magnitude $> 0.2$).


* **Velocidade Vertical:** Considerada instável/alta (`Vy_unstable`) se inferior a um determinado valor (-0.4 com vento, -0.1 sem vento), e estável (`Vy_stable`) caso contrário.


* **Velocidade Angular:** Negativa/horária (`Vθ_clockwise`) ou positiva/anti-horária (`Vθ_anti_clockwise`).


* **Orientação:** Monitoriza a inclinação para a esquerda (`Theta_positive`) e para a direita (`Theta_negative`).


* **Contacto com o solo:** Estado independente da perna esquerda (`contact_left`), da perna direita (`contact_right`), e de ambas em simultâneo (`legs_touching`).


* **Correção em Curso (`correcting`):** Uma percepção complexa que é verdadeira se a nave estiver fora da base mas a sua velocidade já estiver direcionada para o centro (corrigindo a trajetória).

## Ações Disponíveis

O ambiente contínuo do LunarLander aceita um array numpy com 2 valores `[-1.0, 1.0]`. As ações baseadas em vetores são:

* **R_right:** Roda a nave para a direita ativando o motor esquerdo (Vetor: `[0.0, 1.0]`).


* **R_left:** Roda a nave para a esquerda ativando o motor direito (Vetor: `[0.0, -1.0]`).


* **Main_Motor:** Ativa o propulsor principal para abrandar a queda (Vetor: `[1.0, 0.0]`).


* **Do_nothing:** Desliga os motores (Vetor: `[0.0, 0.0]`).



## Sistema de Produção (Lógica de Controlo)

O agente reativo (`reactive_agent`) combina dinamicamente as ações processando os seguintes blocos lógicos por ordem de prioridade. No final, as ações são limitadas via `np.clip` para respeitar os limites do ambiente.

1. **Condição de Aterragem:** Se `legs_touching` é verdadeiro, executa `Do_nothing`.


2. **Controlo de Orientação (Prioridade Máxima):**
* Se `Theta_positive` então adiciona `R_right`.


* Senão, se `Theta_negative` então adiciona `R_left`.




3. **Controlo de Velocidade Angular:**
* Senão, se `Vθ_clockwise` então adiciona `R_left`.


* Senão, se `Vθ_anti_clockwise` então adiciona `R_right`.




4. **Controlo Horizontal:** Se a nave não estiver a corrigir a rota (`not correcting`):
* Se `Vx_positive` então adiciona `R_left`.


* Senão, se `Vx_negative` então adiciona `R_right`.




5. **Controlo Vertical:**
* Se `Vy_unstable` então adiciona `Main_Motor`.


* Senão, se `Vx_very_fast` e `Y_low` adiciona `Main_Motor` (para travar horizontalmente a baixa altitude).


6. **Ajuste de Contacto com o Solo:** Tenta estabilizar a nave ao tocar de forma assimétrica:
* Se `contact_right` e não `contact_left` então adiciona `R_left`.


* Senão, se `contact_left` e não `contact_right` então adiciona `R_right`.





## Como Executar

* O script principal corre `1000` episódios de simulação.
* O modo de renderização pode ser configurado na variável `RENDER_MODE` (`'human'` para observar o agente em tempo real ou `None` para testes rápidos).
* O script inclui ainda a função de agente por teclado (`keyboard_agent`), que permite assumir o controlo manual da nave utilizando as setas do teclado.
