# Lunar Lander: Reactive Agent (Gymnasium)

This repository contains the Practical Assignment for the Artificial Intelligence Fundamentals course, developed at the University of Coimbra during the 2025/2026 academic year. The project focuses on implementing a reactive agent to control spacecraft landing within the `LunarLander-v3` environment from the Gymnasium library.

## Team

* **João Oliveira**
* **Rita Ramos**

## Project Description

The objective of this work is to create a Production System-based agent to control the landing actions of a spacecraft in a simulated physical environment. The script evaluates the agent's performance over 1,000 episodes, calculating the success rate and the average number of steps required for a successful landing, while also supporting testing with simulated wind and turbulence.

## Agent Perceptions

The agent monitors the environment and translates the continuous state into a set of boolean variables (Perceptions). Thresholds adjust according to the presence of wind (`ENABLE_WIND`):

* **Horizontal Position:** Checks whether the craft is to the left (`X_left`, $x < -0.1$) or to the right (`X_right`, $x > 0.1$) of the pad.
* **Vertical Position:** The agent perceives whether it is far from the ground (`Y_high`, $y > 0.5$) or already close to the ground (`Y_low`, $y \le 0.5$).
* **Horizontal Velocity:** Positive when moving to the right (`Vx_positive`) or negative to the left (`Vx_negative`). There is also an alert for critical speeds (`Vx_very_fast`, magnitude $> 0.2$).
* **Vertical Velocity:** Considered unstable/high (`Vy_unstable`) if below a certain value (-0.4 with wind, -0.1 without wind), and stable (`Vy_stable`) otherwise.
* **Angular Velocity:** Clockwise (`Vθ_clockwise`) or counter-clockwise (`Vθ_anti_clockwise`).
* **Orientation:** Monitors left tilt (`Theta_positive`) and right tilt (`Theta_negative`).
* **Ground Contact:** Independent state of the left leg (`contact_left`), the right leg (`contact_right`), and both simultaneously (`legs_touching`).
* **Correcting Trajectory (`correcting`):** A complex perception that is true if the craft is outside the pad but its velocity is already directed towards the center (correcting the trajectory).

## Available Actions

The continuous LunarLander environment accepts a numpy array with 2 values `[-1.0, 1.0]`. Vector-based actions include:

* **R_right:** Rotates the craft to the right by activating the left engine (Vector: `[0.0, 1.0]`).
* **R_left:** Rotates the craft to the left by activating the right engine (Vector: `[0.0, -1.0]`).
* **Main_Motor:** Activates the main thruster to slow down the fall (Vector: `[1.0, 0.0]`).
* **Do_nothing:** Turns off the engines (Vector: `[0.0, 0.0]`).

## Production System (Control Logic)

The reactive agent (`reactive_agent`) dynamically combines actions by processing the following logic blocks in order of priority. Finally, actions are clipped via `np.clip` to respect the environment's boundaries.

1. **Landing Condition:** If `legs_touching` is true, execute `Do_nothing`.
2. **Orientation Control (Highest Priority):**
   * If `Theta_positive` then add `R_right`.
   * Else, if `Theta_negative` then add `R_left`.
3. **Angular Velocity Control:**
   * Else, if `Vθ_clockwise` then add `R_left`.
   * Else, if `Vθ_anti_clockwise` then add `R_right`.
4. **Horizontal Control:** If the craft is not correcting its route (`not correcting`):
   * If `Vx_positive` then add `R_left`.
   * Else, if `Vx_negative` then add `R_right`.
5. **Vertical Control:**
   * If `Vy_unstable` then add `Main_Motor`.
   * Else, if `Vx_very_fast` and `Y_low` add `Main_Motor` (to brake horizontally at low altitude).
6. **Ground Contact Adjustment:** Tries to stabilize the craft when making asymmetric contact:
   * If `contact_right` and not `contact_left` then add `R_left`.
   * Else, if `contact_left` and not `contact_right` then add `R_right`.

## How to Run

* The main script runs `1,000` simulation episodes.
* The render mode can be configured in the `RENDER_MODE` variable (`'human'` to observe the agent in real-time or `None` for fast tests).
* The script also includes a keyboard agent function (`keyboard_agent`), allowing manual control of the spacecraft using keyboard arrow keys.
