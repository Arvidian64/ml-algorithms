import math
import random

def simulate_pendulum(force, x, x_dot, theta, theta_dot):
    # Räknar ut nästa stadie baserat på kraft och nuvarande stadie
    # Översatt från SimulatePendel.m
    gravity = 9.8
    mass_cart = 1.0
    mass_pole = 0.1
    total_mass = mass_cart + mass_pole
    length = 0.5
    polemass_length = mass_pole * length
    step = 0.02
    four_thirds = 1.3333333333333

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)


    temp = (force + polemass_length * theta_dot * theta_dot * sin_theta) / total_mass
    # Acceleration
    theta_acc = (gravity * sin_theta - cos_theta * temp) / (length * (four_thirds - mass_pole * cos_theta**2 / total_mass))

    x_acc = temp - polemass_length * theta_acc * cos_theta / total_mass

    # Fyra stadie-variabler
    next_x = x + step * x_dot
    next_x_dot = x_dot + step * x_acc
    next_theta = theta + step * theta_dot
    next_theta_dot = theta_dot + step * theta_acc

    return next_x, next_x_dot, next_theta, next_theta_dot

def env_step(state, action_idx):
    # Input state som tuple
    # Output next_state, reward, is_terminal
    twelve_degrees_radians = math.radians(12)
    max_x = 2.4

    x, x_dot, theta, theta_dot = state

    force = 10.0 if action_idx == 1 else -10.0

    next_state = simulate_pendulum(force, x, x_dot, theta, theta_dot)
    next_x, _, next_theta, _ = next_state

    failed = (abs(next_x) > max_x) or (abs(next_theta) > twelve_degrees_radians)

    # Straff o belöning
    if failed:
        reward = -100.0
        is_terminal = True
    else:
        reward = 1.0
        is_terminal = False

    return next_state, reward, is_terminal

def discretize_state(state):
    # Kontinuerliga variabler förvandlas till ett diskret stadie-index (0 till 161)

    x, x_dot, theta, theta_dot = state

    if x < -0.8:
        x_bin = 0
    elif x <= 0.8:
        x_bin = 1
    else:
        x_bin = 2

    if x_dot < -0.5:
        x_dot_bin = 0
    elif x_dot <= 0.5:
        x_dot_bin = 1
    else:
        x_dot_bin = 2

    one_deg = math.radians(1)
    six_deg = math.radians(6)
    if theta < -six_deg:
        theta_bin = 0
    elif theta < -one_deg:
        theta_bin = 1
    elif theta < 0.0:
        theta_bin = 2
    elif theta < one_deg:
        theta_bin = 3
    elif theta < six_deg:
        theta_bin = 4
    else:
        theta_bin = 5

    fifty_deg = math.radians(50)
    if theta_dot < -fifty_deg:
        theta_dot_bin = 0
    elif theta_dot <= fifty_deg:
        theta_dot_bin = 1
    else:
        theta_dot_bin = 2

    # fyra värden kombineras till ett heltal mellan 0 och 161
    state_idx = ((x_bin * 3 + x_dot_bin) * 6 + theta_bin) * 3 + theta_dot_bin

    return state_idx

def select_action(state_idx, q_table, epsilon):

    if random.random() < epsilon:
        return random.randint(0, 1)

    q_left = q_table[state_idx][0]
    q_right = q_table[state_idx][1]

    if q_left > q_right:
        return 0
    elif q_right > q_left:
        return 1
    else:
        return random.randint(0, 1)

def q_training(max_episodes, max_steps=3000,
               alpha=0.1, gamma=0.99,
               epsilon=1.0, epsilon_min=0.008, epsilon_decay=0.992,
               x=0.0,x_dot=0.0, theta=0.0, theta_dot=0.0):

    num_states = 162
    q_table = [[0.0, 0.0] for _ in range(num_states)]

    consecutive_successes = 0

    step = 0

    for episode in range(max_episodes):
        state = [x, x_dot, theta, theta_dot]
        state_idx = discretize_state(state)

        steps_made = 0

        for step in range(max_steps):
            action_idx = select_action(state_idx, q_table, epsilon)

            next_state, reward, is_terminal = env_step(state, action_idx)
            next_state_idx = discretize_state(next_state)

            current_q = q_table[state_idx][action_idx]
            if is_terminal:
                target_q = reward
            else:
                max_future_q = max(q_table[next_state_idx])
                target_q = reward + gamma * max_future_q

            q_table[state_idx][action_idx] = current_q + alpha * (target_q - current_q)

            state = next_state
            state_idx = next_state_idx
            steps_made += 1

            if is_terminal:
                break

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if steps_made == max_steps:
            consecutive_successes += 1
        else:
            consecutive_successes = 0

        # Skriv ut status var 100:e episod
        if (episode + 1) % 100 == 0:
            print(
                f"Episod {episode + 1}: Överlevde {steps_made} steg ({steps_made * 0.02:.1f} s) | Epsilon: {epsilon:.3f}")

        # Avbryt träningen tidigt om agenten har klarat 60 sekunder 10 gånger i rad
        if consecutive_successes >= 10:
            print(f"\nTräning klar vid episod {episode + 1}! Agenten behärskar nu pendeln.")
            break


def main():

    twelve_degrees_radians = math.radians(12)
    max_x = 2.4

    q_training(max_episodes=10000, max_steps=3000,
               alpha=0.1, gamma=0.99, )

if __name__ == "__main__":
    main()
