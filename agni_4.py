import gymnasium as gym
import numpy as np

class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt=0.02):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )
        self.prev_error = error
        return output

env = gym.make("CartPole-v1", render_mode="human")
pid = PID(kp=50.0, ki=0.0, kd=0.0)
episodes = 5

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done: # this loops keeps on running until the lever falls or time runs out.
        cart_pos, cart_vel, angle, angle_vel = obs
        error = angle
        force = pid.update(error)
        action = 1 if force > 0 else 0
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()
    print(f"Episode {ep+1}: Score = {total_reward}")
env.close()
