import numpy as np

from neol.actions import to_env_action


def rollout(env, net, max_steps: int):
    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    net.prepare()

    total_reward = 0.0

    for _ in range(max_steps):
        acts = net.activate(obs)
        action = to_env_action(acts, env.action_space)

        obs, reward, terminated, truncated, _ = env.step(action)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        total_reward += reward
        net.set_reward(reward)

        if terminated or truncated:
            break

    return total_reward
