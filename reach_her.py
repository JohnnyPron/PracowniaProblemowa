import time

import gym
import torch as th
from stable_baselines3.her import HerReplayBuffer

import panda_gym
from stable_baselines3 import SAC, HerReplayBuffer, DDPG, TD3
import pandas

# DEFINE PARAMETERS
model_type = SAC
learning_rate = 0.002
gamma = 0.95
arch_values = "[128, 64, 32]"
env = gym.make("PandaReach-v1", render=True)

policy_kwargs = dict(
    activation_fn=th.nn.ReLU,
    net_arch=[128, 64, 32]
)

observation = env.reset()
done = False

# CREATE MODEL
model = model_type(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        max_episode_length=100,
        n_sampled_goal=4,
        goal_selection_strategy='future',
    ),
    verbose=1,
    buffer_size=100000,
    batch_size=256,
    learning_rate=learning_rate,
    learning_starts=1000,
    gamma=gamma,
    policy_kwargs=policy_kwargs
)


# LEARNING
print("Starting learning")
timesteps = 10_000
start = time.time()
model.learn(timesteps)
end = time.time()
print("=== LEARN === {}".format(end - start))

# SAVE RESULTS
total_value, counter, result = 0.0, 0, []
for i in model.ep_success_buffer:
    total_value += i
    counter += 1
    result.append(total_value / counter)

# TRANSFER RESULTS TO CSV
df = pandas.DataFrame(data={arch_values: result})
df.to_csv("result2.csv", sep=';', index=False)

# SAVE MODEL
model.save('reach_her_model')

# LOAD SAVED MODEL
model = model_type.load('reach_her_model', env=env)

# MAKE PREDICTIONS ON LEARNED MODEL
total_success, total_episodes, reward = 0, 50, 0
for i_episode in range(1, total_episodes + 1):
    observation = env.reset()
    for t in range(1, 10 + 1):
        env.render()
        # action = env.action_space.sample()
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(action)
        if done:
            break

    if reward != -1.0:
        total_success += 1
    print(f"Episode={i_episode}", "Success_rate={:.2f}".format((100 * total_success) / i_episode), "%")
    reward = 0

env.close()
