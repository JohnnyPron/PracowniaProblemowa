import os
import json
import gym
import torch as th
import torch.nn as nn
import panda_gym
from stable_baselines3 import SAC, HerReplayBuffer, DDPG, TD3
from sb3_contrib import TQC
import pandas as pd
import time


class PandaModel:
    available_models = {
        "SAC": SAC,
        "DDPG": DDPG,
        "TD3": TD3,
        "TQC": TQC,
    }
    available_activations = {
        "ReLU": nn.ReLU,
        "Sigmoid": nn.Sigmoid,
        "LogSigmoid": nn.LogSigmoid,
        "GELU": nn.GELU,
        "SELU": nn.SELU,
        "ELU": nn.ELU,
        "Tanh": nn.Tanh
    }
    possible_envs = ["PandaReach-v1", "PandaPush-v1", "PandaSlide-v1", "PandaPickAndPlace-v1", "PandaStack-v1"]

    def __init__(self, env="PandaReach-v1", model="SAC"):
        if env not in self.possible_envs:
            raise ValueError("Invalid environment {} datected".format(env))
        self.env_type = env
        self.model_name = model
        self.model_type = self.available_models[model.upper()]

    def train_model_with_hyperparams(self, hparams_filename, model_save, timesteps=30_000):
        # create an environment
        env = gym.make(self.env_type, render=False)
        observation = env.reset()
        # get hyperparameters from chosen json file
        load_path = os.path.join("hyperparameters", hparams_filename)
        if load_path[-5:] != ".json":
            load_path += ".json"
        with open(load_path, "r", encoding="utf-8") as f:
            hparams = json.load(f)
        act_fun_type = hparams["activ_func"]
        act_fun = self.available_activations[act_fun_type]

        # setting up the network architecture
        dim = hparams["network_dims"]
        arch = 2 ** hparams["exponent_arch"]  # it's either 64, 128 or 256
        # setting up the policy
        policy_kwargs = dict(
            activation_fn=act_fun,
            net_arch=[arch for _ in range(dim)]
        )
        # model setup
        model = self.model_type(
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
            learning_rate=hparams["learn_rate"],
            learning_starts=1000,
            gamma=hparams["gamma"],
            policy_kwargs=policy_kwargs
            # ent_coef='auto'
        )
        # model training
        print("=== {} model learning enabled ===".format(self.model_name))
        start = time.time()
        model.learn(timesteps)
        end = time.time()
        print("=== {} model learning finished ===\n {} seconds".format(self.model_name, end - start))
        # getting results (how success rate was changing over time)
        total_value, counter, result = 0, 0, []
        for i in model.ep_success_buffer:
            total_value += i
            counter += 1
            result.append(total_value / counter)
        results_path = os.path.join("results", "learning_results.csv")
        if os.path.exists(results_path):
            df = pd.read_csv(results_path, sep=';')
            df[self.model_name] = result
        else:
            df = pd.DataFrame(data={self.model_name: result})
        df.to_csv(results_path, sep=';', index=False)
        # saving model
        model_path = os.path.join("models", model_save)
        model.save(model_path)
        env.close()

    def test_model(self, model_load, total_episodes=50):
        env = gym.make(self.env_type, render=True)
        model_path = os.path.join("models", model_load)
        model = self.model_type.load(model_path, env=env)

        total_success, reward = 0, 0

        for i_episode in range(1, total_episodes + 1):
            observation = env.reset()
            for t in range(1, 10 + 1):
                env.render()
                action, _states = model.predict(observation)
                observation, reward, done, info = env.step(action)
                if done:
                    break

            if reward != -1.0:
                total_success += 1
            print(f"Episode={i_episode}", "Success_rate={:.2f}".format((100 * total_success) / i_episode), "%")
            reward = 0
        env.close()


if __name__ == "__main__":
    panda_test = PandaModel(model="DDPG")
    panda_test.train_model_with_hyperparams(hparams_filename="DDPG_hyperparams.json", model_save="DDPG_reach")
    #panda_test.test_model("DDPG_reach")
