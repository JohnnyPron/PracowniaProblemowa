import os
import json
import optuna
import gym
import torch as th
import torch.nn as nn
import panda_gym
from stable_baselines3 import SAC, HerReplayBuffer, DDPG, TD3
from sb3_contrib import TQC


class PandaOptuna:
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
        self.model_type = self.available_models[model.upper()]

    def objective(self, trial: optuna.Trial):
        # create an environment
        env = gym.make(self.env_type, render=False)
        observation = env.reset()
        # choosing the activation function
        act_fun_type = trial.suggest_categorical("activ_func", ["ReLU", "Sigmoid", "LogSigmoid", "GELU", "SELU",
                                                                "ELU", "Tanh"])
        act_fun = self.available_activations[act_fun_type]

        # setting up the network architecture
        dim = trial.suggest_int("network_dims", 2, 3)
        arch = 2 ** trial.suggest_int("exponent_arch", 6, 8)  # basically, it's either 64, 128 or 256
        # setting up the policy
        policy_kwargs = dict(
            activation_fn=act_fun,
            net_arch=[arch for _ in range(dim)]
        )
        # choosing the hyperparameters for the model
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
            learning_rate=trial.suggest_float("learn_rate", 0.001, 0.01, log=True),
            learning_starts=1000,
            gamma=trial.suggest_float("gamma", 0.95, 0.99, log=True),
            policy_kwargs=policy_kwargs
            # ent_coef='auto',
        )
        # timesteps = 10_000
        model.learn(self.timesteps)
        # compute results
        total_value, counter, result = 0, 0, []

        for i in model.ep_success_buffer:
            total_value += i
            counter += 1
            result.append(total_value / counter)
        # free memory
        env.close()
        return result[-1]

    def find_optimal_hyperparams(self, hparams_filename, timesteps=10_000, n_trials=100, timeout=None):
        self.timesteps = timesteps
        # the study
        study = optuna.create_study(direction="maximize")
        try:
            study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        except KeyboardInterrupt:
            pass

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)
        hyperparameters = {"best_value": trial.value}

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
            hyperparameters[key] = value
        # save to .json
        save_path = os.path.join("hyperparameters", hparams_filename)
        if save_path[-5:] != ".json":
            save_path += ".json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(hyperparameters, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    panda_optuna = PandaOptuna(model="TQC")
    panda_optuna.find_optimal_hyperparams("TQC_hyperparams.json", n_trials=50, timeout=10800)
