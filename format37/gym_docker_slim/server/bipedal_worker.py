import logging
from datetime import datetime as dt

# enable logging
logging.basicConfig(level=logging.INFO)

logging.info(str(dt.now())+' start')

# logging.error('error')

# Biperdal walker training example
# https://github.com/huggingface/deep-rl-class/blob/main/unit1/unit1.ipynb
from datetime import datetime as dt
# Virtual display
# from pyvirtualdisplay import Display
import gym
# from huggingface_sb3 import load_from_hub, package_to_hub, push_to_hub
# from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.env_util import make_vec_env
import stable_baselines3
import numpy as np


# Create environment
env = gym.make('BipedalWalker-v3')
stable_baselines3.common.utils.get_device()


def get_seed():
    np.random.seed()
    return np.random.randint(0, 2**32)


def train(model, seed, name_postfix):
    #print(dt.now())
    logging.info(str(dt.now())+' learn +')
    #model.learn(total_timesteps=500000, tb_log_name="first_run_"+name_postfix)
    model.learn(total_timesteps=5000, tb_log_name="first_run_"+name_postfix)
    #print(dt.now())
    logging.info(str(dt.now())+' learn -')
    # Save the model
    model_name = "BipedalWalker-v3_"+name_postfix
    model_path = 'data/'
    model.save(model_path+model_name)

    #@title
    eval_env = gym.make("BipedalWalker-v3")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    logging.info(str(dt.now())+' '+f"mean_reward={mean_reward:.2f} +/- {std_reward}")

log_dir = "data/logs/"
seed = get_seed()
model_DDPG_MLP = DDPG(
    policy = 'MlpPolicy',
    env = env,
    tensorboard_log=log_dir,    
    verbose=0,
    seed=seed,
    device='cuda'
)
train(model_DDPG_MLP, seed, 'DDPG-Mlp')
