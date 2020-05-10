import maml_rl.envs
import gym
from gym import wrappers
import torch
import json
import numpy as np
from tqdm import trange
from copy import deepcopy
import time
import pickle
import matplotlib.pyplot as plt
import scipy.signal as signal

from collections import OrderedDict

from maml_rl.episode import BatchEpisodes
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.samplers.multi_task_sampler import SamplerWorker
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns
from maml_rl.utils.torch_utils import weighted_mean, to_numpy
from maml_rl.utils.reinforcement_learning import reinforce_loss
from maml_rl.utils.reinforcement_learning import get_returns
from torch.nn.utils.convert_parameters import parameters_to_vector


plt.rcParams['font.sans-serif']=['SimSun'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def load_pkl(name):
    with open(name, 'rb') as f:
        # Return dict data
        return pickle.load(f)


def make_env(env_name, env_kwargs={}, seed=None):
    def _make_env():
        env = gym.make(env_name, **env_kwargs)
        if hasattr(env, 'seed'):
            env.seed(seed)
        return env
    return _make_env

def sample_trajectoried(env, policy, params=None):
    observations = env.reset()
    with torch.no_grad():
        while True:
            observations_tensor = torch.from_numpy(observations)
            pi = policy(observations_tensor, params=params)
            actions_tensor = pi.sample()
            actions = actions_tensor.cpu().numpy()
            new_observations, rewards, done, infos = env.step(actions)
            print (infos)
            yield (observations, actions, rewards)
            observations = new_observations
            if done:
                break


def main():
    with open('maml-halfcheetah-vel/config.json', 'r') as f:
        config = json.load(f)

    
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    env = gym.make(config['env-name'], **config['env-kwargs'])
    print (env.observation_space)
    print (env.action_space)

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    with open('maml-halfcheetah-vel/policy.th', 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device('cpu'))
        policy.load_state_dict(state_dict)
    policy.share_memory()

    velocity = np.random.uniform(0.0, 2.0, size=1)
    task = {'velocity':velocity[0]}

    print (velocity)

    env.reset_task(task)
    step = 0
      
    params = None
    for i in range(100):
        obs = env.reset()
        env.reset_task(task)
        ep = BatchEpisodes(
            batch_size=1
        )
        before_update = OrderedDict()
        with torch.no_grad():
            while True:
                # env.render()
                obs_tensor = torch.from_numpy(obs)
                pi = policy(obs_tensor, params=params)
                action_tensor = pi.sample()
                actions = action_tensor.cpu().numpy()

                new_obs, rewards, done, infos = env.step(actions)
                ep.append(obs,actions,rewards)
                before_update[step] = infos      
                obs = new_obs
                step += 1
                if done:
                    break

        baseline.fit(ep)
        ep.compute_advantages(
            baseline,gae_lambda=1.0,normalize=True
        )
        with open('test/task_before_update_{}.pkl'.format(i+1), 'wb') as f:
            pickle.dump(before_update, f, pickle.HIGHEST_PROTOCOL)

        loss = reinforce_loss(policy, ep, params=params)
        params = policy.update_params(
            loss,params=params,step_size=0.1,first_order=True
        )


    val_ep = BatchEpisodes(
        batch_size=1
    ) 
    after_update = OrderedDict()
    
    obs = env.reset()
    env.reset_task(task)
    with torch.no_grad():
        while True:
            time.sleep(0.1)
            # env.render()
            obs_tensor = torch.from_numpy(obs)
            pi = policy(obs_tensor, params=params)
            action_tensor = pi.sample()
            actions = action_tensor.cpu().numpy()

            new_obs, rewards, done, infos = env.step(actions)
            val_ep.append(obs,actions,rewards)
            after_update[step] = infos      
            obs = new_obs
            step += 1
            if done:
                break
    with open('test/task_after_update.pkl', 'wb') as f:
        pickle.dump(after_update, f, pickle.HIGHEST_PROTOCOL)

    # baseline.fit(val_ep)
    # val_ep.compute_advantages(
    #     baseline,gae_lambda=1.0,normalize=True
    # )   

def vis():
    hist_before = load_pkl('test/task_before_update_1.pkl')
    hist_after = load_pkl('test/task_before_update_2.pkl')
    print (hist_before[0].keys())
    speed_before = []
    speed_after = []
    rewards_before = []
    rewards_after = []
    task = []
    for key in hist_after.keys():
        item = hist_after[key]
        print (item)
        speed_after.append(item['forward_speed'])
        task.append(item['task'])
        rewards_after.append((item['reward_forward']+item['reward_ctrl']))
    print (np.sum(rewards_after))
    
    for key in hist_before.keys():
        print (item)
        item = hist_before[key]
        speed_before.append(item['forward_speed'])
        rewards_before.append((item['reward_forward']+item['reward_ctrl']))
    print (np.sum(rewards_before))
    
    fig = plt.figure(dpi=128, figsize=(10,6))
    plt.title('优化前后的策略对于双足机器人的速度的控制曲线')
    plt.plot(speed_after, color='coral', label='经过1次梯度优化后的策略')
    plt.plot(task, color='black', label='目标任务')
    plt.plot(speed_before, color='coral', linestyle='dashed', label='优化前的策略')
    
    plt.plot(rewards_before, color='royalblue', linestyle='dashed',label='优化前智能体每一次执行行为获得的奖励')
    plt.plot(rewards_after, color='royalblue', label='1次梯度优化后智能体每一次执行行为获得的奖励')
    plt.savefig('task.png')
    plt.legend(loc='lower right')
    plt.show()

        


if __name__ == '__main__':
    main()
    vis()
    