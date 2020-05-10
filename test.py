import maml_rl.envs
import gym
from gym import wrappers
import torch
import json
import numpy as np
from tqdm import trange
from torch.nn.utils.convert_parameters import parameters_to_vector

from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns


def joint_train(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(config['env-name'], **config['env-kwargs'])
    # env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    policy.share_memory()
    #params_vector = parameters_to_vector(policy.parameters())
    #print (parameters_to_vector(policy.parameters()).shape)
    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                            env_kwargs=config['env-kwargs'],
                            batch_size=config['fast-batch-size'],
                            policy=policy,
                            baseline=baseline,
                            env=env,
                            seed=args.seed,
                            num_workers=args.num_workers)
    logs = dict()
    train_returns, valid_returns = [], []
    for batch in trange(500):
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        train_episodes, valid_episodes = sampler.sample(tasks,
                                                        num_steps=config['num-steps'],
                                                        fast_lr=config['fast-lr'],
                                                        gamma=config['gamma'],
                                                        gae_lambda=config['gae-lambda'],
                                                        device=args.device)
        train_returns.append(get_returns(train_episodes[0]))
        valid_returns.append(get_returns(valid_episodes))
    logs['train_returns'] = np.concatenate(train_returns, axis=0)
    logs['valid_returns'] = np.concatenate(valid_returns, axis=0)

    with open('maml-halfcheetah-vel/joint-train.npz', 'wb') as f:
        np.savez(f, **logs)

    with open('maml-halfcheetah-vel/joint-policy-test.th', 'wb') as f:
        torch.save(policy.state_dict(), f)


def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(config['env-name'], **config['env-kwargs'])
    # env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    with open(args.policy, 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device(args.device))
        policy.load_state_dict(state_dict)
    policy.share_memory()

    joint_policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    with open('maml-halfcheetah-vel/joint-policy.th', 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device(args.device))
        joint_policy.load_state_dict(state_dict)
    joint_policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config['env-kwargs'],
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)

    joint_sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config['env-kwargs'],
                               batch_size=config['fast-batch-size'],
                               policy=joint_policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)

    logs = {'tasks': []}
    train_returns, valid_returns = [], []
    train_speeds, valid_speeds = [], []
    train_rewards, valid_rewards = [], []
    train_tasks, valid_tasks = [], []

    joint_train_returns, joint_valid_returns = [], []
    joint_train_speeds, joint_valid_speeds = [], []
    joint_train_rewards, joint_valid_rewards = [], []
    joint_train_tasks, joint_valid_tasks = [], []

    for batch in trange(args.num_batches):
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        train_episodes, valid_episodes = sampler.sample(tasks,
                                                        num_steps=config['num-steps'],
                                                        fast_lr=config['fast-lr'],
                                                        gamma=config['gamma'],
                                                        gae_lambda=config['gae-lambda'],
                                                        device=args.device)

        joint_train_episodes, joint_valid_episodes = joint_sampler.sample(tasks,
                                                        num_steps=config['num-steps'],
                                                        fast_lr=config['fast-lr'],
                                                        gamma=config['gamma'],
                                                        gae_lambda=config['gae-lambda'],
                                                        device=args.device)
                                                    
        logs['tasks'].extend(tasks)
        #print (len(train_episodes[0][0].forward_speeds[0]))
        train_speeds.append(train_episodes[0][0].forward_speeds)
        valid_speeds.append(valid_episodes[0].forward_speeds)
        train_rewards.append(train_episodes[0][0].rewards_list_return)
        valid_rewards.append(valid_episodes[0].rewards_list_return)
        train_tasks.append(train_episodes[0][0].task_list_return)
        valid_tasks.append(valid_episodes[0].task_list_return)
        train_returns.append(get_returns(train_episodes[0]))
        valid_returns.append(get_returns(valid_episodes))

        #print (len(joint_train_episodes[0][0].forward_speeds[0]))
        joint_train_speeds.append(joint_train_episodes[0][0].forward_speeds)
        joint_valid_speeds.append(joint_valid_episodes[0].forward_speeds)
        joint_train_rewards.append(joint_train_episodes[0][0].rewards_list_return)
        joint_valid_rewards.append(joint_valid_episodes[0].rewards_list_return)
        joint_train_tasks.append(joint_train_episodes[0][0].task_list_return)
        joint_valid_tasks.append(joint_valid_episodes[0].task_list_return)
        joint_train_returns.append(get_returns(joint_train_episodes[0]))
        joint_valid_returns.append(get_returns(joint_valid_episodes))
        

    logs['train_returns'] = np.concatenate(train_returns, axis=0)
    logs['valid_returns'] = np.concatenate(valid_returns, axis=0)
    logs['train_speeds'] = np.concatenate(train_speeds, axis=0)
    logs['valid_speeds'] = np.concatenate(valid_speeds, axis=0)
    logs['train_rewards'] = np.concatenate(train_rewards, axis=0)
    logs['valid_rewards'] = np.concatenate(valid_rewards, axis=0)
    logs['train_tasks'] = np.concatenate(train_tasks, axis=0)
    logs['valid_tasks'] = np.concatenate(valid_tasks, axis=0)

    logs['joint_train_returns'] = np.concatenate(joint_train_returns, axis=0)
    logs['joint_valid_returns'] = np.concatenate(joint_valid_returns, axis=0)
    logs['joint_train_speeds'] = np.concatenate(joint_train_speeds, axis=0)
    logs['joint_valid_speeds'] = np.concatenate(joint_valid_speeds, axis=0)
    logs['joint_train_rewards'] = np.concatenate(joint_train_rewards, axis=0)
    logs['joint_valid_rewards'] = np.concatenate(joint_valid_rewards, axis=0)
    logs['joint_train_tasks'] = np.concatenate(joint_train_tasks, axis=0)
    logs['joint_valid_tasks'] = np.concatenate(joint_valid_tasks, axis=0)

    with open(args.output, 'wb') as f:
        np.savez(f, **logs)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Test')

    parser.add_argument('--config', type=str, required=True,
        help='path to the configuration file')
    parser.add_argument('--policy', type=str, required=True,
        help='path to the policy checkpoint')

    # Evaluation
    evaluation = parser.add_argument_group('Evaluation')
    evaluation.add_argument('--num-batches', type=int, default=10,
        help='number of batches (default: 10)')
    evaluation.add_argument('--meta-batch-size', type=int, default=40,
        help='number of tasks per batch (default: 40)')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output', type=str, required=True,
        help='name of the output folder (default: maml)')
    misc.add_argument('--seed', type=int, default=1,
        help='random seed (default: 1)')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: '
             '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
        help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
        'is not guaranteed. Using CPU is encouraged.')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    # main(args)
    joint_train(args)
