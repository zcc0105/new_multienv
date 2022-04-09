#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios


if __name__ == '__main__':
    # 解析参数
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # 从脚本中加载场景
    scenario = scenarios.load(args.scenario).Scenario()
    # 创建世界
    world = scenario.make_world()
    # 创建多智能体环境
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
    # 创建查看器窗口
    env.render()
    # 为每个代理创建交互式策略
    policies = [InteractivePolicy(env,i) for i in range(env.n)]
    # 执行
    obs_n = env.reset()
    while True:
        # 从每个代理的策略中查询操作
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # 环境更迭
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # 呈现所有代理视图
        env.render()
        # display rewards
        #for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
