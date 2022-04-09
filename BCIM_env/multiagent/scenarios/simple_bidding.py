import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from reward import reward
from utils import parse_graph_txt_file


class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # 设置世界属性
        # world.dim_c = 2
        num_agents = 2  # 竞争者的数目有2个
        world.num_agents = num_agents
        num_landmarks = 5  # 候选种子的数目
        world.num_landmarks = num_landmarks

        candicateSeeds = [8, 47, 86, 170, 239]
        # 添加智能体
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent_%d' % i
            agent.movable = False  # 仅仅有出价行为，设置为不可移动
            agent.silent = True
        # 添加地标  即候选种子集
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'seeds_%d' % i
            landmark.movable = False
        self.reset_world(world)
        return world

    # 返回奖励值
    def reward(self, agent, world):
        # from utils import parse_graph_txt_file
        # import utils
        G = parse_graph_txt_file('D:\BCIM\SNInfluenceMaximization-master\Dataset\BA.txt')
        seeds = agent.seeds
        compSeeds = []
        candSeeds = []
        for landmark in world.landmarks:
            if not landmark.is_involve:
                candSeeds.append(landmark.data_index)
        for other_agent in enumerate(world.agents):
            if other_agent != agent:
                compSeeds.append(landmark.data_index)
        return reward(G, seeds, compSeeds, candSeeds)

    def reset_world(self, world):
        candicateSeeds = [8, 47, 86, 170, 239]
        for i, agent in enumerate(world.agents):
            agent.budget = 3  # 设置智能体初始预算
            agent.budget_left = agent.budget
            agent.seeds = []
            agent.bid_action = 1.0
            agent.isBid = True
            agent.reward = 0
            agent.bid_price = []
        for i, landmark in enumerate(world.landmarks):
            landmark.initial_cost = 0.4
            landmark.data_index = candicateSeeds[i]
            landmark.success_bid_price = 0.0
            landmark.is_involve = False
            landmark.spread_value = 0.0
            landmark.bratio = 0.0

    def observation(self, agent, world):
        # 对于每个竞争者来说，只能观察到每个种子的起拍价以及成交价
        seeds_info = []
        agent_info = []
        for i, landmark in enumerate(world.landmarks):
            seeds_info.append(float(landmark.initial_cost))
            seeds_info.append(landmark.success_bid_price)
        agent_info.append(agent.budget-agent.budget_left)
        agent_info.append(agent.reward)
        return np.concatenate([seeds_info]+[agent_info])

    def benchmark_data(self, agent, world):
        agent.state.cost = agent.budget-agent.budget_left
        agent.state.revenue = agent.reward
        return agent.state

if __name__ == '__main__':
    sc = Scenario()
    world = sc.make_world()
    sc.reset_world(world)
    for agent in world.agents:
        print("%s " % str(agent.name) + str(agent.reward) + str(agent.seeds))
    for landmark in world.landmarks:
        print("%s " % str(landmark.name) + str(landmark.initial_cost) +" "+ str(landmark.bratio))
