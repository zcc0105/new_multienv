import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None):

        self.world = world
        self.agents = self.world.policy_agents
        self.landmarks = self.world.seeds_landmarks
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # 环境参数
        self.discrete_action_space = False
        # 如果为真，则 action 是一个数字 0...N，否则 action 是一个one-hot的 N 维向量
        self.force_discrete_action = True
        # 如果为真，即使动作是连续的，动作也会离散地执行
        self.time = 0

        # 配置空间
        self.action_space = []
        self.observation_space = []
        self.init_cost = []

        for agent in self.agents:
            total_action_space = []
                # 出价动作空间
            if not self.discrete_action_space:
                agent_action_space = spaces.Box(low=0., high=agent.budget_left, shape=(1,), dtype=np.float32)
                total_action_space.append(agent_action_space)
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
                # 观察空间
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
        for landmark in self.landmarks:
            self.init_cost.append(landmark.initial_cost)

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # 为每个智能体设置动作
        for i, agent in enumerate(self.agents):
            agent.bid_action = self._set_action(action_n[i], agent, self.action_space[i])
        for i, landmark in enumerate(self.landmarks):
                landmark.initial_cost = self.init_cost[i]
        self.bid(self.world)
        self._modify_cost(self.world)
        # 记录每个智能体的obs
        for agent in self.agents:
            obs_n.append(self._get_obs(agent, self.world))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            info_n['n'].append(self._get_info(agent))

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # 重置世界
        self.reset_callback(self.world)
        # 记录每个智能体的obs
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent, self.world))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # 获得制定智能体的obs
    def _get_obs(self, agent, world):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, world)

    # 获得一个制定智能体的dones
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    def _modify_cost(self, world):
        range = 0.8
        for i, landmark in enumerate(world.landmarks):
            a = []
            for j, other in enumerate(world.agents):
                a.append(other.bid_price[i] * landmark.bratio)
            alpha = min(max(max(a), -range), range)
            landmark.initial_cost += (alpha * landmark.initial_cost)
            self.init_cost[i] = float(landmark.initial_cost)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # 为特定代理设置env操作
    def _set_action(self, action, agent, action_space, time=None):
        action_new = np.array(action)
        if not agent.isBid:
            return 0
        # if self.force_discrete_action:
        #     d = np.argmax(action)  # 返回action[0]中最大值的索引值
        #     action[0:] = np.random.uniform(0, agent.budget_left)
        #     action[d] = np.random.uniform(agent.budget_left/2, agent.budget_left)
        index = np.arange(len(action_new))
        rnd_index = np.random.choice(index, 1)
        res = action_new[rnd_index]
        return res

    # 竞价过程
    def bid(self, world):
        for i, landmark in enumerate(world.landmarks):
            start_bid_price = self.seedsInitPrice(world, i)
            landmark.is_involve = True
            landmark.success_bid_price, results_agent_price = self.batchBid(world, landmark, start_bid_price[0])
            for i, agent in enumerate(world.agents):
                agent.bid_price.append(results_agent_price[i])
        self.contribution(world)

    # 具体的一轮竞价过程
    def batchBid(self, world, landmark, start_bid_price):
        all_Agent_Price = []
        # 获取所有智能体的出价
        for i, agent in enumerate(world.agents):
            if agent.budget_left < start_bid_price or agent.bid_action < start_bid_price:
                agent.isBid = False
            if not agent.isBid:
                all_Agent_Price.append(0)
                continue
            all_Agent_Price.append(agent.bid_action)
        # 竞价
        result_Agent_Price = []
        for item in all_Agent_Price:
            result_Agent_Price.append(item)
        max_price = max(all_Agent_Price)
        if max_price == 0: return 0, result_Agent_Price
        if all_Agent_Price.count(max_price) > 1:
            max_index = self.maxValueIndex(all_Agent_Price, max_price)
            agent_reward = {}
            for item in max_index:
                for i, agent in enumerate(world.agents):
                    if item == i:
                        agent.seeds.append(landmark.data_index)
                        agent_reward[item] = self._get_reward(agent)
            # 返回奖励值最大的下标
            max_index_ = max(agent_reward, key=lambda k: agent_reward[k])
            for i, agent in enumerate(world.agents):
                if not agent.platform:
                    if landmark.data_index in agent.seeds:
                        if i != max_index_:
                            agent.seeds.remove(landmark.data_index)
                        elif i == max_index_:
                            # 保存当前智能体的奖励
                            seedSpread = agent.reward
                            agent.reward = self._get_reward(agent)
                            landmark.spread_value = agent.reward - seedSpread
                            all_Agent_Price.remove(max_price)
                            second_Price = max(all_Agent_Price)
                            agent.budget_left -= second_Price
        else:
            max_index = all_Agent_Price.index(max_price)
            all_Agent_Price.remove(max_price)
            second_Price = max(all_Agent_Price)
            for i, agent in enumerate(world.agents):
                if i == max_index:
                    agent.budget_left -= second_Price
                    agent.seeds.append(landmark.data_index)
                    seedsSpread = agent.reward
                    agent.reward = self._get_reward(agent)
                    landmark.spread_value = agent.reward - seedsSpread
        return float(second_Price), result_Agent_Price

    def contribution(self, world):
        for landmark in world.landmarks:
            for agent in world.agents:
                if landmark.data_index in agent.seeds:
                    landmark.bratio = landmark.spread_value * 1.0 / agent.reward

    # 返回种子的起拍价
    def seedsInitPrice(self, world, i):
        return [landmark.initial_cost for landmark in world.landmarks if landmark.name == 'seeds_' + str(i)]

    # 返回最大出价的下标
    def maxValueIndex(self, all_Agent_Price, max_price):
        flag = 0
        lis = []
        for i in range(all_Agent_Price.count(max_price)):
            sec = flag
            flag = all_Agent_Price[flag:].index(max_price)
            lis.append(flag + sec)
            flag = lis[-1:][0] + 1
        return lis

# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n

