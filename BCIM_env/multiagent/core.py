import numpy as np


# 代理的状态(包括通信和内部/精神状态)
class AgentState(object):
    def __init__(self):
        super(AgentState, self).__init__()
        self.cost = 0.0
        self.revenue = 0.0


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # entity can move / be pushed
        self.movable = False



# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()
        # 种子
        # 起拍价
        self.initial_cost = 0.4
        # 种子是否参与拍卖
        self.is_involve = False
        # 种子在网络中的编号
        self.data_index = 0
        # 种子的成功拍卖价格
        self.success_bid_price = 0.0
        # 表示种子节点的价值
        self.spread_value = 0.0
        # 种子节点对于智能体奖励的贡献度
        self.bratio = 0.0

# 智能体属性
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # 状态
        self.state = AgentState()
    # 竞争者
        # 预算
        self.budget = 0.0
        self.budget_left = 0.0
        # 代理的种子集
        self.seeds = []
        # 是否参与竞价
        self.isBid = False
        # 代理的奖励
        self.reward = 0
        # 针对不同种子的出价价格
        self.bid_price = []
        self.bid_action = 0.0


# 多智能体世界
class World(object):
    def __init__(self):
        # 智能体和实体的列表
        self.agents = []
        self.landmarks = []
        self.platforms = []
        # 仿真时间步长
        self.dt = 0.1

        self.num_agents = 0
        self.num_landmarks = 0
        self.num_platforms = 0

    # 返回世界中的所有实体
    @property
    def entities(self):
        return self.agents + self.landmarks

    # 返回受外部策略控制的所有智能体
    @property
    def policy_agents(self):
        return [agent for agent in self.agents]

    # 返回所有的种子节点
    @property
    def seeds_landmarks(self):
        return [landmark for landmark in self.landmarks]




