import gym.spaces as spaces
class Agent:
    def __init__(self, start_pos, agent_id):
        self.position = start_pos  # 智能体的当前位置
        self.agent_id = agent_id  # 智能体的名称，可用于调试
        self.obs= None
        # 0,1,2,3,4 represent left, right, up, down, -, five moves.
        self.action_space = spaces.Discrete(4)
        # 观察空间由low和high决定
        self.observation_space = None
        # 可以添加更多智能体的属性，如健康值、分数等
