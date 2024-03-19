import gym.spaces as spaces
class Agent:
    def __init__(self, start_pos, agent_id):
        self.state = [0,0]  # 智能体的当前位置
        self.agent_id = agent_id  # 智能体的名称，可用于调试
        self.direct = [0,1]
        self._state = []
        self._direct= []


