action = {f"agent_{i}":i for i in range(8)}
direct = {f"agent_{i}": [0, 1] * 7 for i in range(6)}
print(direct)
# self._direct[agent] = [self.direct[_agent] for _agent in self.agents if _agent != agent]
_direct = {f"agent_{i}": [0, 1] * 7 for i in range(6)}
