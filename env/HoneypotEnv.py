import gym
from gym import spaces
import numpy as np

class VehicularHoneypotEnv(gym.Env):
    def __init__(self):
        # observation space
        self.observation_space = spaces.Dict({
            'prev_action': spaces.Discrete(4),  # defender上一次的行动
            'security_risk': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # 车联网环境给的安全危险程度
            'residual_resource': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),  # 车辆的剩余资源
            'attacker_status': spaces.Discrete(2)  # 攻击者状态，0表示未被捕获，1表示已被捕获
        })

        # action space
        self.action_space = spaces.Discrete(4)  # 动作包括不开启蜜罐、低交互蜜罐、中等交互蜜罐、高交互蜜罐

        self.security_risk = 0  # 当前车联网安全风险
        self.security_risk_th = 0.5  # 安全风险阈值为0.5

        self.residual_resource = 100  # 车辆剩余资源
        self.resource_upper_bound = 100  # 车辆资源的上限
        self.resource_lower_bound = 10  # 车辆资源的下限

        self.prev_action = None  # 上一次的行动
        self.attacker_status = 0  # 攻击者状态，0表示未被捕获，1表示已被捕获

    def step(self, action):
        # 判断 action 是否合法
        assert self.action_space.contains(action), f"Action {action} is invalid"

        # 记录上一次的行动
        self.prev_action = action

        # 防御者执行行动
        resource_consumption = action  # 资源消耗与行动类型对应
        self.residual_resource -= resource_consumption

        # 检查资源是否低于下限
        if self.residual_resource < self.resource_lower_bound:
            observation = {
                'prev_action': action,
                'security_risk': self.security_risk,
                'residual_resource': self.residual_resource,
                'attacker_status': self.attacker_status
            }
            reward = -50
            done = True
            info = {}
            return observation, reward, done, info

        # 更新安全风险
        self.security_risk = np.random.uniform(0, 1)

        # 判断是否有攻击发生
        self.attacker_status = np.random.choice([0, 1], p=[1 - self.security_risk, self.security_risk])

        # 根据行动和攻击状态给出奖励
        if self.attacker_status == 0:
            reward = 3 - action
        else:
            reward = action

        # 更新观测
        observation = {
            'prev_action': action,
            'security_risk': self.security_risk,
            'residual_resource': self.residual_resource,
            'attacker_status': self.attacker_status
        }

        # 判断是否完成
        done = False
        info = {}

        return observation, reward, done, info

    def reset(self):
        self.security_risk = 0
        self.residual_resource = 100
        self.prev_action = None
        self.attacker_status = 0

        observation = {
            'prev_action': self.prev_action,
            'security_risk': self.security_risk,
            'residual_resource': self.residual_resource,
            'attacker_status': self.attacker_status
        }

        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        pass
