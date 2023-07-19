import random
import gym
from gym import spaces
import numpy as np

class VehicularHoneypotEnv(gym.Env):
    def __init__(self, render: bool = False, noise_std=0.1):
        self.observation_space = spaces.Tuple((
            spaces.Discrete(4),  # defender上一次的行动
            spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # 车联网环境给的安全危险程度
            spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32)  # 车辆的剩余资源
        ))
        # action space
        self.action_space = spaces.Discrete(4)  # 动作包括不开启蜜罐、低交互蜜罐、中等交互蜜罐、高交互蜜罐

        self.security_risk = 0  # 当前车联网安全风险
        self.security_risk_th = 0.5  # 安全风险阈值为 0.5

        # 车辆剩余资源、最大资源限制 和 资源下限
        self.residual_resource = 100  # 车辆剩余资源
        self.resource_lower_bound = 10  # 车辆资源的下限
        self.prev_action = 0  # 上一次的行动
        self.attack_captured = False  # 攻击者是否被捕获
        self.attack_launched = False  # 攻击者是否发起攻击

        self.noise_std = noise_std

    def step(self, action):

        # 判断 action 是否合法
        assert self.action_space.contains(action), f"Action {action} is invalid"
        print("action:", action)

        # 防御者执行行动
        resource_consumption = action

        self.prev_action = action

        self.residual_resource = self.residual_resource - resource_consumption
        self.residual_resource -= self.residual_resource - 1
        print("residual_resource:", self.residual_resource)

        if self.residual_resource < self.resource_lower_bound:
            observation = (action, self.security_risk, self.residual_resource)
            reward = -10
            done = True
            info = {'residual_resource': self.residual_resource}
            return observation, reward, done, info

        # 随机均匀地产生当前的 security_risk 值
        #self.security_risk = np.random.normal(0.5, 0.1)
        self.security_risk = np.random.uniform(0.1, 0.9)
        print("security_risk:", self.security_risk)

        # 确定攻击者是否发动攻击
        self.attack_launched = np.random.choice([True, False], p=[1 - self.security_risk, self.security_risk])
        print("attack_launched:", self.attack_launched)
        if self.attack_launched:
            if action == 0:
                reward = -10
            else:
                reward = action + self.security_risk * 10
            observation = (action, self.security_risk, self.residual_resource)
            info = {"attacker_launched": self.attack_launched}
            done = False
            return observation, reward, done, info
                # observation = (action, self.security_risk, self.residual_resource)
                # info = {"attacker_launched": self.attack_launched}
                # done = True
                # reward = -10
                # return observation, reward, done, info
            # else:
            #     reward = action + self.security_risk * 10  # 增加基于安全风险的奖励机制
            #     observation = (action, self.security_risk, self.residual_resource)
            #     info = {"attacker_launched": self.attack_launched}
            #     done = False
            #     return observation, reward, done, info
        else:
            if action == 0:
                reward = 3
            else:
                reward = -1 * (action + self.security_risk * 2)  # 基于安全风险和资源消耗的奖励
            observation = (action, self.security_risk, self.residual_resource)
            info = {"attacker_launched": self.attack_launched}
            done = False
            return observation, reward, done, info
    def reset(self):
        self.security_risk = 0
        self.residual_resource = 100
        # 重置上一个 action
        self.prev_action = 0
        observation = (self.prev_action, self.security_risk, self.residual_resource)
        return observation

    def render(self):
        pass

    def close(self):
        pass

