"""
VehicularHoneypotEnv 类：用于仿真 vehicular honeypot 的运行环境。

包含的方法及作用：
__init__(self): 初始化函数

reset(self): 状态重置函数

step(self): 动作执行函数，返回值为 state, reward, done, info 四个量

render(self, mode='human'):
    pass
"""
import gym
from gym import spaces
import numpy as np


class VehicularHoneypotEnv():
    # metadata = {'render.modes': ['human']}
    # def __init__(self, max_steps=100, initial_risk=0.5, initial_resource=100):
    def __init__(self, render: bool = False):
        """
        Initializes the Vehicle Honeypot environment.

        Parameters:
        - max_steps (int): The max step
        - initial_risk (float): The initial risk of the IoVs environment.
        - security_risk(float): The current security risk of the IoV environment.

        - initial_resource (int): The initial resource of the autonomous vehicle.
        - residual_resource (int): The residual resource of the autonomous vehicle
        - resource_upper_bound (int): The maximum amount of resources available to the defender.
        - resource_lower_bound (int): The minimum amount of resources that must be maintained for normal vehicle operation.

        - attack_method: The attack method of the attacker when launch an attack (分为三类吧，且每一类的影响各不相同)
        - attacker_launched: 攻击者是否发起了攻击, 初始值为 False, 即不发动攻击

        - honeypot_deployed: 防御者是否部署了 honeypot, 初始值为 False, 即不部署蜜罐
        - honeypot_interact: 防御者部署了那种类型的蜜罐，初始值为 0，即不部署蜜罐

        """
        """
        step 方法是在环境中动态运行的一个时间步长，表示智能体决策前和决策后 其中的一个运行逻辑，输入是一个 action
        也就是说，在这一个 action 中，智能体的奖励、环境状态的变化等
        """

        # 定义环境状态空间，包括上一次防御者的行动和奖励，车联网环境的安全危险程度（security risk）和车辆的剩余资源（residual resource）
        # self.observation_space = spaces.Tuple((
        #     spaces.Discrete(4),  # 防御者上一次的行动
        #     spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # 车联网环境的安全危险程度
        #     spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32)  # 车辆的剩余资源
        # ))

        self.observation_space = spaces.Dict({
            'prev_action': spaces.Discrete(4),  # defender上一次的行动
            'security_risk': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # 车联网环境给的安全危险程度
            'residual_resource': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32)  # 车辆的剩余资源
        })

        # self.observation_space = gym.spaces.Dict({
        #     "security_risk": gym.spaces.Discrete(5),  # 0 = security, 1 = low, 2 = medium, 3 = high, 4 = critical
        #     "residual_resource": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
        #     "attacker_status": gym.spaces.Discrete(2)  # 0 = not captured, 1 = captured
        # })

        # 定义动作空间
        self.action_space = spaces.Discrete(4)  # 动作包括不开启蜜罐、低交互蜜罐、中等交互蜜罐、高交互蜜罐

        self.initial_risk = 0.5
        self.initial_resource = 100

        # 定义初始的安全风险，初始资源，车辆资源的上限和下限

        self.security_risk = self.initial_risk  # 当前车联网安全风险
        self.residual_resource = self.initial_resource  # 初始剩余资源
        self.resource_upper_bound = 100  # 车辆资源的上限
        self.resource_lower_bound = 20  # 车辆资源的下限

        # 定义 攻击者发动攻击的概率、攻击者是否发动攻击、是否部署 honeypot 以及 honeypot 的初始类型。
        self.attack_method = None  # 攻击者发动攻击的类型，初始化为 None
        self.attack_probability = 0  # 攻击者发起攻击的概率，初始化为 0
        self.attacker_launched = False  # 攻击者是否发动攻击，初始化为 False
        self.honeypot_deployed = False  # 蜜罐是否部署，初始化为 False

    def step(self, action):

        """
        执行一个 action 并返回观测、奖励、完成标志和信息

        Args:
            action: int, 防御者的 action，其中 0 表示不开启蜜罐，1 表示开启低交互蜜罐，2 表示开启中等交互蜜罐，3 表示开启高交互蜜罐

        Returns:
            observation: np.ndarray, 当前状态信息，包括车联网环境的安全危险程度和车辆的剩余资源
            reward: float, 防御者在这个时间步的奖励值，根据防御者是否成功捕获攻击者而有所不同
            done: bool, 是否完成任务，本环境中 done 始终为 False
            info: dict, 包含一些附加信息，包括攻击者是否被捕获，消耗的资源，奖励等
        """

        # 判断 action 是否合法
        assert self.action_space.contains(action), f"Action {action} is invalid"
        # 更新当前车辆资源
        self.residual_resource -= 1  # 车辆单位时间内消耗资源为 1
        self.residual_resource -= action  # honeypot deployment costs
        reward = 0

        # 获取上一个action
        prev_action = self.prev_action
        print("上一个action：", prev_action)
        # 资源回收机制：这里考虑的是当上一个 action 结束之后，车辆可以通过资源回收机制来恢复一些资源
        if prev_action == 3:
            self.residual_resource += 2
        elif prev_action == 2:
            self.residual_resource += 1
        elif prev_action == 1:
            self.residual_resource -= 0

        print("residual_resource:", self.residual_resource)

        # 如果当前资源低于下限，为了确保车辆的正常运行，不开启蜜罐
        if self.residual_resource < self.resource_lower_bound:
            return {
                'observation': {
                    'prev_action': prev_action,
                    'security_risk': self.security_risk,
                    'resource': self.residual_resource
                },
                'reward': -100,  # 这个 reward 的设定不太合理
                'done': True,
                'info': {}
            }
        # 资源回收机制结束之后，就更新上一个action
        self.prev_action = action
        print("pre_action:", self.prev_action)

        """
        这里的问题是：如何进行攻击发生的判断：
        方案1：用 攻击者发动攻击的概率 和 当前环境安全风险进行对比来确定是否发生攻击
        方案2：根据当前环境的安全风险来随机生成 是否发动攻击。        

        attack_occurred = np.random.choice([True, False], p=[0.1, 0.9])
        上面表示 选择 True 的概率为 0.1，选择 False 的概率为 0.9
        在论文中，defender 的先验概率越大，那么 atacker 发动攻击的概率就小。
        我们可以根据环境的安全风险程度来确定，当安全风险大的时候，攻击者发动攻击概率就小
        因此，我们可以做一个判断,如果危险程度大于 0.5，那么攻击的概率就小一些，
        比如：security_risk = 0.6,那么，就是[0.4,0.6]        

        考虑 attack method, 是因为 
            1. 部署的 honeypot 类型的能力不同，如果部署过高，那么就大材小用，如果部署过低，那么可能就被敌手所利用。
            2. 没有部署 honeypot 时，对于normal system 的影响也是不一样的。
        """

        self.security_risk = np.random.uniform(0, 1)
        print("security_risk:", self.security_risk)
        # 确定攻击者是否发动攻击
        self.attacker_launched = np.random.choice([True, False], p=[1 - self.security_risk, self.security_risk])
        print("attacker_launched:", self.attacker_launched)
        # 如果发动攻击了，那么久选择攻击方式，并根据 defender 采取的 action 以及 atacker 的 method 来更新 reward
        if self.attacker_launched:
            self.attack_method = np.random.choice(['low', 'medium', 'high'])  # 随机选择攻击方式
            print("attack_method:", self.attack_method)
            # 当 defender 部署 高交互蜜罐时
            if action == 3:
                self.honeypot_deployed = True
                if self.attack_method == 'low':
                    reward = 1
                elif self.attack_method == 'medium':
                    reward = 2
                elif self.attack_method == 'high':
                    reward = 3
            # 当defender 部署 中等交互蜜罐时
            elif action == 2:
                self.honeypot_deployed = True
                if self.attack_method == 'low':
                    reward = 1
                elif self.attack_method == 'medium':
                    reward = 2
                elif self.attack_method == 'high':
                    reward = 1
            # 当defender 部署 低交互蜜罐时
            elif action == 1:
                self.honeypot_deployed = True
                if self.attack_method == 'low':
                    reward = 2
                elif self.attack_method == 'medium':
                    reward = 1
                elif self.attack_method == 'high':
                    reward = 1
            elif action == 0:
                self.honeypot_deployed = False
                if self.attack_method == 'low':
                    reward = -1
                elif self.attack_method == 'medium':
                    reward = -2
                elif self.attack_method == 'high':
                    reward = -3
            print("发生攻击时，defender 的奖励为：", reward)
            print("honeypot_deployed:", self.honeypot_deployed)

            observation = {
                'prev_action': prev_action,
                'security_risk': self.security_risk,
                'resource': self.residual_resource
            }
            info = {"honeypot_deployed": self.honeypot_deployed, "attacker_launched": self.attacker_launched}
            done = False
            return observation, reward, done, info

        # 当 攻击者没有发生攻击，defender 采取了 action 所产生的影响
        else:
            if action == 3:
                self.honeypot_deployed = True
                reward = -3
            elif action == 2:
                self.honeypot_deployed = True
                reward = -2
            elif action == 1:
                self.honeypot_deployed = True
                reward = -1
            elif action == 0:
                reward = 1
            print("没有发生攻击时，defender 的奖励为：", reward)
            print("honeypot_deployed:", self.honeypot_deployed)
            # 返回相应的值
            observation = {
                'prev_action': prev_action,
                'security_risk': self.security_risk,
                'resource': self.residual_resource
            }
            info = {"honeypot_deployed": self.honeypot_deployed, "attacker_launched": self.attacker_launched}
            done = False
            return observation, reward, done, info

    def reset(self):
        """
        reset the vehicular honeypot environment.
        :param:
            security_risk
            residual_resource
            attacker_launched
            honeypot_deployed
            honeypot_interact
        :return: security_risk and residual_resource
        """
        self.security_risk = self.initial_risk
        self.residual_resource = self.initial_resource
        self.attacker_launched = False
        self.honeypot_deployed = False
        # 重置上一个 action
        self.prev_action = None
        observation = {
            'prev_action': self.prev_action,
            'security_risk': self.security_risk,
            'resource': self.residual_resource
        }
        return observation

    def render(self):
        pass

    def close(self):
        pass

    def choose_action(self):
        pass

# if __name__ == '__main__':
#     env = VehicularHoneypotEnv()
#     obs = env.reset()
#     step = 0
#     while True:
#         step += 1
#         print("-----------step:", step)
#         print("------")
#         action = np.random.randint(4)
#         print("action:",action)
#         obs, reward, done, _ = env.step(action)
#         if done:
#             break
#         print(f"state : {obs}, reward : {reward}")
#
