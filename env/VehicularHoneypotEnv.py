import random
import gym
from gym import spaces
import numpy as np

class VehicularHoneypotEnv(gym.Env):
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
        # observation space
        self.observation_space = spaces.Dict({
            'prev_action': spaces.Discrete(4),  # defender上一次的行动
            'security_risk': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # 车联网环境给的安全危险程度
            'residual_resource': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32)  # 车辆的剩余资源
        })

        # action space
        self.action_space = spaces.Discrete(4)  # 动作包括不开启蜜罐、低交互蜜罐、中等交互蜜罐、高交互蜜罐

        self.security_risk = 0  # 当前车联网安全风险
        self.security_risk_th = 0.5 # 安全风险阈值为 0.5

        # 车辆剩余资源、最大资源限制 和 资源下限
        self.residual_resource = 100    # 车辆剩余资源
        self.resource_upper_bound = 100  # 车辆资源的上限
        self.resource_lower_bound = 10  # 车辆资源的下限
        self.prev_action = None     #上一次的行动
        self.attack_captured = False    # 攻击者是否被捕获
        self.attack_launched = False    # 攻击者是否发起攻击
        '''
        # 定义 攻击者发动攻击的概率、攻击者是否发动攻击、是否部署 honeypot 以及 honeypot 的初始类型。
        self.attack_method = None  # 攻击者发动攻击的类型，初始化为 None
        self.attack_probability = 0  # 攻击者发起攻击的概率，初始化为 0
        self.attacker_launched = False  # 攻击者是否发动攻击，初始化为 False
        self.honeypot_deployed = False  # 蜜罐是否部署，初始化为 False
        '''


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

        # 防御者执行行动
        resource_consumption = action

        self.prev_action = action

        self.residual_resource -= resource_consumption
        print("residual_resource:", self.residual_resource)

        if self.residual_resource < self.resource_lower_bound:
            return {
                'observation': {
                    'prev_action': action,
                    'security_risk': self.security_risk,
                    'residual_resource': self.residual_resource,
                },
                'reward': -50,  # 这个 reward 的设定不太合理
                'done': True,
                'info': {'residual_resource': self.residual_resource}
            }
        """
        这里的问题是：如何进行攻击发生的判断：
        方案1：用攻击者发动攻击的概率 和 当前环境安全风险进行对比来确定是否发生攻击
        方案2：根据当前环境的安全风险来随机生成 是否发动攻击。        

        attack_occurred = np.random.choice([True, False], p=[0.1, 0.9])
        上面表示 选择 True 的概率为 0.1，选择 False 的概率为 0.9
        在论文中，defender 的先验概率越大，那么 atacker 发动攻击的概率就小。
        我们可以根据环境的安全风险程度来确定，当安全风险大的时候，攻击者发动攻击概率就小
        因此，我们可以做一个判断,如果危险程度大于 0.5，那么攻击的概率就小一些，
        比如：security_risk = 0.6,那么，就是[0.4,0.6]        
               
        终止条件:
            1. 车辆的资源消耗完
            2. 当 defender 采取 normal system 时，被攻击。也就是 action == 0 以及 attack_launched  
        """

        # 随机均匀地产生当前的 security_risk 值
        self.security_risk = np.random.uniform(0, 1)
        print("security_risk:", self.security_risk)

        # 确定攻击者是否发动攻击
        self.attack_launched = np.random.choice([True, False], p=[1 - self.security_risk, self.security_risk])
        print("attack_launched:", self.attack_launched)
        if self.attack_launched:
            if action == 0:
                observation = {
                    'prev_action': action,
                    'security_risk': self.security_risk,
                    'residual_resource': self.residual_resource
                }
                info = {"attacker_launched": self.attack_launched}
                done = True
                reward = -50
                return observation, reward, done, info
            else:
                reward = action
                observation = {
                    'prev_action': action,
                    'security_risk': self.security_risk,
                    'residual_resource': self.residual_resource
                }
                info = {"attacker_launched": self.attack_launched}
                done = False
                return observation, reward, done, info

        else:
            if action == 0:
                reward = 3
            else:
                reward = -action
            observation = {
                'prev_action': action,
                'security_risk': self.security_risk,
                'residual_resource': self.residual_resource
            }
            info = {"attacker_launched": self.attack_launched}
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
        self.security_risk = 0
        self.residual_resource = 100
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


# if __name__ == '__main__':
#     env = VehicularHoneypotEnv()
#     obs = env.reset()
#     step = 0
#     for t in range(10):
#         step += 1
#         print("\n")
#         print("-----------step:", step)
#         print("------")
#         action = np.random.randint(4)
#         print("action:",action)
#         obs, reward, done, _ = env.step(action)
#         #if done:
#         #    break
#         print(f"state : {obs}, reward : {reward}")
#         print("是否结束:", done)

