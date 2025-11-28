import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.stats import chisquare


def mock_qpp_transform(input_data, permutation_idx):
    """
    QPP 순열 변환을 모방하며, 특정 순열(홀수 인덱스)에 인위적인 편향을 주입합니다.
    """
    np.random.seed(permutation_idx)
    shuffle_map = np.arange(256)
    np.random.shuffle(shuffle_map)
    input_int = input_data % 256  
    transformed_int = shuffle_map[input_int]
    if permutation_idx % 2 == 1: 
        output_data = (transformed_int ^ (permutation_idx * 7)) % 64 
    else:
        output_data = (transformed_int ^ (permutation_idx * 7)) % 256
    
    return output_data
def get_randomness_reward(data_array):
    """
    난수열의 Chi-Square 균일성을 측정하여 보상으로 변환합니다.
    Chi-Square 값이 256에 가까울수록 (이상적인 균일성) 높은 보상을 줍니다.
    """
    if len(data_array) < 100:
        return -100 
    observed_frequencies, _ = np.histogram(data_array, bins=256, range=(0, 256))
    expected_frequency = len(data_array) / 256
    chi2_stat, _ = chisquare(observed_frequencies)
    ideal_chi2 = 255
    reward = -abs(chi2_stat - ideal_chi2) / 100.0 

    return reward
class QPP_RL_Env(gym.Env):
    """
    강화 학습을 위한 QPP pQRNG 최적화 환경
    """
    def __init__(self, num_permutations=64, output_data_len=256):
        super(QPP_RL_Env, self).__init__()
        self.num_permutations = num_permutations  
        self.output_data_len = output_data_len    
        self.action_space = spaces.Discrete(num_permutations)
        self.observation_space = spaces.Box(low=0, high=255, shape=(output_data_len,), dtype=np.uint8)
        self.counter = 0       
        self.output_buffer = np.zeros(output_data_len, dtype=np.uint8) 
        self.last_output = 0    

    def step(self, action):
        permutation_idx = action 
        input_data = self.counter & 0xFF
        input_data = (self.counter & 0xFF) ^ self.last_output
        new_random_byte = mock_qpp_transform(input_data, permutation_idx)
        self.output_buffer = np.roll(self.output_buffer, -1) 
        self.output_buffer[-1] = new_random_byte              
        reward = get_randomness_reward(self.output_buffer)
        self.last_output = new_random_byte
        self.counter += 1
        done = False 
        truncated = False

        return self.output_buffer, reward, done, truncated, {} 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.counter = 0
        self.last_output = 0
        self.output_buffer = np.zeros(self.output_data_len, dtype=np.uint8)

        info = {}
        return self.output_buffer, info

    def render(self):
        pass

    def close(self):
        pass
from stable_baselines3 import PPO
import time
NUM_PERMUTATIONS = 64
env = QPP_RL_Env(num_permutations=NUM_PERMUTATIONS)
model = PPO("MlpPolicy", env, 
            learning_rate=0.0001, verbose=0, device="cpu")

print(f"--- AI 기반 QPP 디스패처 학습 시작 (총 50,000 스텝) ---")
start_time = time.time()
model.learn(total_timesteps=200000)
end_time = time.time()
print(f"--- AI 디스패처 학습 완료 (소요 시간: {end_time - start_time:.2f}초) ---")
def test_strategy(model_or_random="random"):
    obs, info = env.reset()
    total_reward = 0
    generated_sequence = []
    for i in range(10240):
        if model_or_random == "random":
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        generated_sequence.append(obs[-1])
    final_reward = get_randomness_reward(obs)

    return final_reward, generated_sequence
ai_final_reward, ai_sequence = test_strategy("ai")
random_final_reward, random_sequence = test_strategy("random")

print("\n=======================================================")
print("             AI 기반 QPP 최적화 시연 결과")
print("=======================================================")

print(f"1. AI 최적화 전략 최종 난수 품질 보상(Reward): {ai_final_reward:.4f}")
print(f"2. 무작위 선택 전략 최종 난수 품질 보상(Reward): {random_final_reward:.4f}")

performance_gain = (abs(random_final_reward) - abs(ai_final_reward)) / abs(random_final_reward) * 100

print(f"\n✨ AI 디스패처를 통한 난수 품질 개선 효과: 약 {performance_gain:.2f}%")
print("(보상 수치는 Chi-Square 통계량과 이상적인 값 255의 차이를 나타냄. 0% 이상이면 AI가 더 균일한 난수열을 생성했음을 의미)")
print("=======================================================")