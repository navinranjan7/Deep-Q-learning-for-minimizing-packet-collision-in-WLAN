from utility.environment_interface import EnvironmentInterface
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np



observation_dict_to_tensor_mapping = {
    str({'type': 'sensing', 'is_sensed': {0: False, 1: False,2:False,3:False}}): np.array([0,0,0,0]),
    str({'type': 'sensing', 'is_sensed': {0: False, 1: False,2:False,3:True}}): np.array([0,0,0,1]),
    str({'type': 'sensing', 'is_sensed': {0: False, 1: False,2:True,3:False}}): np.array([0,0,1,0]),
    str({'type': 'sensing', 'is_sensed': {0: False, 1: False,2:True,3:True}}): np.array([0,0,1,1]),
    str({'type': 'sensing', 'is_sensed': {0: False, 1: True,2:False,3:False}}): np.array([0,1,0,0]),
    str({'type': 'sensing', 'is_sensed': {0: False, 1: True,2:False,3:True}}): np.array([0,1,0,1]),
    str({'type': 'sensing', 'is_sensed': {0: False, 1: True,2:True,3:False}}): np.array([0,1,1,0]),
    str({'type': 'sensing', 'is_sensed': {0: False, 1: True,2:True,3:True}}): np.array([0,1,1,1]),
    str({'type': 'sensing', 'is_sensed': {0: True, 1: False,2:False,3:False}}): np.array([1,0,0,0]),
    str({'type': 'sensing', 'is_sensed': {0: True, 1: False,2:False,3:True}}): np.array([1,0,0,1]),
    str({'type': 'sensing', 'is_sensed': {0: True, 1: False,2:True,3:False}}): np.array([1,0,1,0]),
    str({'type': 'sensing', 'is_sensed': {0: True, 1: False,2:True,3:True}}): np.array([1,0,1,1]),
    str({'type': 'sensing', 'is_sensed': {0: True, 1: True,2:False,3:False}}): np.array([1,1,0,0]),
    str({'type': 'sensing', 'is_sensed': {0: True, 1: True,2:False,3:True}}): np.array([1,1,0,1]),
    str({'type': 'sensing', 'is_sensed': {0: True, 1: True,2:True,3:False}}): np.array([1,1,1,0]),
    str({'type': 'sensing', 'is_sensed': {0: True, 1: True,2:True,3:True}}): np.array([1,1,1,1]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': []}): np.array([2,2,2,2]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [0]}): np.array([3,2,2,2]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [1]}): np.array([2,3,2,2]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [2]}): np.array([2,2,3,2]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [3]}): np.array([2,2,2,3]),
    
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [0,1]}): np.array([3,3,2,2]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [1,0]}): np.array([3,3,2,2]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [0,2]}): np.array([3,2,3,2]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [2,0]}): np.array([3,2,3,2]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [0,3]}): np.array([3,2,2,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [3,0]}): np.array([3,2,2,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [1,2]}): np.array([2,3,3,2]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [2,1]}): np.array([2,3,3,2]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [1,3]}): np.array([2,3,2,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [3,1]}): np.array([2,3,2,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [2,3]}): np.array([2,2,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [3,2]}): np.array([2,2,3,3]),
    
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [0,1,2]}): np.array([3,3,3,2]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [0,2,1]}): np.array([3,3,3,2]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [2,0,1]}): np.array([3,3,3,2]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [2,1,0]}): np.array([3,3,3,2]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [1,2,0]}): np.array([3,3,3,2]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [1,0,2]}): np.array([3,3,3,2]),

    
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [0,1,3]}): np.array([3,3,2,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [0,3,1]}): np.array([3,3,2,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [3,0,1]}): np.array([3,3,2,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [3,1,0]}): np.array([3,3,2,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [1,0,3]}): np.array([3,3,2,3]), 
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [1,3,0]}): np.array([3,3,2,3]),
    
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [0,2,3]}): np.array([3,2,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [0,3,2]}): np.array([3,2,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [2,0,3]}): np.array([3,2,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [2,3,0]}): np.array([3,2,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [3,2,0]}): np.array([3,2,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [3,0,2]}): np.array([3,2,3,3]),
    
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [1,2,3]}): np.array([2,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [1,3,2]}): np.array([2,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [2,1,3]}): np.array([2,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [2,3,1]}): np.array([2,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [3,1,2]}): np.array([2,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [3,2,1]}): np.array([2,3,3,3]),
    
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [0,1,2,3]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [0,1,3,2]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [0,3,1,2]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [0,3,2,1]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [0,2,3,1]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [0,2,1,3]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [1,2,3,0]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [1,2,0,3]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [1,3,2,0]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [1,3,0,2]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [1,0,2,3]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [1,0,3,2]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [2,3,0,1]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [2,3,1,0]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [2,1,3,0]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [2,1,0,3]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [2,0,1,3]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [2,0,3,1]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [3,0,1,2]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [3,0,2,1]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [3,1,0,2]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [3,1,2,0]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [3,2,1,0]}): np.array([3,3,3,3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [3,2,0,1]}): np.array([3,3,3,3])
}



action_index_to_dict_mapping = {
    0: {'type': 'sensing'},
    1: {'type': 'tx_data_packet', 'freq_channel_list': [0], 'num_unit_packet': 1},
    2: {'type': 'tx_data_packet', 'freq_channel_list': [1], 'num_unit_packet': 1},
    3: {'type': 'tx_data_packet', 'freq_channel_list': [2], 'num_unit_packet': 1},
    4: {'type': 'tx_data_packet', 'freq_channel_list': [3], 'num_unit_packet': 1},   
    5: {'type': 'tx_data_packet', 'freq_channel_list': [0], 'num_unit_packet': 2},
    6: {'type': 'tx_data_packet', 'freq_channel_list': [1], 'num_unit_packet': 2},
    7: {'type': 'tx_data_packet', 'freq_channel_list': [2], 'num_unit_packet': 2},
    8: {'type': 'tx_data_packet', 'freq_channel_list': [3], 'num_unit_packet': 2},  
    9: {'type': 'tx_data_packet', 'freq_channel_list': [0], 'num_unit_packet': 3},
    10: {'type': 'tx_data_packet', 'freq_channel_list': [1], 'num_unit_packet': 3},
    11: {'type': 'tx_data_packet', 'freq_channel_list': [2], 'num_unit_packet': 3},
    12: {'type': 'tx_data_packet', 'freq_channel_list': [3], 'num_unit_packet': 3},   
    13: {'type': 'tx_data_packet', 'freq_channel_list': [0], 'num_unit_packet': 4},
    14: {'type': 'tx_data_packet', 'freq_channel_list': [1], 'num_unit_packet': 4},
    15: {'type': 'tx_data_packet', 'freq_channel_list': [2], 'num_unit_packet': 4},
    16: {'type': 'tx_data_packet', 'freq_channel_list': [3], 'num_unit_packet': 4},
    
    17: {'type': 'tx_data_packet', 'freq_channel_list': [0,1], 'num_unit_packet': 1},
    18: {'type': 'tx_data_packet', 'freq_channel_list': [0,2], 'num_unit_packet': 1},
    19: {'type': 'tx_data_packet', 'freq_channel_list': [0,3], 'num_unit_packet': 1},
    20: {'type': 'tx_data_packet', 'freq_channel_list': [0,1], 'num_unit_packet': 2},
    21: {'type': 'tx_data_packet', 'freq_channel_list': [0,2], 'num_unit_packet': 2},
    22: {'type': 'tx_data_packet', 'freq_channel_list': [0,3], 'num_unit_packet': 2},
    23: {'type': 'tx_data_packet', 'freq_channel_list': [0,1], 'num_unit_packet': 3},
    24: {'type': 'tx_data_packet', 'freq_channel_list': [0,2], 'num_unit_packet': 3},
    25: {'type': 'tx_data_packet', 'freq_channel_list': [0,3], 'num_unit_packet': 3},
    26: {'type': 'tx_data_packet', 'freq_channel_list': [0,1], 'num_unit_packet': 4},
    27: {'type': 'tx_data_packet', 'freq_channel_list': [0,2], 'num_unit_packet': 4},
    28: {'type': 'tx_data_packet', 'freq_channel_list': [0,3], 'num_unit_packet': 4},  
    29: {'type': 'tx_data_packet', 'freq_channel_list': [1,2], 'num_unit_packet': 1},
    30: {'type': 'tx_data_packet', 'freq_channel_list': [1,3], 'num_unit_packet': 1},
    31: {'type': 'tx_data_packet', 'freq_channel_list': [1,2], 'num_unit_packet': 2},
    32: {'type': 'tx_data_packet', 'freq_channel_list': [1,3], 'num_unit_packet': 2},
    33: {'type': 'tx_data_packet', 'freq_channel_list': [1,2], 'num_unit_packet': 3},
    34: {'type': 'tx_data_packet', 'freq_channel_list': [1,3], 'num_unit_packet': 3},
    35: {'type': 'tx_data_packet', 'freq_channel_list': [1,2], 'num_unit_packet': 4},
    36: {'type': 'tx_data_packet', 'freq_channel_list': [1,3], 'num_unit_packet': 4},   
    37: {'type': 'tx_data_packet', 'freq_channel_list': [2,3], 'num_unit_packet': 1},
    38: {'type': 'tx_data_packet', 'freq_channel_list': [2,3], 'num_unit_packet': 2},
    39: {'type': 'tx_data_packet', 'freq_channel_list': [2,3], 'num_unit_packet': 3},
    40: {'type': 'tx_data_packet', 'freq_channel_list': [2,3], 'num_unit_packet': 4},
    
    41: {'type': 'tx_data_packet', 'freq_channel_list': [0,1,2], 'num_unit_packet': 1},
    42: {'type': 'tx_data_packet', 'freq_channel_list': [0,1,3], 'num_unit_packet': 1},
    43: {'type': 'tx_data_packet', 'freq_channel_list': [0,1,2], 'num_unit_packet': 2},
    44: {'type': 'tx_data_packet', 'freq_channel_list': [0,1,3], 'num_unit_packet': 2},
    45: {'type': 'tx_data_packet', 'freq_channel_list': [0,1,2], 'num_unit_packet': 3},
    46: {'type': 'tx_data_packet', 'freq_channel_list': [0,1,3], 'num_unit_packet': 3},
    47: {'type': 'tx_data_packet', 'freq_channel_list': [0,1,2], 'num_unit_packet': 4},
    48: {'type': 'tx_data_packet', 'freq_channel_list': [0,1,3], 'num_unit_packet': 4},
    
    49: {'type': 'tx_data_packet', 'freq_channel_list': [0,2,3], 'num_unit_packet': 1},
    50: {'type': 'tx_data_packet', 'freq_channel_list': [0,2,3], 'num_unit_packet': 2},
    51: {'type': 'tx_data_packet', 'freq_channel_list': [0,2,3], 'num_unit_packet': 3},
    52: {'type': 'tx_data_packet', 'freq_channel_list': [0,2,3], 'num_unit_packet': 4},
    
    53: {'type': 'tx_data_packet', 'freq_channel_list': [1,2,3], 'num_unit_packet': 1},
    54: {'type': 'tx_data_packet', 'freq_channel_list': [1,2,3], 'num_unit_packet': 2},
    55: {'type': 'tx_data_packet', 'freq_channel_list': [1,2,3], 'num_unit_packet': 3},
    56: {'type': 'tx_data_packet', 'freq_channel_list': [1,2,3], 'num_unit_packet': 4},
    
    57: {'type': 'tx_data_packet', 'freq_channel_list': [0,1,2,3], 'num_unit_packet': 1},
    58: {'type': 'tx_data_packet', 'freq_channel_list': [0,1,2,3], 'num_unit_packet': 2},
    59: {'type': 'tx_data_packet', 'freq_channel_list': [0,1,2,3], 'num_unit_packet': 3},
    60: {'type': 'tx_data_packet', 'freq_channel_list': [0,1,2,3], 'num_unit_packet': 4},
    
}



class Agent:
    def __init__(self, environment, unit_packet_success_reward, unit_packet_failure_reward, discount_factor, dnn_learning_rate,
                 initial_epsilon, epsilon_decay, min_epsilon,):
        self._env = environment
        self._unit_packet_success_reward = unit_packet_success_reward
        self._unit_packet_failure_reward = unit_packet_failure_reward
        self._discount_factor = discount_factor
        self._dnn_learning_rate = dnn_learning_rate
        self._num_freq_channel = 4
        self._num_action = 61
        self._epsilon = initial_epsilon
        self._epsilon_decay = epsilon_decay
        self._min_epsilon = min_epsilon
        self._replay_memory = deque()
        self._observation = np.zeros(self._num_freq_channel)
        self._past_observation_dict = []
        self._model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(4, activation='relu', kernel_initializer='glorot_normal'))
        model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(1024, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
#        model.add(Dense(61, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(self._num_action,activation='sigmoid', kernel_initializer='he_normal'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self._dnn_learning_rate))
        return model

    def set_init(self):
        initial_action = {'type': 'sensing'}
        observation_dict = self._env.step(initial_action)
        self._observation = observation_dict_to_tensor_mapping[str(observation_dict)]    # initial observation
        self._past_observation_dict = observation_dict

    def train(self, num_episode, run_time, dnn_epochs):
        self._env.disable_video_logging()
        self._env.disable_text_logging()
        for i in range(num_episode):
            self._env.start_simulation(time_us=run_time)
            self.set_init()
            self._replay_memory.clear()   #
            print("< episode %d >" % i)
            while True:
                sim_finish = self.accumulate_replay_memory(self._epsilon)
                if sim_finish:
                    break
            observation, reward = self.replay()

            self._model.fit(observation, reward, epochs=dnn_epochs)
            print(f"(epsilon: {self._epsilon})")
            self._epsilon = max(self._epsilon * self._epsilon_decay, self._min_epsilon)
        self._model.save_weights('my_model')

    def test(self, run_time: int):
        self._env.enable_video_logging()
        self._env.enable_text_logging()
        self._env.start_simulation(time_us=run_time)
        self.set_init()
        self._model.load_weights('my_model')
        while True:
            action, _, _ = self.get_dnn_action_and_value(self._observation)
            action_dict = action_index_to_dict_mapping[int(action)]
            observation_dict = self._env.step(action_dict)
            if observation_dict == 0:
                break
            self._observation = observation_dict_to_tensor_mapping[str(observation_dict)]
            print(f"{self._env.get_score()            }\r", end='', flush=True)

    def get_dnn_action_and_value(self, observation):
        if observation.ndim == 1:
            observation = observation[np.newaxis, ...]
        action_value = self._model.predict(observation)
        best_action = np.argmax(action_value, axis=1)
        best_value = np.amax(action_value, axis=1)
#        print(best_action, best_value, action_value)
        return best_action, best_value, action_value

    def accumulate_replay_memory(self, random_prob):
        if np.random.rand() < random_prob:  # epsilon
            past_observation_dict = self._env.random_action_step()
#            print(f"exploration: {past_observation_dict}")
            if past_observation_dict == {}:
                return True
#            observation = observation_dict_to_tensor_mapping[str(past_observation_dict)]
#            action, _, _ = self.get_dnn_action_and_value(observation)
            print('Exploration')
            action = np.array([np.random.choice([i for i in range (0, self._num_action)])])
        else:
            past_observation_dict = self._past_observation_dict
#            print(f"esfijuaksdf: {past_observation_dict}")
            action, _, _ = self.get_dnn_action_and_value(self._observation)
        action_dict = action_index_to_dict_mapping[int(action)]
        observation_dict = self._env.step(action_dict)
        print(f"observation_dict : {observation_dict}")
        if observation_dict == 0:
            return True
        else:
            reward = self.get_reward(past_observation_dict, action_dict, observation_dict)
            print(reward)
            next_observation = observation_dict_to_tensor_mapping[str(observation_dict)]
#            print(next_observation)
            experience = (self._observation, action, reward, next_observation)
            print(experience)
            self._replay_memory.append(experience)
            self._observation = next_observation
            self._past_observation_dict = observation_dict
#            print(self._observation,self._past_observation_dict )

    def get_reward(self, last_observation, action, next_observation):
#        print(last_observation)
        last_observation_type = last_observation['type']
        current_observation_type = next_observation['type']
        reward = 0
        if last_observation_type == 'sensing':
            available_channels = [key for key, values in last_observation['is_sensed'].items() if values == False]
            count_available_channels = len(available_channels)
            if current_observation_type =='sensing':
                if count_available_channels != 0:
                    reward = (self._num_freq_channel - count_available_channels) *5 - 20 
                elif count_available_channels == 0:
                    reward = 20
            elif current_observation_type == 'tx_data_packet':
                if count_available_channels != 0:
                    action_channel_list = action['freq_channel_list']
                    match_channels_count = len([x for x in available_channels for y in action_channel_list if x == y])
                    unmatch_channels_count = count_available_channels - match_channels_count
#                    collision_packet = len(action_channel_list) - len(next_observation['success_freq_channel_list'])
                    reward1 = ((match_channels_count - unmatch_channels_count)/count_available_channels)*5 
                    num_tx_packet = len(action['freq_channel_list']) * action['num_unit_packet']
                    num_success_packet = len(next_observation['success_freq_channel_list']) * action['num_unit_packet']
                    num_failure_packet = num_tx_packet - num_success_packet
                    reward2 = num_success_packet * self._unit_packet_success_reward + num_failure_packet * self._unit_packet_failure_reward
                    reward = reward1 + reward2
                
                elif count_available_channels == 0:
                    reward = -10
        if last_observation_type == 'tx_data_packet':
            if current_observation_type == 'sensing':
                reward = 0
            elif current_observation_type == 'tx_data_packet':
                num_tx_packet = len(action['freq_channel_list'])  * action['num_unit_packet']
                num_success_packet = len(next_observation['success_freq_channel_list'])  * action['num_unit_packet']
                num_failure_packet = num_tx_packet - num_success_packet
                reward = num_success_packet * self._unit_packet_success_reward + num_failure_packet * self._unit_packet_failure_reward
        return reward

    def replay(self):
        observation = np.stack([x[0] for x in  self._replay_memory], axis=0)
        next_observation = np.stack([x[3] for x in  self._replay_memory], axis=0)
        _, _, action_reward = self.get_dnn_action_and_value(observation)
        _, future_reward, _ = self.get_dnn_action_and_value(next_observation)
        for ind, sample in enumerate(self._replay_memory):
            action = sample[1]
            immediate_reward = sample[2]
            action_reward[ind, action] = immediate_reward + self._discount_factor * future_reward[ind]
        return observation, action_reward

if __name__ == "__main__":
    env = EnvironmentInterface()
    env.connect()
    agent = Agent(environment=env, unit_packet_success_reward=1, unit_packet_failure_reward=-5, discount_factor=0.9,
                  dnn_learning_rate=0.0001, initial_epsilon=1, epsilon_decay=0.989, min_epsilon=0.05)

    agent.train(num_episode=300, run_time=500000, dnn_epochs=10)
    agent.test(10000000)
