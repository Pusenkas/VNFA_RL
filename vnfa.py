from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
import random
import data

class VNFA(gym.Env):
    # FUNCTION_AMOUNT = 6
    # APP_AMOUNT = 2
    # COMP_AMOUNT = 3
    # APP_COMP_DELAY = np.array([[1, 10, 100], [100, 10, 1]], dtype=np.float32)
    # COMP_COMP_DELAY = np.array([[0, 1, 10], [1, 0, 1], [10, 1, 0]], dtype=np.float32)
    # SFC = np.array([3, 3], dtype=int)
    # AVAILABLE_RESOURCE = np.array([1, 1, 1], dtype=np.float32)
    # REQUIRED_RESOURCE = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    # SFC_SIZE = 3
    EPS = 0.0001

    def __init__(self, app_amount, comp_amount, app_comp_delay, comp_comp_delay, sfc, available_resources, required_resources, use_default=False, seed=42):
        self.APP_AMOUNT = app_amount
        self.COMP_AMOUNT = comp_amount
        self.APP_COMP_DELAY = app_comp_delay
        self.COMP_COMP_DELAY = comp_comp_delay
        self.SFC = sfc
        self.AVAILABLE_RESOURCE = available_resources
        self.REQUIRED_RESOURCE = required_resources
        self.SFC_SIZE = np.max(sfc) # TODO

        random.seed(seed)
        self.observation_space = gym.spaces.Dict({
            "available_resources": Box(0, 1, shape=(self.COMP_AMOUNT,)),
            "delays": Box(0, np.inf, shape=(self.COMP_AMOUNT,)),
            "resources_queue": Box(0, 1, shape=(self.SFC_SIZE,)),
            "F": Box(0, np.inf, shape=(1,)),
            "sfc_delay": Box(0, np.inf, shape=(1,)),
            })
        self.action_space = gym.spaces.Discrete(self.COMP_AMOUNT)
        
        self.reset()


    def _fill_queue(self, pos):
        queue = np.zeros(self.SFC_SIZE, dtype=np.float32)
        self._current_sfc_length = self._current_sfc[self._sfc_number]
        for i in range(self._current_sfc[self._sfc_number]):
            queue[i] = self._required_resources[pos + i]
        self._sfc_number += 1
        return queue

    def _move_queue(self):
        self._state_resources_queue[0] = 0
        self._state_resources_queue = np.roll(self._state_resources_queue, shift=-1)

    def _calculate_delay_reward(chosen, best, second_best):
        if best == 0.0:
            if chosen == 0.0:
                return 1.0
            else:
                return float(second_best/ chosen) * 0.5
        else:
            return float(best / chosen)

    # def _create_queue_app_array(self):
    #     result = np.zeros(self._state_FUNCTION_AMOUNT, dtype=int)
    #     i = 0
    #     for app_number, sfc_len in enumerate(self.SFC, 1):
    #         for j in range(sfc_len):
    #             result[i] = app_number
    #             i += 1
    #     return result


    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {"available_resources": self._state_resources, "delays": self._state_delays, "resources_queue": self._state_resources_queue, "F":self._state_F, "sfc_delay": self._state_SFC_delay}


    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self._current_function = 0
        self._sfc_number = 0
        self._current_sfc_length = 0

        function_amount_sampled, resources_sampled, sfc_sampled = data.sample_sfc(self.SFC, self.REQUIRED_RESOURCE, self.APP_AMOUNT, seed=random.randint(1, 1000))
        self._current_sfc = sfc_sampled
        self._required_resources = resources_sampled
        self._function_amount = function_amount_sampled


        self._state_resources = self.AVAILABLE_RESOURCE.copy()
        self._state_delays = self.APP_COMP_DELAY[0]
        self._state_resources_queue = self._fill_queue(self._current_function)
        self._state_F = np.array([0.0], dtype=np.float32)
        self._state_SFC_delay = np.array([0.0], dtype=np.float32)

        observation = self._get_obs()
        info = self._get_info()

        
        return observation, info
    
    def custom_reset(self, function_amount, required_resources, sfc):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=None)

        # Randomly place the agent anywhere on the grid
        self._current_function = 0
        self._sfc_number = 0

        self._current_sfc = sfc
        self._required_resources = required_resources
        self._function_amount = function_amount

        self._state_resources = self.AVAILABLE_RESOURCE.copy()
        self._state_delays = self.APP_COMP_DELAY[0]
        self._state_resources_queue = self._fill_queue(self._current_function)
        self._state_F = np.array([0.0], dtype=np.float32)
        self._state_SFC_delay = np.array([0.0], dtype=np.float32)

        observation = self._get_obs()
        info = self._get_info()

        
        return observation, info

    def action_masks(self):
        needed_resource = self._required_resources[self._current_function]
        mask_array = np.ones(self.COMP_AMOUNT, dtype=bool)
        for i, left_resource in enumerate(self._state_resources):
            if needed_resource > left_resource:
                mask_array[i] = False
        return mask_array

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-2 for actions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        truncated = False
        terminated = False
        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        resource_overload = False
        old_value = self._state_F[0]
        if self._current_function + 1 == self._function_amount:
            self._state_resources[action] -= self._state_resources_queue[0]
            if self._state_resources[action] < -VNFA.EPS:
                resource_overload = True
            self._state_SFC_delay[0] += self._state_delays[action]
            # chosen_delay_frac = VNFA._calculate_delay_reward(self._state_delays[action], np.min(self._state_delays), np.partition(self._state_delays, 1)[1])
            self._state_F[0] = max(self._state_F[0], self._state_SFC_delay[0])
            self._state_SFC_delay[0] = 0
            self._state_delays = np.zeros(self.COMP_AMOUNT, dtype=np.float32)
            self._state_resources_queue.fill(0)
            terminated = True
            self.success = True
        elif self._current_sfc_length == 1:
            self._state_resources[action] -= self._state_resources_queue[0]
            if self._state_resources[action] < -VNFA.EPS:
                resource_overload = True
            self._state_SFC_delay[0] += self._state_delays[action]
            self._state_F[0] = max(self._state_F[0], self._state_SFC_delay[0])
            self._state_SFC_delay[0] = 0
            self._state_delays = self.APP_COMP_DELAY[self._sfc_number]
            self._state_resources_queue = self._fill_queue(self._current_function + 1)
        else:
            self._current_sfc_length -= 1
            self._state_resources[action] -= self._state_resources_queue[0]
            if self._state_resources[action] < -VNFA.EPS:
                resource_overload = True
            self._state_SFC_delay[0] += self._state_delays[action]
            self._state_F[0] = max(self._state_F[0], self._state_SFC_delay[0])
            self._move_queue()
            self._state_delays = self.COMP_COMP_DELAY[action]

        if resource_overload:
            print("ERROR")
            observation = self._get_obs()
            info = self._get_info()
            terminated = True
            reward = -100.0 #+ float(self._current_function) / float(self._state_FUNCTION_AMOUNT)
            return observation, reward, terminated, truncated, info

        self._current_function += 1

        reward = float(-self._state_F[0] + old_value)


        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info