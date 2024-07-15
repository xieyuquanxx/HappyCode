import os

import gym
import numpy as np

from utils import write_video


class EnvRecorderWrapper(gym.Wrapper):
    def __init__(self, env, record_dir: str):
        super().__init__(env)
        self.record_dir = record_dir

        self.timeout = self.env.task.max_episode_steps
        self.num_steps = 0

        self.video_frames = []
        self.actions = []

    def reset(self):
        os.makedirs(self.record_dir, exist_ok=True)
        self.video_frames.clear()
        self.actions.clear()

        self.timeout = self.env.task.max_episode_steps
        self.num_steps = 0

        return super().reset()

    def step(self, action):
        observation, reward, done, info = super().step(action)
        frame = observation["pov"].astype(np.uint8)

        self.actions.append(action)
        self.video_frames.append(frame)

        self.num_steps += 1
        if self.num_steps >= self.timeout:
            done = True
        return observation, reward, done, info

    def save(self, output_video_file_path: str):
        write_video(output_video_file_path, self.video_frames)


class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs = self.env.reset()
        return obs

    def save(self, output_video_file_path: str):
        if hasattr(self.env, "save"):
            self.env.save(output_video_file_path)  # type: ignore
