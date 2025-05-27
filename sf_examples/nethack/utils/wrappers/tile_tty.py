import gymnasium as gym
import numpy as np
from nle import nethack


class TileTTY(gym.Wrapper):
    def __init__(
        self,
        env,
        font_size=9,
        crop_size=12,
        rescale_font_size=(6, 6),
    ):
        super().__init__(env)
        self.font_size = font_size
        self.crop_size = crop_size
        self.rescale_font_size = rescale_font_size

        crop_rows = crop_size or nethack.nethack.TERMINAL_SHAPE[0]
        crop_cols = crop_size or nethack.nethack.TERMINAL_SHAPE[1]

        self.char_width = rescale_font_size[0]
        self.char_height = rescale_font_size[1]

        self.chw_image_shape = (
            2,
            crop_rows * self.char_height,
            crop_cols * self.char_width,
        )

        obs_spaces = {"screen_image": gym.spaces.Box(low=0, high=255, shape=self.chw_image_shape, dtype=np.uint8)}
        obs_spaces.update([(k, self.env.observation_space[k]) for k in self.env.observation_space])
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def crop_around_cursor(self, array, cursor_pos):
        rows, cols = array.shape[-2:]
        half_crop = self.crop_size // 2

        # Calculate boundaries
        start_h = max(0, cursor_pos[0] - half_crop)
        end_h = min(rows, cursor_pos[0] + half_crop + (self.crop_size % 2))
        start_w = max(0, cursor_pos[1] - half_crop)
        end_w = min(cols, cursor_pos[1] + half_crop + (self.crop_size % 2))

        # Create output array
        output = np.zeros(array.shape[:-2] + (self.crop_size, self.crop_size), dtype=array.dtype)

        # Calculate where in the output to place the valid data
        out_start_h = max(0, half_crop - cursor_pos[0])
        out_start_w = max(0, half_crop - cursor_pos[1])
        out_end_h = out_start_h + (end_h - start_h)
        out_end_w = out_start_w + (end_w - start_w)

        # Single slice operation
        output[..., out_start_h:out_end_h, out_start_w:out_end_w] = array[..., start_h:end_h, start_w:end_w]

        return output

    def _populate_obs(self, obs):
        tty_cursor = obs["tty_cursor"]
        tty_chars = obs["tty_chars"]
        tty_colors = obs["tty_colors"]
        tty = np.stack([tty_chars, tty_colors], axis=0)

        cropped_tty = self.crop_around_cursor(tty, tty_cursor)
        screen = np.tile(cropped_tty, (1, self.char_height, self.char_width))

        obs["screen_image"] = screen

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._populate_obs(obs)

        return obs, info

    def step(self, action):
        obs, reward, term, trun, info = super().step(action)
        self._populate_obs(obs)

        return obs, reward, term, trun, info


if __name__ == "__main__":
    env = gym.make("NetHackChallenge-v0")
    env = TileTTY(env)
    env.reset()
