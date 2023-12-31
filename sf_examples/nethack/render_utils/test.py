import sys
import time
from concurrent.futures import ThreadPoolExecutor

import gym
import nle  # NOQA: F401
import numpy as np
import render_utils as m
import tqdm
from PIL import Image as im

sys.path.append("/private/home/ehambro/fair/workspaces/wrapper-hackrl/hackrl")
import wrappers  # NOQA: E402


def create_env():
    return wrappers.RenderCharImagesWithNumpyWrapper(gym.make("NetHackChallenge-v0"), blstats_cursor=False)


def load_obs():
    e = create_env()
    e.reset()
    e.step(0)
    obs = e.step(1)[0]
    obs = e.step(5)[0]

    images = e.char_array.copy()

    return (
        obs["tty_chars"].copy(),
        obs["tty_colors"].copy(),
        obs["tty_cursor"].copy(),
        images,
        obs["screen_image"].copy(),
    )


def test_main():
    assert m.__version__ == "0.0.1"
    assert m.add(1, 2) == 3
    assert m.subtract(1, 2) == -1


np.set_printoptions(threshold=sys.maxsize)


def test(_):
    obs = [np.ascontiguousarray(x) for x in load_obs()]
    chars, colors, cursor, images, screen_image = obs
    out = np.zeros_like(screen_image, order="C")
    out = np.zeros((3, 72, 72), order="C", dtype=np.uint8)

    m.render_crop(chars, colors, cursor, images, out, screen_image)

    if not np.all(out == screen_image):
        scr_im = im.fromarray(np.transpose(screen_image, (1, 2, 0)))
        out_im = im.fromarray(np.transpose(out, (1, 2, 0)))

        # saving the final output
        # as a PNG file
        out_im.save("out_im.png")
        scr_im.save("scr_im.png")
        print(cursor[1] - 6, cursor[1] + 6)
        print(
            chars[
                max(cursor[0] - 6, 0) : cursor[0] + 6,
                max(cursor[1] - 6, 0) : cursor[1] + 6,
            ]
        )

        np.testing.assert_array_equal(out, screen_image)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=10) as tp:

        def fn(_):
            obs = [np.ascontiguousarray(x) for x in load_obs()]
            chars, colors, cursor, images, screen_image = obs

            out = np.zeros_like(screen_image, order="C")
            m.render_crop(chars, colors, cursor, images, out)
            np.testing.assert_array_equal(screen_image, out)

        def fn_batched(_):
            obs = [np.ascontiguousarray(x) for x in load_obs()]
            chars, colors, cursor, images, screen_image = obs
            obs = [np.ascontiguousarray(np.stack([x] * 10)) for x in (chars, colors, cursor, screen_image)]
            obs = [np.ascontiguousarray(np.stack([x] * 20)) for x in obs]
            (chars, colors, cursor, screen_image) = obs

            out = np.zeros_like(screen_image, order="C")
            m.render_crop(chars, colors, cursor, images, out)
            np.testing.assert_array_equal(screen_image, out)

        retries = 100
        batch_size = (100, 100)
        obs = []
        for _ in range(retries):
            this_obs = [np.ascontiguousarray(x) for x in load_obs()]
            chars, colors, cursor, images, screen_image = this_obs
            z = [np.ascontiguousarray(np.stack([x] * batch_size[0])) for x in (chars, colors, cursor, screen_image)]
            z = [np.ascontiguousarray(np.stack([x] * batch_size[1])) for x in z]
            (chars, colors, cursor, screen_image) = z
            this_obs = chars, colors, cursor, images, screen_image
            out = np.zeros_like(screen_image, order="C")

            obs.append((chars, colors, cursor, images, out))

        print("Testing")
        list(map(fn, tqdm.tqdm(range(200))))

        print("Testing Batched")
        list(map(fn_batched, tqdm.tqdm(range(200))))

        print("Profile Single Thread")
        start = time.time()
        for o in obs:
            chars, colors, cursor, images, out = o
            m.render_crop(chars, colors, cursor, images, out)
        t = time.time() - start
        print("Time:", t)
        print("SPS:", retries * batch_size[0] * batch_size[1] / t)

        print("Profile Batch")
        start = time.time()

        def _parallel(o, i):
            chars, colors, cursor, images, out = o
            m.render_crop(chars[i], colors[i], cursor[i], images, out[i])

        for o in obs:
            list(tp.map(_parallel, [(o, i) for i in range(batch_size[0])]))

        t = time.time() - start
        print("Time:", t)
        print("SPS:", retries * batch_size[0] * batch_size[1] / t)
