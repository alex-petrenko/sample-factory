import deepmind_lab
import matplotlib.pyplot as plt

observations = ['DEBUG.CAMERA.TOP_DOWN']
env = deepmind_lab.Lab('openfield_map2_fixed_loc3', observations,
                       config={'width': '640',    # screen size, in pixels
                               'height': '480',   # screen size, in pixels
                               },  # lt_chasm option.
                       renderer='hardware')       # select renderer.
env.reset()
obs = env.observations()
print(obs['DEBUG.CAMERA.TOP_DOWN'].dtype)
plt.figure()
plt.imshow(obs)
plt.savefig('tryfig.png')