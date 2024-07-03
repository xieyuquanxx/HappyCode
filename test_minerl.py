#! cmd: xvgb-run -a python test_minerl.py
import minerl  # noqa pylint: disable=unused-import

from env import make_custom_env, custom_env_register


custom_env_register("testEnv-v0", 200, "desert", None)

env = make_custom_env(env_name="testEnv-v0")
print("custom env ok")

obs = env.reset()
print("env reset :)")
done = False

while not done:
    # ac = env.action_space.no_op()
    # Spin around to see what is around us
    # ac["camera"] = [0, 3]
    ac = env.action_space.sample()
    obs, reward, done, info = env.step(ac)
    # env.render()
    print("info:", info)

env.save("videos/test.mp4")
env.close()
print("success :)")
