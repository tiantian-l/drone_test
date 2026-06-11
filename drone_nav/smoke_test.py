"""Quick standalone sanity check for NavigationAviary (no JAX / DreamerV3).

Run from the workspace root:
    python -m drone_nav.smoke_test
"""
import numpy as np

from drone_nav.envs.nav_aviary import NavigationAviary


def main():
    env = NavigationAviary(gui=False, randomize_goal=True, num_obstacles=4)
    print("obs_space :", env.observation_space)
    print("act_space :", env.action_space)

    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape, (obs.shape, env.observation_space.shape)
    print("num obstacles spawned:", env.OBSTACLE_POS.shape[0])
    print("reset obs :", np.round(obs, 3), "goal:", np.round(info.get("goal", env.TARGET_POS), 3))

    ep_ret = 0.0
    n_success = n_collision = 0
    for t in range(200):
        # Naive proportional policy: fly toward the goal at moderate speed.
        rel = obs[0:3]
        direction = rel / (np.linalg.norm(rel) + 1e-8)
        action = np.array([[direction[0], direction[1], direction[2], 0.6]], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_ret += reward
        if terminated or truncated:
            n_success += int(info["is_success"])
            n_collision += int(info.get("is_collision", False))
            print(f"episode end @ t={t} ret={ep_ret:.2f} "
                  f"dist={info['distance']:.3f} success={info['is_success']} "
                  f"collision={info.get('is_collision', False)} "
                  f"obs_dist={info.get('obstacle_distance', 0.0):.3f}")
            obs, info = env.reset()
            ep_ret = 0.0
    env.close()
    print(f"smoke test OK (success={n_success}, collision={n_collision})")


if __name__ == "__main__":
    main()
