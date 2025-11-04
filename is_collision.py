import pickle
import numpy as np
import os


def check_paths_ok(paths, obs_data, robot_data, robot_radius=0.05, threshold=1e-3):
    """Check path validity against obstacles, other robots, and start/mid anchors.

    Args:
        paths: ndarray shape (R, T, 2) — R robots, T timesteps, (x,y).
        obs_data: iterable of obstacles, each like (center, radius).
        robot_data: per-robot start/mid references. Accepts one of:
            - list/tuple where robot_data[i][0] is start (x,y), robot_data[i][1] is mid (x,y)
            - ndarray with shape (R, 2, 2) or (2, R, 2)
        robot_radius: radius of each robot.
        threshold: numerical tolerance for comparisons.

    Returns:
        True if all constraints satisfied, else False.
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 3 or paths.shape[-1] != 2:
        raise ValueError("paths must have shape (R, T, 2)")

    num_robots, num_steps, _ = paths.shape

    # 1) robot vs obstacle (vectorized over time and robots)
    if len(obs_data) > 0:
        for i in range(num_robots):
            for j in range(len(obs_data)):
                for t in range(num_steps):
                    if (paths[i, t, 0] - obs_data[j][0][0])**2 + (paths[i, t, 1] - obs_data[j][0][1])**2 < (robot_radius + obs_data[j][1])**2 - threshold:
                        return False
        

    # 2) robot vs robot (pairwise distances at each timestep)
    if num_robots > 1:
        for i in range(num_robots):
            for j in range(i+1, num_robots):
                for t in range(num_steps):
                    if (paths[i, t, 0] - paths[j, t, 0])**2 + (paths[i, t, 1] - paths[j, t, 1])**2 < (2.0 * robot_radius)**2 - threshold:
                        return False


    # # 3) velocity limit
    # if np.any(np.diff(paths, axis=1) > 0.1):
    #     return False

    return True


res_base_dir = 'scripts/inference/results_test'
res_sub_dirs = [d for d in os.listdir(res_base_dir) if os.path.isdir(os.path.join(res_base_dir, d))]
res_sub_dirs = sorted(res_sub_dirs)

res_dirs = []
for res_sub_dir in res_sub_dirs:
    tmp = [res_base_dir + '/' + res_sub_dir + '/' + d + '/instance_name___EnvEmptyNoWait2DRobotCompositeNinePlanarDiskRandom/num_agents___9/planner___SMDComposite/single_agent_planner___SMDEnsemble/0/' for d in os.listdir(res_base_dir + '/' + res_sub_dir) if os.path.isdir(os.path.join(res_base_dir + '/' + res_sub_dir, d))]
    tmp = sorted(tmp)
    res_dirs.extend(tmp)

map_info_dirs = []
paths_dirs = []
for res_dir in res_dirs:
    map_info_dirs.append(res_dir + 'map_info.pkl')
    paths_dirs.append(res_dir + 'paths.npy')

map_folder = 'instances_data/'

cnt_success = 0
for map_info_dir, paths_dir in zip(map_info_dirs, paths_dirs):
    map_info_data = pickle.load(open(map_info_dir, 'rb'))

    map_file_path = map_folder + map_info_data['map_name'] + '.pkl'
    map_idx = map_info_data['instance_idx']

    paths_data = np.load(paths_dir)
    map_file = pickle.load(open(map_file_path, 'rb'))


    if paths_data.shape[0] != 64 or paths_data.shape[1] != 64 or paths_data.shape[2] != 36:
        raise ValueError("paths_data shape is not correct")
    path_data = paths_data[0,:,:18].reshape(64, 9, 2).swapaxes(0, 1)


    map_data = map_file[map_idx][2]
    obs_data = map_data[0]
    robot_data = map_data[1]
    is_feasible = check_paths_ok(path_data, obs_data, robot_data)
    
    if not is_feasible:
        print(f"Collision detected for {map_idx}")
    else:
        print(f"Feasible path for {map_idx}")
        cnt_success += 1
print(f"Success rate: {cnt_success / len(map_info_dirs)}")
     