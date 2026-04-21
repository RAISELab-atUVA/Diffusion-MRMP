import argparse
import pickle
from pathlib import Path

import numpy as np

from smd.runtime import resolve_runtime_config


def check_paths_ok(paths, obs_data, robot_data, robot_radius=0.05, threshold=1e-3):
    """Check path validity against obstacles and other robots."""
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 3 or paths.shape[-1] != 2:
        raise ValueError("paths must have shape (R, T, 2)")

    num_robots, num_steps, _ = paths.shape

    if len(obs_data) > 0:
        for i in range(num_robots):
            for j in range(len(obs_data)):
                for t in range(num_steps):
                    if (
                        (paths[i, t, 0] - obs_data[j][0][0]) ** 2
                        + (paths[i, t, 1] - obs_data[j][0][1]) ** 2
                        < (robot_radius + obs_data[j][1]) ** 2 - threshold
                    ):
                        return False

    if num_robots > 1:
        for i in range(num_robots):
            for j in range(i + 1, num_robots):
                for t in range(num_steps):
                    if (
                        (paths[i, t, 0] - paths[j, t, 0]) ** 2
                        + (paths[i, t, 1] - paths[j, t, 1]) ** 2
                        < (2.0 * robot_radius) ** 2 - threshold
                    ):
                        return False

    return True


def parse_args(runtime):
    parser = argparse.ArgumentParser(description="Check composite planning outputs for collisions.")
    parser.add_argument(
        "--results-root",
        default=runtime["experiments_root"],
        help="Root directory containing experiment outputs.",
    )
    parser.add_argument(
        "--experiment-name",
        default="EnvEmptyNoWait2DRobotCompositeNinePlanarDiskRandom",
        help="Experiment instance name embedded in the result directory tree.",
    )
    parser.add_argument("--num-agents", type=int, default=9, help="Number of agents encoded in the saved result tensors.")
    parser.add_argument(
        "--map-folder",
        default=runtime["instances_root"],
        help="Directory containing pickled map files.",
    )
    parser.add_argument("--planner", default="SMDComposite", help="Planner directory label to match.")
    parser.add_argument("--single-agent-planner", default="SMDEnsemble", help="Single-agent planner directory label to match.")
    return parser.parse_args()


def resolve_path(path_str, repo_root):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def find_result_dirs(results_root, experiment_name, num_agents, planner_name, single_agent_planner):
    result_dirs = []
    for map_info_path in sorted(results_root.rglob("map_info.pkl")):
        result_dir = map_info_path.parent
        paths_path = result_dir / "paths.npy"
        if not paths_path.exists():
            continue

        parts = set(result_dir.parts)
        if f"instance_name___{experiment_name}" not in parts:
            continue
        if f"num_agents___{num_agents}" not in parts:
            continue
        if f"planner___{planner_name}" not in parts:
            continue
        if f"single_agent_planner___{single_agent_planner}" not in parts:
            continue

        result_dirs.append(result_dir)

    return result_dirs


def extract_robot_paths(paths_data, num_agents):
    if paths_data.ndim != 3:
        raise ValueError("paths.npy must have shape (samples, horizon, state_dim)")
    if paths_data.shape[0] < 1 or paths_data.shape[1] < 1:
        raise ValueError("paths.npy must contain at least one sampled trajectory")

    pos_dim = 2 * num_agents
    if paths_data.shape[2] < pos_dim:
        raise ValueError(
            f"paths.npy last dimension must be at least {pos_dim} for {num_agents} agents; "
            f"got {paths_data.shape[2]}"
        )

    first_sample = paths_data[0, :, :pos_dim]
    return first_sample.reshape(paths_data.shape[1], num_agents, 2).swapaxes(0, 1)


def evaluate_result_dir(result_dir, map_folder, num_agents):
    with (result_dir / "map_info.pkl").open("rb") as handle:
        map_info_data = pickle.load(handle)

    map_file_path = map_folder / f"{map_info_data['map_name']}.pkl"
    with map_file_path.open("rb") as handle:
        map_file = pickle.load(handle)

    paths_data = np.load(result_dir / "paths.npy")
    path_data = extract_robot_paths(paths_data, num_agents)

    map_data = map_file[map_info_data["instance_idx"]][2]
    obs_data = map_data[0]
    robot_data = map_data[1]
    return map_info_data["instance_idx"], check_paths_ok(path_data, obs_data, robot_data)


def main():
    runtime = resolve_runtime_config()
    args = parse_args(runtime)
    repo_root = Path(__file__).resolve().parent
    results_root = resolve_path(args.results_root, repo_root)
    map_folder = resolve_path(args.map_folder, repo_root)

    result_dirs = find_result_dirs(
        results_root=results_root,
        experiment_name=args.experiment_name,
        num_agents=args.num_agents,
        planner_name=args.planner,
        single_agent_planner=args.single_agent_planner,
    )
    if not result_dirs:
        raise FileNotFoundError(f"No result directories found under {results_root}")

    cnt_success = 0
    for result_dir in result_dirs:
        map_idx, is_feasible = evaluate_result_dir(result_dir, map_folder, args.num_agents)
        if not is_feasible:
            print(f"Collision detected for {map_idx}")
        else:
            print(f"Feasible path for {map_idx}")
            cnt_success += 1

    print(f"Success rate: {cnt_success / len(result_dirs)}")


if __name__ == "__main__":
    main()
