import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.autograd.functional import jacobian

from torch_robotics.environments import EnvEmptyNoWait2D
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.environments.utils import create_grid_spheres
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes

import pickle
class EnvEmptyNoWait2DExtraObjects(EnvEmptyNoWait2D):

    def __init__(self, tensor_args=None, **kwargs):
        instance_idx = kwargs['instance_idx']
        map_name = kwargs['map_name']

        file_name = '../../instances_data/'+map_name+'.pkl'
        with open(file_name, 'rb') as f:
            loaded_set = pickle.load(f)

        obs_info = loaded_set[instance_idx][0][0]

        # get the positions of the obstacles
        obs_pos = []
        for obs in obs_info:
            obs_pos.append(obs[0].tolist())
        
        obs_r = []
        for obs in obs_info:
            obs_r.append(obs[1])

        obj_extra_list = [
            MultiSphereField(
                np.array(obs_pos),
                np.array(obs_r),
                tensor_args=tensor_args
            ),
            # MultiSphereField(
            #     np.array(
            #     [[1, -1], [1, -0.5], [1, 0], [1, 0.5], [1, 1], [-1, -1], [-1, -0.5], [-1, 0.5], [-1, 1]]),
            #     np.array(
            #     [0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]
            #     )
            #     ,
            #     tensor_args=tensor_args
            # ),
            # MultiBoxField(
            #     np.array(  # (n, 2) array of box centers.
            #         [
            #             [0.0, -0.2],
            #         ]
            #     ),
            #     np.array(  # (n, 2) array of box sizes.
            #         [
            #             [0.4, 0.39],
            #         ]
            #     ),
            #     tensor_args=tensor_args
            # )
        ]

        super().__init__(
            name=self.__class__.__name__,
            obj_extra_list=[ObjectField(obj_extra_list, 'emptynowait2d-extraobjects')],
            tensor_args=tensor_args,
            **kwargs
        )


if __name__ == '__main__':
    env = EnvEmptyNoWait2DExtraObjects(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()
