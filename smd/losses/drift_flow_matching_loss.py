from smd.models import build_context


class DriftFlowMatchingLoss:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def loss_fn(self, flow_model, input_dict, dataset, step=None):
        del step
        traj_normalized = input_dict[f"{dataset.field_key_traj}_normalized"]
        context = build_context(flow_model, dataset, input_dict)
        hard_conds = input_dict.get("hard_conds", {})
        loss, info = flow_model.loss(traj_normalized, context, hard_conds)
        return {"diffusion_loss": loss}, info
