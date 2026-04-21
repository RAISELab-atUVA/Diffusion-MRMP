from __future__ import annotations

import math
import random
from collections import defaultdict

from torch.utils.data import Sampler


class TaskBalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        subset,
        *,
        tasks_per_batch: int,
        trajectories_per_task: int,
        drop_last: bool = True,
    ):
        self.subset = subset
        self.dataset = subset.dataset
        self.tasks_per_batch = tasks_per_batch
        self.trajectories_per_task = trajectories_per_task
        self.drop_last = drop_last

        if self.tasks_per_batch <= 0:
            raise ValueError("tasks_per_batch must be positive")
        if self.trajectories_per_task <= 0:
            raise ValueError("trajectories_per_task must be positive")

        self.task_to_subset_indices = defaultdict(list)
        for subset_offset, dataset_index in enumerate(subset.indices):
            task_id = int(self.dataset.map_trajectory_id_to_task_id[int(dataset_index)])
            self.task_to_subset_indices[task_id].append(int(subset_offset))
        self.eligible_task_ids = list(self.task_to_subset_indices)
        if not self.eligible_task_ids:
            raise ValueError("Subset does not contain any task ids for task-balanced batching.")

    def __len__(self):
        batch_size = self.tasks_per_batch * self.trajectories_per_task
        if self.drop_last:
            return max(1, len(self.subset.indices) // batch_size)
        return max(1, math.ceil(len(self.subset.indices) / batch_size))

    def __iter__(self):
        per_task = {task_id: indices[:] for task_id, indices in self.task_to_subset_indices.items()}
        for indices in per_task.values():
            random.shuffle(indices)
        task_ids = self.eligible_task_ids[:]
        random.shuffle(task_ids)
        cursors = {task_id: 0 for task_id in task_ids}

        for _ in range(len(self)):
            random.shuffle(task_ids)
            batch = []
            selected = task_ids[: self.tasks_per_batch]
            if len(selected) < self.tasks_per_batch:
                if self.drop_last:
                    break
                selected = (selected * self.tasks_per_batch)[: self.tasks_per_batch]

            for task_id in selected:
                indices = per_task[task_id]
                if len(indices) < self.trajectories_per_task:
                    batch.extend(random.choices(indices, k=self.trajectories_per_task))
                    continue
                cursor = cursors[task_id]
                if cursor + self.trajectories_per_task > len(indices):
                    random.shuffle(indices)
                    cursor = 0
                batch.extend(indices[cursor:cursor + self.trajectories_per_task])
                cursors[task_id] = cursor + self.trajectories_per_task
            yield batch
