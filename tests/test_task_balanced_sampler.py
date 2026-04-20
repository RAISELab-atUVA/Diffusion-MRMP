import unittest

from torch.utils.data import Dataset, Subset

from smd.datasets.samplers import TaskBalancedBatchSampler


class DummyDataset(Dataset):
    def __init__(self):
        self.map_trajectory_id_to_task_id = {
            0: 0, 1: 0, 2: 0,
            3: 1, 4: 1, 5: 1,
            6: 2, 7: 2, 8: 2,
        }

    def __len__(self):
        return len(self.map_trajectory_id_to_task_id)

    def __getitem__(self, index):
        return index


class TaskBalancedBatchSamplerTests(unittest.TestCase):
    def test_groups_indices_by_task(self):
        dataset = DummyDataset()
        subset = Subset(dataset, list(range(len(dataset))))
        sampler = TaskBalancedBatchSampler(
            subset,
            tasks_per_batch=2,
            trajectories_per_task=2,
            drop_last=True,
        )

        batch = next(iter(sampler))
        task_ids = [dataset.map_trajectory_id_to_task_id[index] for index in batch]

        self.assertEqual(len(batch), 4)
        self.assertEqual(len(set(task_ids[:2])), 1)
        self.assertEqual(len(set(task_ids[2:])), 1)
        self.assertNotEqual(task_ids[0], task_ids[2])


if __name__ == "__main__":
    unittest.main()
