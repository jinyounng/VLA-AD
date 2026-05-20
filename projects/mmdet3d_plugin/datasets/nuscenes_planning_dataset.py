from mmdet.datasets import DATASETS

from .nuscenes_dataset import CustomNuScenesDataset


@DATASETS.register_module()
class CustomNuScenesPlanningDataset(CustomNuScenesDataset):
    """NuScenes dataset variant that keeps only full planning GT samples."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.test_mode:
            before = len(self.data_infos)
            self.data_infos = [
                info for info in self.data_infos
                if self._has_full_planning_gt(info)
            ]
            print(
                f'Filtered invalid planning samples: {before} -> '
                f'{len(self.data_infos)}'
            )

            if self.seq_mode:
                self._set_sequence_group_flag()

    @staticmethod
    def _has_full_planning_gt(info):
        if 'gt_planning' not in info or 'gt_planning_mask' not in info:
            return False
        mask = info['gt_planning_mask'][:].any(axis=-1)
        return int(mask.sum()) == 6
