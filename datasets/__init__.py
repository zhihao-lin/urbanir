from datasets.tnt import tntDataset
from .colmap import ColmapDataset
from .tnt import tntDataset
from .kitti360 import KittiDataset
from .waymo import WaymoDataset

dataset_dict = {
    'colmap': ColmapDataset,
    'tnt': tntDataset,
    'kitti': KittiDataset,
    'waymo': WaymoDataset
}
