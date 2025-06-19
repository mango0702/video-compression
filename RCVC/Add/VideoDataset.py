import random

from pathlib import Path
import ipdb
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import datetime

class VideoFolder(Dataset):
    """Load a video folder database. Training and testing video clips
    are stored in a directorie containing mnay sub-directorie like Vimeo90K Dataset:

    .. code-block::

        - rootdir/
            train.list
            test.list
            - sequences/
                - 00010/
                    ...
                    -0932/
                    -0933/
                    ...
                - 00011/
                    ...
                - 00012/
                    ...

    training and testing (valid) clips are withdrew from sub-directory navigated by
    corresponding input files listing relevant folders.

    This class returns a set of three video frames in a tuple.
    Random interval can be applied to if subfolders includes more than 6 frames.

    Args:
        root (string): root directory of the dataset
        rnd_interval (bool): enable random interval [1,2,3] when drawing sample frames
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'test')
    """

    def __init__(
        self,
        root,
        rnd_interval=False,
        rnd_temp_order=False,
        transform=None,
        split="train",
    ):
        if transform is None:
            raise RuntimeError("Transform must be applied")

        splitfile = Path(f"{root}/{split}.list")
        splitdir = Path(f"{root}/sequences")
        
        print(splitfile)
        print(splitfile.is_file())

        if not splitfile.is_file():
            raise RuntimeError(f'Invalid file "{root}"')

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        with open(splitfile, "r") as f_in:
            self.sample_folders = [Path(f"{splitdir}/{f.strip()}") for f in f_in]

        self.max_frames = 2  # hard coding for now
        self.rnd_interval = rnd_interval
        self.rnd_temp_order = rnd_temp_order
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
#         print('￥￥￥￥￥￥￥￥￥￥￥')
        sample_folder = self.sample_folders[index]
        samples = sorted(f for f in sample_folder.iterdir() if f.is_file())
        max_interval = (len(samples) + 2) // self.max_frames
        interval = random.randint(1, max_interval) if self.rnd_interval else 1
        frame_paths = (samples[::interval])[: self.max_frames]
        frames = [self.transform(Image.open(p)) for p in frame_paths]
#         print('￥￥￥￥￥￥￥￥￥￥￥******')
        if self.rnd_temp_order:
            if random.random() < 0.5:
#                 print(random.random())
#                 print('￥￥￥￥￥￥￥￥￥￥￥******…………………')
#                 tensor = transforms.ToPILImage()(frames[0])
#                 timestamp = datetime.datetime.now().strftime("%H-%M-%S")#通过时间命名存储结果
#                 savepath = root_path +timestamp+ '_r.jpg'
#                 tensor.save(savepath)
                return frames[::-1] #交换两个帧的顺序
#         print('xxxxxxxxxxxx', len(frames))
        
#         print(frames[0].size())
#         tensor = transforms.ToPILImage()(frames[0])
# #         plt.imshow(tensor)
#         timestamp = datetime.datetime.now().strftime("%H-%M-%S")#通过时间命名存储结果
#         savepath = root_path1+timestamp + '_r.jpg'
#         tensor.save(savepath)
        return frames

    def __len__(self):
        return len(self.sample_folders)