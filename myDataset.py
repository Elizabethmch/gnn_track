"""
Fix Me: define your own dataloader with the function: get_dataloader
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch_geometric
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch_geometric.data import Data
from torch_geometric.data.separate import separate
import copy
import warnings

from config import args



def get_dataloader(loaderType, data_slice_id, num_slices, data_size, batch_size):
    loader = None
    if loaderType=="train":
        dataId = data_slice_id + 1
        trainset = MyDataset(dataId, dataAug=True)
        loader = torch_geometric.loader.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    elif loaderType=='test':
        dataId = args.num_slices_train + data_slice_id + 1
        testset = MyDataset(dataId)
        loader = torch_geometric.loader.DataLoader(testset, batch_size=batch_size, shuffle=False)
    elif loaderType=='apply':
        dataId = args.num_slices_train+args.num_slices_test + data_slice_id + 1
        applyset = MyDataset(dataId)
        loader = torch_geometric.loader.DataLoader(applyset, batch_size=batch_size, shuffle=False)
    return loader


class MyDataset(torch_geometric.data.Dataset):
    def __init__(self, data_id, transform=None, pre_transform=None, pre_filter=None, dataAug=False):
        super().__init__(transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        # Data augmentation
        self.datAug = dataAug

        self.data_dir = args.fileList
        data_list, slice_list = [], []
        for f in self.data_dir:
            data, slices = torch.load(f'{f}/data{data_id}.pt')
            data_list.append(data)
            slice_list.append(slices)
        self.data, self.slices = data_list, slice_list

    @property
    def processed_file_names(self):
        return [self.data_dir]


    def len(self) -> int:
        if self.slices is None:
            return 1

        num = 0
        self.break_points = [0]
        for slice in self.slices:
            for _, value in nested_iter(slice):
                entries = len(value) - 1
                num += entries if (entries < args.data_size or args.data_size==-1) else args.data_size
                self.break_points.append(num)
                break
        return num

    def get(self, idx: int) -> Data:
        if self.len() == 1:
            return copy.copy(self._data)

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        file_id = 0
        for i in range(1, len(self.break_points)):
            if idx < self.break_points[i]:
                idx -= self.break_points[i-1]
                file_id = i - 1
                break

        data = separate(
            cls=self._data[0].__class__,
            batch=self._data[file_id],
            idx=idx,
            slice_dict=self.slices[file_id],
            decrement=False,
        )

        # manage data
        data.x = data.tag[:,[0,1,2]]
        data.y -= 1

        # self._data_list[idx] = copy.copy(data)
        return  data

    @property
    def data(self) -> Any:
        msg1 = ("It is not recommended to directly access the internal "
                "storage format `data` of an 'InMemoryDataset'.")
        msg2 = ("The given 'InMemoryDataset' only references a subset of "
                "examples of the full dataset, but 'data' will contain "
                "information of the full dataset.")
        msg3 = ("The data of the dataset is already cached, so any "
                "modifications to `data` will not be reflected when accessing "
                "its elements. Clearing the cache now by removing all "
                "elements in `dataset._data_list`.")
        msg4 = ("If you are absolutely certain what you are doing, access the "
                "internal storage via `InMemoryDataset._data` instead to "
                "suppress this warning. Alternatively, you can access stacked "
                "individual attributes of every graph via "
                "`dataset.{attr_name}`.")
        msg = msg1
        if self._indices is not None:
            msg += f' {msg2}'
        if self._data_list is not None:
            msg += f' {msg3}'
            self._data_list = None
        msg += f' {msg4}'

        warnings.warn(msg)
        return self._data

    @data.setter
    def data(self, value: Any):
        self._data = value
        self._data_list = None

def nested_iter(node: Union[Mapping, Sequence]) -> Iterable:
    if isinstance(node, Mapping):
        for key, value in node.items():
            for inner_key, inner_value in nested_iter(value):
                yield inner_key, inner_value
    elif isinstance(node, Sequence):
        for i, inner_value in enumerate(node):
            yield i, inner_value
    else:
        yield None, node