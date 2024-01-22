from tensorlayerx.dataflow import Dataset


class BaseDataSet(Dataset):
    def __init__(self, data, label, transforms=None):
        self.data = data
        self.label = label
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = []
        super(BaseDataSet, self).__init__()

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return self.transform(data, label)

    def __len__(self):
        return len(self.data)

    def register_transform_hook(self, transform_hook, index=None):
        if index is None:
            self.transforms.append(transform_hook)
        else:
            if not isinstance(index, int):
                raise ValueError(f"{index} is not int.")
            self.transforms.insert(index, transform_hook)

    def transform(self, data, label):
        for transform in self.transforms:
            data, label = transform(data, label)
        return data, label


class Datasets(dict):
    @classmethod
    def load(cls):
        raise NotImplementedError
