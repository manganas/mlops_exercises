from src.data.make_dataset import CorruptMnist
from pathlib import Path
from tests import _PROJECT_ROOT

in_folder = _PROJECT_ROOT + '/data'+'/raw'
out_folder = _PROJECT_ROOT + '/data'+'/processed'


N_train = 40000 # corrupted v2 : 40000, otherwise 25000
N_test = 5000
img_tensor_dims = [1,28,28]

train_dataset = CorruptMnist(train=True, in_folder=in_folder, out_folder=out_folder)
test_dataset = CorruptMnist(train=False, in_folder=in_folder, out_folder=out_folder)

def test_train_dataset_dims():
    assert train_dataset.data.shape[1] == img_tensor_dims[0]
    assert train_dataset.data.shape[2] == img_tensor_dims[1]
    assert train_dataset.data.shape[3] == img_tensor_dims[2]
    assert train_dataset.data.shape[0] == N_train

def test_test_dataset_dims():
    assert test_dataset.data.shape[0] == N_test
    assert test_dataset.data.shape[1] == img_tensor_dims[0]
    assert test_dataset.data.shape[2] == img_tensor_dims[1]
    assert test_dataset.data.shape[3] == img_tensor_dims[2]


def test_labels_represented():
    # I know it is mnist, so the number of distinct labels should be 10
    assert len(train_dataset.targets.unique())==10
    assert len(test_dataset.targets.unique()) == 10
    assert (train_dataset.targets.unique() == test_dataset.targets.unique()).all()



