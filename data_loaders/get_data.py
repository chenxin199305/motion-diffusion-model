from torch.utils.data import DataLoader

from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, t2m_prefix_collate


def get_dataset_class(name):
    """
    Retrieves the dataset class based on the given dataset name.

    Args:
        name (str): The name of the dataset.

    Returns:
        class: The dataset class corresponding to the given name.

    Raises:
        ValueError: If the dataset name is not supported.
    """
    if name == "amass":
        # TODO: support AMASS dataset
        from data_loaders.amass import AMASS
        return AMASS
    elif name == "uestc":
        from data_loaders.a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from data_loaders.a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')


def get_collate_fn(name, hml_mode='train', pred_len=0, batch_size=1):
    """
    Retrieves the appropriate collation function based on the dataset name and mode.

    Args:
        name (str): The name of the dataset.
        hml_mode (str): The mode for HumanML datasets ('train' or 'gt').
        pred_len (int): The length of the prediction sequence (for prefix collation).
        batch_size (int): The batch size.

    Returns:
        function: The collation function to use.
    """
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate

    if name in ["humanml", "kit"]:
        if pred_len > 0:
            return lambda x: t2m_prefix_collate(x, pred_len=pred_len)
        return lambda x: t2m_collate(x, batch_size)
    else:
        return all_collate


def get_dataset(name,
                num_frames,
                split='train',
                hml_mode='train',
                abs_path='.',
                fixed_len=0,
                device=None,
                autoregressive=False,
                cache_path=None):
    """
    Retrieves the dataset object based on the given parameters.

    Args:
        name (str): The name of the dataset.
        num_frames (int): The number of frames in the dataset.
        split (str): The data split ('train', 'test', etc.).
        hml_mode (str): The mode for HumanML datasets ('train' or 'gt').
        abs_path (str): The absolute path to the dataset.
        fixed_len (int): The fixed length for sequences.
        device (torch.device, optional): The device to use for the dataset.
        autoregressive (bool): Whether to use autoregressive mode.
        cache_path (str, optional): The path to cache the dataset.

    Returns:
        object: The dataset object.
    """
    DATA = get_dataset_class(name)

    if name in ["humanml", "kit"]:
        dataset = DATA(split=split,
                       num_frames=num_frames,
                       mode=hml_mode,
                       abs_path=abs_path,
                       fixed_len=fixed_len,
                       device=device,
                       autoregressive=autoregressive)
    else:
        dataset = DATA(split=split,
                       num_frames=num_frames)

    return dataset


def get_dataset_loader(name,
                       batch_size,
                       num_frames,
                       split='train',
                       hml_mode='train',
                       fixed_len=0,
                       pred_len=0,
                       device=None,
                       autoregressive=False):
    """
    Creates a DataLoader for the specified dataset.

    Args:
        name (str): The name of the dataset.
        batch_size (int): The batch size for the DataLoader.
        num_frames (int): The number of frames in the dataset.
        split (str): The data split ('train', 'test', etc.).
        hml_mode (str): The mode for HumanML datasets ('train' or 'gt').
        fixed_len (int): The fixed length for sequences.
        pred_len (int): The length of the prediction sequence (for prefix collation).
        device (torch.device, optional): The device to use for the dataset.
        autoregressive (bool): Whether to use autoregressive mode.

    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    dataset = get_dataset(name,
                          num_frames,
                          split=split,
                          hml_mode=hml_mode,
                          fixed_len=fixed_len,
                          device=device,
                          autoregressive=autoregressive)

    collate = get_collate_fn(name,
                             hml_mode,
                             pred_len,
                             batch_size)

    # dataset：要加载的数据集对象，必须实现 __getitem__ 和 __len__ 方法。DataLoader 会从该对象中按需读取数据样本。
    # batch_size：每个批次（batch）包含的数据样本数量。比如设为 32，则每次迭代会返回 32 个样本组成的一个 batch。
    # shuffle=True：每个 epoch 开始时，是否将数据集中的样本顺序打乱。设为 True 可以提高模型泛化能力，常用于训练阶段。
    # num_workers=8：用于加载数据的子进程数量。设为 8 表示会启动 8 个进程并行读取数据，加快数据加载速度。设为 0 时则在主进程中加载数据。
    # drop_last=True：如果数据集大小不能被 batch_size 整除，是否丢弃最后一个不完整的 batch。设为 True 时只返回完整的 batch，常用于训练。
    # collate_fn=collate：用于将一组样本合并成一个 batch 的函数。自定义 collate 函数可以处理不同格式或长度的数据，比如对序列进行 padding 等。
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle the data at every epoch.
        num_workers=8,  # Number of subprocesses to use for data loading.
        drop_last=True,  # Drop the last incomplete batch if the dataset size is not divisible by the batch size.
        collate_fn=collate  # Function to merge a list of samples into a batch.
    )

    return loader
