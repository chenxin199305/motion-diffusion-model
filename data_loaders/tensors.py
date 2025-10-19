import torch


def lengths_to_mask(lengths, max_len):
    """
    Converts a list of sequence lengths into a boolean mask tensor.

    Args:
        lengths (torch.Tensor): A tensor containing the lengths of sequences.
        max_len (int): The maximum length of the sequences.

    Returns:
        torch.Tensor: A boolean mask tensor of shape (len(lengths), max_len).
    """
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    """
    Collates a batch of tensors with varying sizes into a single tensor by zero-padding.

    Args:
        batch (list[torch.Tensor]): A list of tensors to collate.

    Returns:
        torch.Tensor: A single tensor containing the collated batch with zero-padding.
    """
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    """
    Collates a batch of data dictionaries into tensors and other structured outputs.

    Args:
        batch (list[dict]): A list of dictionaries containing data to collate.

    Returns:
        tuple: A tuple containing:
            - motion (torch.Tensor): The collated input tensor.
            - cond (dict): A dictionary containing additional collated data.
    """
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1)  # unsqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text'] for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    if 'prefix' in notnone_batches[0]:
        cond['y'].update({'prefix': collate_tensors([b['prefix'] for b in notnone_batches])})

    if 'orig_lengths' in notnone_batches[0]:
        cond['y'].update({'orig_lengths': torch.as_tensor([b['orig_lengths'] for b in notnone_batches])})

    if 'key' in notnone_batches[0]:
        cond['y'].update({'db_key': [b['key'] for b in notnone_batches]})

    return motion, cond


def t2m_collate(batch, target_batch_size):
    """
    Adapts a batch of data to a target batch size by repeating and truncating.

    Args:
        batch (list): A list of data items.
        target_batch_size (int): The desired batch size.

    Returns:
        tuple: A tuple containing:
            - motion (torch.Tensor): The collated input tensor.
            - cond (dict): A dictionary containing additional collated data.
    """
    repeat_factor = -(-target_batch_size // len(batch))  # Ceiling division
    repeated_batch = batch * repeat_factor
    full_batch = repeated_batch[:target_batch_size]  # Truncate to the target batch size
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2],  # b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
        'key': b[7] if len(b) > 7 else None,
    } for b in full_batch]
    return collate(adapted_batch)


def t2m_prefix_collate(batch, pred_len):
    """
    Adapts a batch of data for prefix-based collation (整理，校正).

    Args:
        batch (list): A list of data items.
        pred_len (int): The length of the prediction sequence.

    Returns:
        tuple: A tuple containing:
            - motion (torch.Tensor): The collated input tensor.
            - cond (dict): A dictionary containing additional collated data.

    用于把一个 batch 的样本整理成模型能够直接使用的输入格式。

    在 MDM 中有两种训练策略：
    - 全序列预测（standard）—— 模型输入整个动作序列，学习生成完整动作。
    - prefix-based 预测（autoregressive）—— 模型给定前半段动作（prefix），预测后半段动作（future）。

    这个函数 t2m_prefix_collate 专门用于 prefix 模式。
    它会从每个样本的动作序列中：
    - 取出前半段（prefix）作为条件输入；
    - 取出后半段（inp）作为模型的预测目标。
    最后再通过 collate() 函数打包成 batch 张量。
    """

    """
    1. 遍历每个样本并提取动作序列
        b 是一条样本，一般结构为（T2M数据中常见）：
        b = (
            <unused0>,     # caption dict
            <unused1>,     # audio/misc
            text,          # caption string
            <unused3>,     
            motion_data,   # numpy array, shape [seq_len, J]
            length,        # original motion length
            tokens,        # text token IDs
            key            # sample key (e.g. filename)
        )
    
    2. 构建输入和输出张量
        假设原动作 b[4] 是形状 [seq_len, J]
            → .T 变成 [J, seq_len]
            → unsqueeze(1) 变成 [J, 1, seq_len]
        
        这样的形状方便模型处理（[joint, channel, time] 风格）。

    3. 分割成 prefix 和预测部分
        [..., -pred_len:]   # 后 pred_len 帧 → 要预测的部分（inp）
        [..., :-pred_len]   # 前部分 → prefix 条件输入        
        
        inp = 后段动作，模型需要预测的目标；
        prefix = 前段动作，提供的上下文信息。
        
    4. 其他字段的含义
        'text': b[2]	原始文本描述（自然语言）
        'tokens': b[6]	文本的 token 表示（GloVe 或 BERT 向量）
        'lengths': pred_len	当前样本预测长度（可统一）
        'orig_lengths': b[5][0]	原始动作序列的长度（评估用）
        'key': b[7]	样本标识（如文件名）
        
    5. 返回组装结果
        collate() 是一个通用函数，用于：
            把 list of dict 合并成一个 batch dict；
            对齐张量维度（如 padding）；
            输出模型能直接使用的 batch 格式：
        motion 对应 'inp'（目标动作部分）
        cond 是一个 dict，包含 'prefix'、'text'、'tokens' 等条件输入。
    """
    adapted_batch = [
        {
            'inp': torch.tensor(b[4].T).float().unsqueeze(1)[..., -pred_len:],  # [seqlen, J] -> [J, 1, seqlen]
            'prefix': torch.tensor(b[4].T).float().unsqueeze(1)[..., :-pred_len],
            'text': b[2],  # b[0]['caption']
            'tokens': b[6],
            'lengths': pred_len,  # b[5],
            'orig_lengths': b[5][0],  # For evaluation
            'key': b[7] if len(b) > 7 else None,
        }
        for b in batch  # 遍历每个样本并提取动作序列
    ]

    return collate(adapted_batch)
