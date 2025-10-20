import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from model.BERT.BERT_encoder import load_bert
from utils.misc import WeightedSum


class MDM(nn.Module):
    """
    Motion Diffusion Model (MDM) class for generating motion sequences.

    Attributes:
        modeltype (str): Type of the model.
        njoints (int): Number of joints in the motion data.
        nfeats (int): Number of features per joint.
        num_actions (int): Number of possible actions.
        translation (bool): Whether to include translation in the motion data.
        pose_rep (str): Representation of the pose (e.g., 'rot6d').
        glob (bool): Whether to include global position.
        glob_rot (bool): Whether to include global rotation.
        latent_dim (int): Dimensionality of the latent space.
        ff_size (int): Size of the feedforward layers.
        num_layers (int): Number of layers in the architecture.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        ablation (str): Ablation study configuration.
        activation (str): Activation function to use.
        legacy (bool): Whether to use legacy settings.
        data_rep (str): Data representation format.
        dataset (str): Dataset name.
        clip_dim (int): Dimensionality of the CLIP embeddings.
        arch (str): Architecture type ('trans_enc', 'trans_dec', 'gru').
        emb_trans_dec (bool): Whether to embed transformer decoder.
        clip_version (str): Version of the CLIP model.
        **kargs: Additional keyword arguments.

    MDM 类是整个模型的核心类，继承自 torch.nn.Module。
    它由 三大功能模块 构成：
    1. 输入编码模块：InputProcess
        把关节数据（rot6d、xyz等格式）映射到潜空间 (latent_dim)。

    2. 时序建模模块：
        可选择三种架构：
        Transformer Encoder (trans_enc)
        Transformer Decoder (trans_dec)
        GRU (gru)

    3. 输出解码模块：OutputProcess
        把潜空间表示还原为动作帧 [batch, njoints, nfeats, nframes]。

    InputProcess
        功能：把输入动作序列变换为 Transformer/GRU 可处理的时间序列特征。
        x: [batch, njoints, nfeats, nframes] → reshape → [seq_len, batch, latent_dim]

        不同 data_rep（数据表示）对应不同处理方式：
        - 'rot6d' / 'xyz'：直接线性映射。
        - 'rot_vel'：将第一帧位置 + 后续帧的速度分开嵌入。

        | 维度名称                | 含义         | 示例                          |
        | ------------------- | ---------- | --------------------------- |
        | **bs (batch size)** | 批次样本数      | 一次输入 32 个动作样本               |
        | **njoints**         | 骨架关节数      | 22（SMPL）、21（KIT）、52（AMASS）等 |
        | **nfeats**          | 每个关节的特征维度  | 6（rot6d）、3（xyz）             |
        | **nframes**         | 时间帧数（序列长度） | 60 帧、120 帧 等                |

    时序结构选择 (arch)
        (a) Transformer Encoder (trans_enc)
            每一帧看作一个 token。
            用 PositionalEncoding 加上时间位置感。
            输入序列经过若干层自注意力 (nn.TransformerEncoder)。
            ➡ 用于无条件或简单条件的动作生成。

        (b) Transformer Decoder (trans_dec)
            用文本嵌入 (CLIP/BERT) 作为 memory。
            xseq 为目标序列，emb 为文本 memory。
            支持 emb_trans_dec（时间步嵌入 + decoder 输入拼接）。
            ➡ 用于 文本到动作 (Text2Motion) 任务。

        (c) GRU (gru)
            更轻量的循环结构，用时间步嵌入重复加在每帧上。
            输入形状 [batch, seq, latent_dim]。
            ➡ 用于动作补全或中短序列预测。

    条件嵌入 (Conditional Embedding)
        MDM 可以在多种条件下生成动作：
        | 条件类型   | 模块                                   | 说明                  |
        | ------ | ------------------------------------ | ------------------- |
        | 文本     | CLIP / BERT                          | 将文本描述嵌入为特征向量        |
        | 动作类别   | `EmbedAction`                        | 将动作类别索引嵌入 latent 向量 |
        | 目标关节位置 | `EmbedTargetLoc{Single/Multi/Split}` | 在运动规划任务中使用目标点或方向    |

        这些条件会与时间嵌入 (time_emb) 结合，控制扩散生成。
        设计支持 多种组合策略：
            'add'：直接相加（默认）
            'concat'：拼接后送入 transformer

    TimestepEmbedder
        在扩散模型中，每一步噪声去除都有一个时间步 t。
        此模块通过：timestep → positional encoding → MLP → latent vector
        让模型知道当前生成的“扩散阶段”，以便在不同噪声水平下做不同的预测。

    OutputProcess
        把 Transformer/GRU 输出的 [seq_len, batch, latent_dim] 还原为动作序列：
        [seq_len, batch, latent_dim] → [batch, njoints, nfeats, nframes]
        支持多种输出表示：
            rot6d（常用，用于 SMPL）
            xyz（笛卡尔坐标）
            rot_vel（角速度）

    Rotation2xyz
        封装在 model.rotation2xyz.Rotation2xyz，用于：
        将预测的旋转参数转换为 3D 坐标；
        与 SMPL 模型对接；
        实现从潜空间到真实关节位置的最终映射。
    """

    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        """
        Initializes the MDM class with the given parameters.
        """
        super().__init__()

        # Model configuration
        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        # Pose and motion settings
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        # Latent space and architecture settings
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)
        self.input_feats = self.njoints * self.nfeats
        self.normalize_output = kargs.get('normalize_encoder_output', False)

        # Conditional settings
        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.mask_frames = kargs.get('mask_frames', False)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim)

        self.emb_policy = kargs.get('emb_policy', 'add')

        # Positional encoding
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=kargs.get('pos_embed_max_len', 5000))
        self.emb_trans_dec = emb_trans_dec

        # Sequence length settings
        self.pred_len = kargs.get('pred_len', 0)
        self.context_len = kargs.get('context_len', 0)
        self.total_len = self.pred_len + self.context_len
        self.is_prefix_comp = self.total_len > 0
        self.all_goal_joint_names = kargs.get('all_goal_joint_names', [])

        # Multi-target conditioning
        self.multi_target_cond = kargs.get('multi_target_cond', False)
        self.multi_encoder_type = kargs.get('multi_encoder_type', 'multi')
        self.target_enc_layers = kargs.get('target_enc_layers', 1)
        if self.multi_target_cond:
            if self.multi_encoder_type == 'multi':
                self.embed_target_cond = EmbedTargetLocMulti(self.all_goal_joint_names, self.latent_dim)
            elif self.multi_encoder_type == 'single':
                self.embed_target_cond = EmbedTargetLocSingle(self.all_goal_joint_names, self.latent_dim, self.target_enc_layers)
            elif self.multi_encoder_type == 'split':
                self.embed_target_cond = EmbedTargetLocSplit(self.all_goal_joint_names, self.latent_dim, self.target_enc_layers)

        # Architecture-specific initialization
        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        # Timestep embedding
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Conditional embedding
        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                # We support CLIP encoder and DistilBERT
                print('EMBED TEXT')

                self.text_encoder_type = kargs.get('text_encoder_type', 'clip')

                if self.text_encoder_type == "clip":
                    print('Loading CLIP...')
                    self.clip_version = clip_version
                    self.clip_model = self.load_and_freeze_clip(clip_version)
                    self.encode_text = self.clip_encode_text

                elif self.text_encoder_type == 'bert':
                    assert self.arch == 'trans_dec'
                    # assert self.emb_trans_dec == False # passing just the time embed so it's fine
                    print("Loading BERT...")
                    # bert_model_path = 'model/BERT/distilbert-base-uncased'
                    bert_model_path = 'distilbert/distilbert-base-uncased'
                    self.clip_model = load_bert(bert_model_path)  # Sorry for that, the naming is for backward compatibility
                    self.encode_text = self.bert_encode_text
                    self.clip_dim = 768

                else:
                    raise ValueError('We only support [CLIP, BERT] text encoders')

                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)

            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

        # Output processing
        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats)

        # Rotation to XYZ conversion
        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        """
        Returns the model parameters excluding those of the CLIP model.
        """
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        """
        Loads and freezes the CLIP model weights.

        Args:
            clip_version (str): Version of the CLIP model to load.

        Returns:
            clip_model: The loaded and frozen CLIP model.
        """
        clip_model, clip_preprocess = clip.load(clip_version,
                                                device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        """
        Masks the conditional input based on the training mode and mask probability.

        Args:
            cond (torch.Tensor): Conditional input tensor.
            force_mask (bool): Whether to force masking.

        Returns:
            torch.Tensor: Masked conditional input.
        """
        bs = cond.shape[-2]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def clip_encode_text(self, raw_text):
        """
        Encodes text using the CLIP model.

        Args:
            raw_text (list): List of strings containing input text prompts.

        Returns:
            torch.Tensor: Encoded text embeddings.
        """
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2  # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device)  # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length - context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device)  # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float().unsqueeze(0)

    def bert_encode_text(self, raw_text):
        """
        Encodes text using the BERT model.

        Args:
            raw_text (list): List of strings containing input text prompts.

        Returns:
            tuple: Encoded text embeddings and attention mask.
        """
        # enc_text = self.clip_model(raw_text)
        # enc_text = enc_text.permute(1, 0, 2)
        # return enc_text
        enc_text, mask = self.clip_model(raw_text)  # self.clip_model.get_last_hidden_state(raw_text, return_mask=True)  # mask: False means no token there
        enc_text = enc_text.permute(1, 0, 2)
        mask = ~mask  # mask: True means no token there, we invert since the meaning of mask for transformer is inverted  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        return enc_text, mask

    def forward(self, x, timesteps, y=None):
        """
        Forward pass of the MDM model.

        Args:
            x (torch.Tensor): Input motion data tensor.
            timesteps (torch.Tensor): Timestep embeddings.
            y (dict): Additional conditional inputs.

            x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
            timesteps: [batch_size] (int)

        Returns:
            torch.Tensor: Output motion data tensor.
        """
        bs, njoints, nfeats, nframes = x.shape
        time_emb = self.embed_timestep(timesteps)  # [1, bs, d]

        if 'target_cond' in y.keys():
            # NOTE: We don't use CFG for joints - but we do wat to support uncond sampling for generation and eval!
            time_emb += self.mask_cond(self.embed_target_cond(y['target_cond'], y['target_joint_names'], y['is_heading'])[None], force_mask=y.get('target_uncond', False))  # For uncond support and CFG
            # time_emb += self.embed_target_cond(y['target_cond'], y['target_joint_names'], y['is_heading'])[None]

        # Build input for prefix completion
        if self.is_prefix_comp:
            x = torch.cat([y['prefix'], x], dim=-1)
            y['mask'] = torch.cat([torch.ones([bs, 1, 1, self.context_len], dtype=y['mask'].dtype, device=y['mask'].device),
                                   y['mask']], dim=-1)

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            if 'text_embed' in y.keys():  # caching option
                enc_text = y['text_embed']
            else:
                enc_text = self.encode_text(y['text'])
            if type(enc_text) == tuple:
                enc_text, text_mask = enc_text
                if text_mask.shape[0] == 1 and bs > 1:  # casting mask for the single-prompt-for-all case
                    text_mask = torch.repeat_interleave(text_mask, bs, dim=0)
            text_emb = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))  # casting mask for the single-prompt-for-all case
            if self.emb_policy == 'add':
                emb = text_emb + time_emb
            else:
                emb = torch.cat([time_emb, text_emb], dim=0)
                text_mask = torch.cat([torch.zeros_like(text_mask[:, 0:1]), text_mask], dim=1)
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb = time_emb + self.mask_cond(action_emb, force_mask=force_mask)
        if self.cond_mode == 'no_cond':
            # unconstrained
            emb = time_emb

        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints * nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)  # [#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)  # [bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  # [bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru), axis=1)  # [bs, d+joints*feat, 1, #frames]

        x = self.input_process(x)

        # TODO - move to collate
        frames_mask = None
        is_valid_mask = y['mask'].shape[-1] > 1  # Don't use mask with the generate script
        if self.mask_frames and is_valid_mask:
            frames_mask = torch.logical_not(y['mask'][..., :x.shape[0]].squeeze(1).squeeze(1)).to(device=x.device)
            if self.emb_trans_dec or self.arch == 'trans_enc':
                step_mask = torch.zeros((bs, 1), dtype=torch.bool, device=x.device)
                frames_mask = torch.cat([step_mask, frames_mask], dim=1)

        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq, src_key_padding_mask=frames_mask)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((time_emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]

            if self.text_encoder_type == 'clip':
                output = self.seqTransDecoder(tgt=xseq, memory=emb, tgt_key_padding_mask=frames_mask)
            elif self.text_encoder_type == 'bert':
                output = self.seqTransDecoder(tgt=xseq, memory=emb, memory_key_padding_mask=text_mask, tgt_key_padding_mask=frames_mask)  # Rotem's bug fix
            else:
                raise ValueError()

            if self.emb_trans_dec:
                output = output[1:]  # [seqlen, bs, d]

        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        # Extract completed suffix
        if self.is_prefix_comp:
            output = output[self.context_len:]
            y['mask'] = y['mask'][..., self.context_len:]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output

    def _apply(self, fn):
        """
        Applies a function to all model parameters and buffers.
        """
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        """
        Sets the model to training mode.
        """
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding for sequence data, which adds information about the position of elements
    in a sequence to the input embeddings. This is commonly used in transformer models.

    Attributes:
        dropout (nn.Dropout): Dropout layer applied to the positional encoding.
        pe (torch.Tensor): Precomputed positional encodings stored as a buffer.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): Dimensionality of the model (size of the embeddings).
            dropout (float): Dropout rate to apply to the positional encodings.
            max_len (int): Maximum length of the sequence for which positional encodings are precomputed.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register the positional encodings as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encodings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [sequence_length, batch_size, embedding_dim].

        Returns:
            torch.Tensor: Input tensor with positional encodings added.
        """
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    """
    Embeds timesteps into a latent space using a learned embedding and positional encoding.

    Attributes:
        latent_dim (int): Dimensionality of the latent space.
        sequence_pos_encoder (PositionalEncoding): Positional encoding module.
        time_embed (nn.Sequential): Neural network for embedding timesteps.
    """

    def __init__(self, latent_dim, sequence_pos_encoder):
        """
        Initializes the TimestepEmbedder module.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            sequence_pos_encoder (PositionalEncoding): Positional encoding module.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        # Define the time embedding network
        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    """
    Processes input motion data into a latent representation.

    Attributes:
        data_rep (str): Representation format of the input data (e.g., 'rot6d', 'xyz', 'rot_vel').
        input_feats (int): Number of input features per frame.
        latent_dim (int): Dimensionality of the latent space.
        poseEmbedding (nn.Linear): Linear layer for embedding pose data.
        velEmbedding (nn.Linear, optional): Linear layer for embedding velocity data (used for 'rot_vel').
    """

    def __init__(self, data_rep, input_feats, latent_dim):
        """
        Initializes the InputProcess module.

        Args:
            data_rep (str): Representation format of the input data.
            input_feats (int): Number of input features per frame.
            latent_dim (int): Dimensionality of the latent space.
        """
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        """
        Processes the input tensor into a latent representation.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_joints, num_features, num_frames].

        Returns:
            torch.Tensor: Processed tensor of shape [sequence_length, batch_size, latent_dim].
        """
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [sequence_length, batch_size, latent_dim]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, batch_size, input_feats]
            first_pose = self.poseEmbedding(first_pose)  # [1, batch_size, latent_dim]
            vel = x[1:]  # [sequence_length-1, batch_size, input_feats]
            vel = self.velEmbedding(vel)  # [sequence_length-1, batch_size, latent_dim]
            return torch.cat((first_pose, vel), axis=0)  # [sequence_length, batch_size, latent_dim]
        else:
            raise ValueError("Unsupported data representation format.")


class OutputProcess(nn.Module):
    """
    Processes the latent representation back into the original motion data format.

    Attributes:
        data_rep (str): Representation format of the output data (e.g., 'rot6d', 'xyz', 'rot_vel').
        input_feats (int): Number of input features per frame.
        latent_dim (int): Dimensionality of the latent space.
        njoints (int): Number of joints in the motion data.
        nfeats (int): Number of features per joint.
        poseFinal (nn.Linear): Linear layer for generating pose data.
        velFinal (nn.Linear, optional): Linear layer for generating velocity data (used for 'rot_vel').
    """

    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        """
        Initializes the OutputProcess module.

        Args:
            data_rep (str): Representation format of the output data.
            input_feats (int): Number of input features per frame.
            latent_dim (int): Dimensionality of the latent space.
            njoints (int): Number of joints in the motion data.
            nfeats (int): Number of features per joint.
        """
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        """
        Processes the latent representation into the original motion data format.

        Args:
            output (torch.Tensor): Latent representation of shape [sequence_length, batch_size, latent_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_joints, num_features, num_frames].
        """
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [sequence_length, batch_size, input_feats]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, batch_size, latent_dim]
            first_pose = self.poseFinal(first_pose)  # [1, batch_size, input_feats]
            vel = output[1:]  # [sequence_length-1, batch_size, latent_dim]
            vel = self.velFinal(vel)  # [sequence_length-1, batch_size, input_feats]
            output = torch.cat((first_pose, vel), axis=0)  # [sequence_length, batch_size, input_feats]
        else:
            raise ValueError("Unsupported data representation format.")
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [batch_size, num_joints, num_features, num_frames]
        return output


class EmbedAction(nn.Module):
    """
    Embeds action indices into a latent space.

    Attributes:
        action_embedding (nn.Parameter): A learnable parameter matrix of shape (num_actions, latent_dim)
            where each row corresponds to the embedding of an action.
    """

    def __init__(self, num_actions, latent_dim):
        """
        Initializes the EmbedAction module.

        Args:
            num_actions (int): The number of possible actions.
            latent_dim (int): The dimensionality of the latent space.
        """
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        """
        Retrieves the embedding for the given action indices.

        Args:
            input (torch.Tensor): A tensor of shape [batch_size, 1] containing action indices.

        Returns:
            torch.Tensor: A tensor of shape [batch_size, latent_dim] containing the action embeddings.
        """
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output


class EmbedTargetLocSingle(nn.Module):
    """
    Embeds target locations into a latent space using a single MLP.

    Attributes:
        extended_goal_joint_names (list): List of joint names extended with 'traj' and 'heading'.
        target_cond_dim (int): Dimensionality of the input to the MLP (number of joints * 4).
        latent_dim (int): Dimensionality of the latent space.
        mlp (nn.Sequential): A multi-layer perceptron for embedding the input.
    """

    def __init__(self, all_goal_joint_names, latent_dim, num_layers=1):
        """
        Initializes the EmbedTargetLocSingle module.

        Args:
            all_goal_joint_names (list): List of joint names.
            latent_dim (int): Dimensionality of the latent space.
            num_layers (int): Number of layers in the MLP.
        """
        super().__init__()
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']
        self.target_cond_dim = len(self.extended_goal_joint_names) * 4  # 4 => (x,y,z,is_valid)
        self.latent_dim = latent_dim
        _layers = [nn.Linear(self.target_cond_dim, self.latent_dim)]
        for _ in range(num_layers):
            _layers += [nn.SiLU(), nn.Linear(self.latent_dim, self.latent_dim)]
        self.mlp = nn.Sequential(*_layers)

    def forward(self, input, target_joint_names, target_heading):
        """
        Embeds the target locations into the latent space.

        Args:
            input (torch.Tensor): A tensor of shape [batch_size, num_joints, 3] containing target locations.
            target_joint_names (list): List of joint names for each sample in the batch.
            target_heading (list): List of booleans indicating whether to include 'heading' for each sample.

        Returns:
            torch.Tensor: A tensor of shape [batch_size, latent_dim] containing the embedded target locations.
        """
        # TODO - generate validity from outside the model
        validity = torch.zeros_like(input)[..., :1]
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names
            for j in sample_joint_names_w_heading:
                validity[sample_idx, self.extended_goal_joint_names.index(j)] = 1.

        mlp_input = torch.cat([input, validity], dim=-1).view(input.shape[0], -1)
        return self.mlp(mlp_input)


class EmbedTargetLocSplit(nn.Module):
    """
    Embeds target locations into a latent space using separate MLPs for each joint.

    Attributes:
        extended_goal_joint_names (list): List of joint names extended with 'traj' and 'heading'.
        target_cond_dim (int): Dimensionality of the input for each joint (4).
        latent_dim (int): Dimensionality of the latent space.
        splited_dim (int): Dimensionality of the latent space per joint.
        mini_mlps (nn.ModuleList): A list of MLPs, one for each joint.
    """

    def __init__(self, all_goal_joint_names, latent_dim, num_layers=1):
        """
        Initializes the EmbedTargetLocSplit module.

        Args:
            all_goal_joint_names (list): List of joint names.
            latent_dim (int): Dimensionality of the latent space.
            num_layers (int): Number of layers in each MLP.
        """
        super().__init__()
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']
        self.target_cond_dim = 4
        self.latent_dim = latent_dim
        self.splited_dim = self.latent_dim // len(self.extended_goal_joint_names)
        assert self.latent_dim % len(self.extended_goal_joint_names) == 0
        self.mini_mlps = nn.ModuleList()
        for _ in self.extended_goal_joint_names:
            _layers = [nn.Linear(self.target_cond_dim, self.splited_dim)]
            for _ in range(num_layers):
                _layers += [nn.SiLU(), nn.Linear(self.splited_dim, self.splited_dim)]
            self.mini_mlps.append(nn.Sequential(*_layers))

    def forward(self, input, target_joint_names, target_heading):
        """
        Embeds the target locations into the latent space.

        Args:
            input (torch.Tensor): A tensor of shape [batch_size, num_joints, 3] containing target locations.
            target_joint_names (list): List of joint names for each sample in the batch.
            target_heading (list): List of booleans indicating whether to include 'heading' for each sample.

        Returns:
            torch.Tensor: A tensor of shape [batch_size, latent_dim] containing the embedded target locations.
        """
        # TODO - generate validity from outside the model
        validity = torch.zeros_like(input)[..., :1]
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names
            for j in sample_joint_names_w_heading:
                validity[sample_idx, self.extended_goal_joint_names.index(j)] = 1.

        mlp_input = torch.cat([input, validity], dim=-1)
        mlp_splits = [self.mini_mlps[i](mlp_input[:, i]) for i in range(mlp_input.shape[1])]
        return torch.cat(mlp_splits, dim=-1)


class EmbedTargetLocMulti(nn.Module):
    """
    Embeds target locations into a latent space using a separate embedding for each joint.

    Attributes:
        extended_goal_joint_names (list): List of joint names extended with 'traj' and 'heading'.
        extended_goal_joint_idx (dict): Mapping of joint names to their indices.
        n_extended_goal_joints (int): Number of extended goal joints.
        target_loc_emb (nn.ParameterDict): A dictionary of embedding layers for each joint.
        target_all_loc_emb (WeightedSum): A module to combine embeddings from all joints.
        latent_dim (int): Dimensionality of the latent space.
    """

    def __init__(self, all_goal_joint_names, latent_dim):
        """
        Initializes the EmbedTargetLocMulti module.

        Args:
            all_goal_joint_names (list): List of joint names.
            latent_dim (int): Dimensionality of the latent space.
        """
        super().__init__()

        # todo: use a tensor of weight per joint, and another one for biases, then apply a selection in one go like we to for actions
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']
        self.extended_goal_joint_idx = {joint_name: idx for idx, joint_name in enumerate(self.extended_goal_joint_names)}
        self.n_extended_goal_joints = len(self.extended_goal_joint_names)
        self.target_loc_emb = nn.ParameterDict({joint_name:
            nn.Sequential(
                nn.Linear(3, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim))
            for joint_name in self.extended_goal_joint_names})  # todo: check if 3 works for heading and traj
        # nn.Linear(3, latent_dim) for joint_name in self.extended_goal_joint_names})  # todo: check if 3 works for heading and traj
        self.target_all_loc_emb = WeightedSum(self.n_extended_goal_joints)  # nn.Linear(self.n_extended_goal_joints, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, input, target_joint_names, target_heading):
        """
        Embeds the target locations into the latent space.

        Args:
            input (torch.Tensor): A tensor of shape [batch_size, num_joints, 3] containing target locations.
            target_joint_names (list): List of joint names for each sample in the batch.
            target_heading (list): List of booleans indicating whether to include 'heading' for each sample.

        Returns:
            torch.Tensor: A tensor of shape [batch_size, latent_dim] containing the embedded target locations.
        """
        output = torch.zeros((input.shape[0], self.latent_dim), dtype=input.dtype, device=input.device)

        # Iterate over the batch and apply the appropriate filter for each joint
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names
            output_one_sample = torch.zeros((self.n_extended_goal_joints, self.latent_dim), dtype=input.dtype, device=input.device)
            for joint_name in sample_joint_names_w_heading:
                layer = self.target_loc_emb[joint_name]
                output_one_sample[self.extended_goal_joint_idx[joint_name]] = layer(input[sample_idx, self.extended_goal_joint_idx[joint_name]])
            output[sample_idx] = self.target_all_loc_emb(output_one_sample)
            # print(torch.where(output_one_sample.sum(axis=1)!=0)[0].cpu().numpy())

        return output
