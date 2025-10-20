import torch
from model.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_mode
from data_loaders.humanml_utils import HML_EE_JOINT_NAMES


def load_model_wo_clip(model, state_dict):
    # assert (state_dict['sequence_pos_encoder.pe'][:model.sequence_pos_encoder.pe.shape[0]] == model.sequence_pos_encoder.pe).all()  # TEST
    # assert (state_dict['embed_timestep.sequence_pos_encoder.pe'][:model.embed_timestep.sequence_pos_encoder.pe.shape[0]] == model.embed_timestep.sequence_pos_encoder.pe).all()  # TEST
    del state_dict['sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
    del state_dict['embed_timestep.sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') or 'sequence_pos_encoder' in k for k in missing_keys])


def create_model_and_diffusion(args, data):
    """
    Creates and returns a model and diffusion process based on the provided arguments and dataset.

    Args:
        args (argparse.Namespace): The configuration arguments for the model and diffusion process.
        data (object): The dataset object containing dataset-specific information.

    Returns:
        tuple: A tuple containing:
            - model (MDM): The created MDM model initialized with the provided arguments and dataset.
            - diffusion (SpacedDiffusion): The diffusion process configured with the specified parameters.
    """
    model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args, data):
    """
    Prepares and returns a dictionary of model arguments based on the provided configuration and dataset.

    Args:
        args (argparse.Namespace): The configuration arguments for the model.
        data (object): The dataset object containing dataset-specific information.

    Returns:
        dict: A dictionary containing the model arguments, including:
            - modeltype (str): The type of the model (default is an empty string).
            - njoints (int): The number of joints in the dataset.
            - nfeats (int): The number of features per joint.
            - num_actions (int): The number of actions in the dataset.
            - translation (bool): Whether to include translation in the model.
            - pose_rep (str): The representation of the pose (e.g., 'rot6d').
            - glob (bool): Whether to include global position information.
            - glob_rot (bool): Whether to include global rotation information.
            - latent_dim (int): The latent dimension size.
            - ff_size (int): The feedforward layer size.
            - num_layers (int): The number of layers in the model.
            - num_heads (int): The number of attention heads.
            - dropout (float): The dropout rate.
            - activation (str): The activation function to use (e.g., 'gelu').
            - data_rep (str): The data representation type (e.g., 'hml_vec').
            - cond_mode (str): The conditioning mode.
            - cond_mask_prob (float): The probability of masking conditioning inputs.
            - action_emb (str): The type of action embedding.
            - arch (str): The model architecture.
            - emb_trans_dec (bool): Whether to use embedding transformation in the decoder.
            - clip_version (str): The version of the CLIP model to use.
            - dataset (str): The name of the dataset.
            - text_encoder_type (str): The type of text encoder to use.
            - pos_embed_max_len (int): The maximum length for positional embeddings.
            - mask_frames (bool): Whether to mask frames in the input.
            - pred_len (int): The prediction length.
            - context_len (int): The context length.
            - emb_policy (str): The embedding policy (e.g., 'add').
            - all_goal_joint_names (list): A list of goal joint names.
            - multi_target_cond (bool): Whether to use multi-target conditioning.
            - multi_encoder_type (str): The type of multi-encoder to use.
            - target_enc_layers (int): The number of target encoder layers.
    """

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6
    all_goal_joint_names = []

    """
    数据集兼容层（HumanML3D / KIT）
    不同数据集的骨架结构（joint 数、feature 维度）不同：
        HumanML3D: 263 joints × 1 feature；
        KIT: 251 joints × 1 feature；
        SMPL-based 数据则是 25 joints × 6D rotation。
    这里抽象出统一接口，让模型不必关心数据集内部的差异。
    """
    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
        all_goal_joint_names = ['pelvis'] + HML_EE_JOINT_NAMES
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    # Compatibility with old models
    if not hasattr(args, 'pred_len'):
        args.pred_len = 0
        args.context_len = 0

    emb_policy = args.__dict__.get('emb_policy', 'add')
    multi_target_cond = args.__dict__.get('multi_target_cond', False)
    multi_encoder_type = args.__dict__.get('multi_encoder_type', 'multi')
    target_enc_layers = args.__dict__.get('target_enc_layers', 1)

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset,
            'text_encoder_type': args.text_encoder_type,
            'pos_embed_max_len': args.pos_embed_max_len, 'mask_frames': args.mask_frames,
            'pred_len': args.pred_len, 'context_len': args.context_len, 'emb_policy': emb_policy,
            'all_goal_joint_names': all_goal_joint_names, 'multi_target_cond': multi_target_cond,
            'multi_encoder_type': multi_encoder_type, 'target_enc_layers': target_enc_layers,
            }


def create_gaussian_diffusion(args):
    """
    Creates and returns a Gaussian diffusion process based on the provided arguments.

    Args:
        args (argparse.Namespace): The configuration arguments for the diffusion process, including:
            - diffusion_steps (int): The number of diffusion steps.
            - noise_schedule (str): The noise schedule to use.
            - sigma_small (bool): Whether to use a small sigma for variance.
            - lambda_vel (float): The weight for velocity loss.
            - lambda_rcxyz (float): The weight for coordinate constraint loss.
            - lambda_fc (float): The weight for physical constraint loss.
            - lambda_target_loc (float, optional): The weight for target location loss.

    Returns:
        SpacedDiffusion: A configured diffusion process object.
    """
    # Default parameters
    predict_xstart = True  # Always predict the original data x0
    steps = args.diffusion_steps  # Number of diffusion steps
    scale_beta = 1.0  # No scaling for beta
    timestep_respacing = ''  # Time step resampling (used for DDIM)
    learn_sigma = False  # Do not learn variance
    rescale_timesteps = False  # Do not rescale time steps

    # Get the beta schedule based on the noise schedule and steps
    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE  # Use Mean Squared Error (MSE) loss

    # Default to using all time steps if no resampling is specified
    if not timestep_respacing:
        timestep_respacing = [steps]

    # Check if lambda_target_loc is provided, otherwise set to default
    if hasattr(args, 'lambda_target_loc'):
        lambda_target_loc = args.lambda_target_loc
    else:
        lambda_target_loc = 0.0  # Default value

    # Create and return the SpacedDiffusion object
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,  # Weight for velocity loss
        lambda_rcxyz=args.lambda_rcxyz,  # Weight for coordinate constraint loss
        lambda_fc=args.lambda_fc,  # Weight for physical constraint loss
        lambda_target_loc=lambda_target_loc,  # Weight for target location loss
    )


def load_saved_model(model, model_path, use_avg: bool = False):
    """
    Loads a saved model from a checkpoint file and updates the provided model.

    Args:
        model (torch.nn.Module): The model to update with the loaded state dictionary.
        model_path (str): The path to the checkpoint file.
        use_avg (bool, optional): Whether to load the averaged model state if available. Defaults to False.

    Returns:
        torch.nn.Module: The updated model with the loaded state dictionary.
    """
    # Load the state dictionary from the checkpoint file
    state_dict = torch.load(model_path, map_location='cpu')

    # Use the averaged model state if specified and available
    if use_avg and 'model_avg' in state_dict.keys():
        print('loading avg model')
        state_dict = state_dict['model_avg']
    else:
        # Load the standard model state if available
        if 'model' in state_dict:
            print('loading model without avg')
            state_dict = state_dict['model']
        else:
            # Fallback if no averaged model is available
            print('checkpoint has no avg model, loading as usual.')

    # Update the model with the loaded state dictionary
    load_model_wo_clip(model, state_dict)
    return model
