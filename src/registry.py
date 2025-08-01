from collections import defaultdict
import copy
import inspect
from logging import getLogger, warn
import warnings
import torch
from torch import nn
import logging
from typing import Tuple, Dict, Any, Callable


logger = getLogger("model_registry")
logger.setLevel(logging.INFO)


_MODEL_REGISTRY = {}
_PRETRAINED_CFG_REGISTRY = defaultdict(dict)


def register_model(func_or_name):
    if callable(func_or_name):
        _register_model(func_or_name.__name__, func_or_name)
        return func_or_name
    else:

        def wrapper(inner_func):
            _register_model(func_or_name, inner_func)
            return inner_func

        return wrapper


def _register_model(name, model_entrypoint):
    if name in _MODEL_REGISTRY:
        warnings.warn(f"Model {name} already registered, overwriting")
    _MODEL_REGISTRY[name] = model_entrypoint


class PretrainedConfig:
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        model_kwargs: Dict[str, Any] = {},
        file: str | None = None,
        preprocess_state_dict_fn: Callable | None = None,
        load_fn: Callable | None = None,
        load_strict: bool = False,
        **kwargs,
    ):
        self.mean = mean
        self.std = std
        self.model_kwargs = model_kwargs
        self.file = file
        self.preprocess_state_dict_fn = preprocess_state_dict_fn
        self.load_fn = load_fn
        self.load_strict = load_strict
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return vars(self).__repr__()

    def __getitem__(self, key):
        return getattr(self, key)


def register_pretrained_cfgs(cfg_dict: Dict[str, PretrainedConfig]):
    for k, v in cfg_dict.items():
        register_pretrained_cfg(k, v)


def register_pretrained_cfg(name, cfg: PretrainedConfig):
    if len(name.split(".")) == 2:
        name, tag = name.split(".")
    else:
        name = name
        tag = None
    _PRETRAINED_CFG_REGISTRY[name][tag] = cfg


def create_model(name, pretrained=None, checkpoint=None, **kwargs):
    if len(name.split(".")) == 2:
        name, pretrained_tag = name.split(".")
    else:
        pretrained_tag = None

    if pretrained is None and pretrained_tag is not None:
        pretrained = True

    if name in _MODEL_REGISTRY:
        model_entrypoint = _MODEL_REGISTRY[name]
    else: 
        raise ValueError(f"Unknown model {name}")

    if pretrained:
        if name not in _PRETRAINED_CFG_REGISTRY:
            raise ValueError(f"No pretrained cfg found for model {name}")
        if pretrained_tag is None:
            pretrained_tag = next(iter(_PRETRAINED_CFG_REGISTRY[name].keys()))

        pretrained_cfg: PretrainedConfig = copy.deepcopy(
            _PRETRAINED_CFG_REGISTRY[name][pretrained_tag]
        )
        logger.info(f"Loading pretrained config for {name}")
    else:
        pretrained_cfg = None

    if pretrained_cfg:
        model_kwargs = pretrained_cfg.model_kwargs.copy()
        model_kwargs.update(kwargs)
    else:
        model_kwargs = kwargs

    # If the model has a pretrained_cfg parameter, use it

    if pretrained_cfg and 'pretrained_cfg' in inspect.signature(model_entrypoint).parameters:
        model_kwargs['pretrained_cfg'] = pretrained_cfg
        pretrained_cfg = None
    if checkpoint and 'checkpoint' in inspect.signature(model_entrypoint).parameters: 
        model_kwargs['checkpoint'] = checkpoint
        checkpoint = None
    model = model_entrypoint(**model_kwargs)

    if pretrained_cfg and pretrained_cfg.file:
        logging.info(f"Loading pretrained weights from file {pretrained_cfg.file}")

        state_dict = torch.load(pretrained_cfg.file, map_location="cpu")
        
        if pretrained_cfg.preprocess_state_dict_fn:
            state_dict = pretrained_cfg.preprocess_state_dict_fn(state_dict)

        if pretrained_cfg.load_fn:
            msg = pretrained_cfg.load_fn(model, state_dict)
        else:
            msg = model.load_state_dict(state_dict, strict=pretrained_cfg.load_strict)

        logging.info(
            f"Loaded pretrained weights from {pretrained_cfg.file} with msg: {msg}"
        )

        model.pretrained_cfg = pretrained_cfg

    if checkpoint:
        logging.info(f"Loading checkpoint {checkpoint}")
        state_dict = torch.load(checkpoint, map_location="cpu")
        message = model.load_state_dict(state_dict, strict=False)
        logging.info(f"Load message: {message}")

    model.model_kwargs = model_kwargs
    model.name = name
    model.pretrained_tag = pretrained_tag

    return model


def list_models():
    return list(_MODEL_REGISTRY.keys())


def model_help(name):
    help(_MODEL_REGISTRY[name])

