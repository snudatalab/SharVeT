import argparse
import yaml


def add_args():
    paser = argparse.ArgumentParser(description="ShareModel")
    paser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    paser.add_argument(
        "--calibration_size",
        "--cs",
        help="calibration size",
        type=int,
        default=256,
    )
    paser.add_argument(
        "--dataset_name",
        help="dataset for load",
        type=str,
        default="slimpajama",
    )
    paser.add_argument(
        "--dataset_cache_dir",
        help="change dataset cache dir",
        type=str,
        default=None,
    )
    args, unknown = paser.parse_known_args()
    return args


class ShareConfig:
    """
    Central configuration object for SharVeT.

    This class loads the YAML configuration, maps model names to short names,
    exposes per-weight-shape metadata, and attaches command-line overrides
    parsed by `add_args()` to the configuration instance.
    """
    name_map = {
        "mistralai/Mistral-7B-v0.1": "mistral-7b",
        "meta-llama/Meta-Llama-3-8B": "llama3-8b"
    }

    weight_info = {
        "llama3-8b": {
            "self_attn.q_proj": (4096, 4096),
            "self_attn.k_proj": (4096, 1024),
            "self_attn.v_proj": (4096, 1024),
            "self_attn.o_proj": (4096, 4096),
            "mlp.up_proj": (4096, 14336),
            "mlp.gate_proj": (4096, 14336),
            "mlp.down_proj": (14336, 4096),
        },
        "mistral-7b": {
            "self_attn.k_proj": (4096, 1024),
            "self_attn.q_proj": (4096, 4096),
            "self_attn.v_proj": (4096, 1024),
            "self_attn.o_proj": (4096, 4096),
            "mlp.up_proj": (4096, 14336),
            "mlp.gate_proj": (4096, 14336),
            "mlp.down_proj": (14336, 4096),
        }
    }

    def __init__(self, cmd_args):
        cmd_args_dict = cmd_args.__dict__
        self.configuration = self.load_yaml_config(cmd_args.yaml_config_file)
        self.set_attr_from_config(self.configuration)
        for arg_key, arg_val in cmd_args_dict.items():
            setattr(self, arg_key, arg_val)

    @staticmethod
    def load_yaml_config(yaml_path):
        with open(yaml_path, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError("Yaml error - check yaml file")

    def set_attr_from_config(self, configuration):
        for _, param_family in configuration.items():
            for key, val in param_family.items():
                setattr(self, key, val)
