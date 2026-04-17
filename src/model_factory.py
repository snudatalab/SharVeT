from transformers import AutoConfig, AutoModelForCausalLM
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from config import ShareConfig
from utils import match_state_dict
from calib import Calib
from prepare_data import prepare_data
from utils import compute_num_basis
from group import change_model
from models.llama import ShareLlamaForCausalLM
from models.mistral import ShareMistralForCausalLM
from similarity import Similarity


def create_model(config):
    """
    Create the SharVeT model and corresponding teacher model.

    This function prepares the tokenizer and HF configuration, optionally
    builds calibration data, computes layer groups and basis sizes for
    shared/private parts, initializes the SharVeT model, optionally loads
    matched weights, and applies basis and coefficient transformations.

    Parameters:
        config: Configuration with model, dataset, grouping, compression and calibration options.

    Returns:
        (model, std_model): The SharVeT model and the original teacher model.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if config.model_type == "llama":
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.pad_token = "[PAD]"
    print("Start create model!")
    model_config = AutoConfig.from_pretrained(config.model_name)
    model_config.use_cache = False
    model_config.on_refinement = config.on_refinement
    std_model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto")

    if config.build_calib:
        train_dataset, val_dataset, tokenized_test, data_collator = prepare_data(config.dataset_name, tokenizer,
                                                                                    config.context_length, config.dataset_cache_dir)
        # Prepare Dataloader for calibration data
        torch.manual_seed(2023)
        index = torch.randperm(len(train_dataset))
        index = index[:config.calibration_size]
        subset = Subset(train_dataset, index)
        dataloader = DataLoader(subset, batch_size=config.calib_batch_size, shuffle=False, collate_fn=data_collator,
                                pin_memory=True, num_workers=4)

        print("Start create calib!")
        calib_names = []
        if hasattr(config, "k_name"):
            # calibration data for k, q, v is the same
            calib_names.append(config.k_name)
        if hasattr(config, "attn_name"):
            calib_names.append(config.attn_name)
        calib_names.append(config.o_name)
        calib_names.append(config.up_name)
        calib_names.append(config.down_name)
        Calib.build_calibration_dataset(std_model, dataloader, calib_names, config.model_type, config.calib_path)
        print("Calib build done!")
    
    short_model_name = ShareConfig.name_map[config.model_name]

    # Share Part
    names = config.share_part
    if config.num_group:
        print("similarity")
        train_dataset, _, _, _ = prepare_data(config.dataset_name, tokenizer, config.context_length, config.dataset_cache_dir)
        similarity = Similarity(std_model, names, train_dataset, config, include_bias=config.include_bias_in_similarity)
    for name in names:
        print("Config for {}".format(name))
        nx, nf = ShareConfig.weight_info[short_model_name][getattr(config, name + "_name")]
        if config.num_group:
            num_group = config.num_group
            group = similarity.get_groups_from_similarity(num_group, name)
            print(f"Generated groups for {name}: {group}")
        else:
            num_group = model_config.num_hidden_layers // config.group_size
            rest = model_config.num_hidden_layers % config.group_size
            gs = config.group_size
            group = [[gs * i + j for j in range(config.group_size)] for i in range(num_group)]
            if rest != 0:
                group += [[num_group * config.group_size + i for i in range(rest)]]
        setattr(model_config, name + "_groups", group)
        
        print("similarity of {}".format(name))
        print(f"minimum_basis_ratio: {config.minimum_basis_ratio}")
        num_basis = compute_num_basis(nx, nf, config.compression_ratio, group, config.strategy, similarity.similarity[name], minimum_basis_ratio=config.minimum_basis_ratio)
        
        setattr(model_config, "num_basis_" + name, num_basis)
        print("num_basis {}".format(num_basis))
    if config.num_group:
        del similarity

    # Private Part
    names = config.private_part
    for name in names:
        print("Config for {}".format(name))
        setattr(model_config, name + "_groups", [[i] for i in range(model_config.num_hidden_layers)])
        nx, nf = ShareConfig.weight_info[short_model_name][getattr(config, name + "_name")]
        num_basis = compute_num_basis(nx, nf, config.compression_ratio, [[1]])
        setattr(model_config, "num_basis_" + name, num_basis)
        print("num_basis {}".format(num_basis))

    if config.model_type == "llama":
        model = ShareLlamaForCausalLM(model_config)
    elif config.model_type == "mistral":
        model = ShareMistralForCausalLM(model_config)
    else:
        raise NotImplementedError

    print("Model init finished!")
    if not hasattr(config, "tfs"):
        matched_state_dict, _ = match_state_dict(model.state_dict(), std_model.state_dict())
        model.load_state_dict(matched_state_dict, strict=False)
        names = config.share_part + config.private_part
        for name in names:
            print("Change {}".format(name))
            model = change_model(std_model=std_model,
                                    model=model,
                                    model_type=config.model_type,
                                    groups=getattr(model_config, name + "_groups"),
                                    name=getattr(config, name + "_name"),
                                    step=ShareConfig.weight_info[short_model_name][getattr(config, name + "_name")][1],
                                    num_basis=getattr(model_config, "num_basis_" + name),
                                    basis_name=name + "_basis",
                                    calib_path=config.calib_path,
                                    on_refinement=getattr(config, 'on_refinement', False),
                                    )

    return model, std_model
