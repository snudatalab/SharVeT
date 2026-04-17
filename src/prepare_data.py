from functools import partial
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling


def tokenize_func(example, tokenizer, content):
    return tokenizer(example[content])


def group_text(examples, context_length):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // context_length) * context_length
    result = {
        k: [t[i: i + context_length] for i in range(0, total_length, context_length)]
        for k, t in concatenated_examples.items()
    }
    return result


def prepare_data(dataset_name, tokenizer, context_length, dataset_cache_dir=None):
    """
    Prepare datasets and collators for a given dataset name.

    Parameters:
        dataset_name: One of {'wikitext', 'c4', 'slimpajama'}.
        tokenizer: HF tokenizer instance.
        context_length: Block length for grouping tokens.
        dataset_cache_dir : HF datasets cache directory.

    Returns:
        Tuple[Dataset|None, Dataset|None, Any, DataCollator]: Train, val, test tokens, and collator.
    """
    if dataset_name == "wikitext":
        train_dataset, val_dataset, tokenized_test, data_collator = prep_wikitext_2_raw_v1(context_length, tokenizer,
                                                                                           dataset_cache_dir)
    elif dataset_name == "c4":
        train_dataset, val_dataset, tokenized_test, data_collator = prep_c4(context_length, tokenizer,
                                                                            dataset_cache_dir)
    elif dataset_name == 'slimpajama':
        train_dataset, val_dataset, tokenized_test, data_collator = prep_slimpajama(context_length, tokenizer,
                                                                                dataset_cache_dir)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return train_dataset, val_dataset, tokenized_test, data_collator


def prep_wikitext_2_raw_v1(context_length, tokenizer, dataset_cache_dir=None):
    """
    Prepare Wikitext-2-raw-v1 datasets and collator.
    """
    print("load wikitext dataset")
    train_raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                                     dataset_cache_dir=dataset_cache_dir)
    val_raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation",
                                   dataset_cache_dir=dataset_cache_dir)
    test_raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", dataset_cache_dir=dataset_cache_dir)
    func = partial(tokenize_func, tokenizer=tokenizer, content="text")
    tokenized_train = train_raw_dataset.map(func, num_proc=4, batched=True, remove_columns="text")
    tokenized_val = val_raw_dataset.map(func, num_proc=4, batched=True, remove_columns="text")
    tokenized_test = tokenizer("\n\n".join(test_raw_dataset["text"]), return_tensors="pt")

    func = partial(group_text, context_length=context_length)
    train_dataset = tokenized_train.map(func, num_proc=4, batch_size=1024, batched=True)
    val_dataset = tokenized_val.map(func, num_proc=4, batch_size=1024, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")
    return train_dataset, val_dataset, tokenized_test, data_collator


def prep_c4(context_length, tokenizer, dataset_cache_dir=None):
    """
    Prepare a small C4 validation slice for perplexity evaluation.
    """
    print("load C4 dataset")
    val_raw_dataset = load_dataset("allenai/c4", split='validation', data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"})
    test_raw_dataset = val_raw_dataset

    tokenized_test_data = tokenizer("\n\n".join(test_raw_dataset['text'][0:2000]), return_tensors="pt")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")
    train_dataset = None
    val_dataset = None
    return train_dataset, val_dataset, tokenized_test_data, data_collator


def prep_slimpajama(context_length, tokenizer, dataset_cache_dir=None):
    """
    Prepare SlimPajama-6B train/val subsets for training and validation.
    """
    print("load slimpajama dataset")
    train_raw_dataset = load_dataset("DKYoon/SlimPajama-6B", split="train[:5000]",
                                     dataset_cache_dir=dataset_cache_dir)
    val_raw_dataset = load_dataset("DKYoon/SlimPajama-6B", split="validation[:500]",
                                     dataset_cache_dir=dataset_cache_dir)
    func = partial(tokenize_func, tokenizer=tokenizer, content="text")
    tokenized_train = train_raw_dataset.map(func, num_proc=4, batched=True, remove_columns=["text", "meta", "__index_level_0__"])
    tokenized_val = val_raw_dataset.map(func, num_proc=4, batched=True, remove_columns=["text", "meta", "__index_level_0__"])
    func = partial(group_text, context_length=context_length)
    train_dataset = tokenized_train.map(func, num_proc=4, batch_size=1024, batched=True)
    val_dataset = tokenized_val.map(func, num_proc=4, batch_size=1024, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")
    return train_dataset, val_dataset, None, data_collator
