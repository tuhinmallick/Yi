# Part of the code is adapted from https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/utils/data/raw_datasets.py
from datasets import load_dataset


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        print(f"init PromptRawDataset with dataname {dataset_name}")
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


# English dataset
class DahoasRmstaticDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        print("init DahoasRMStaicDataset")
        super().__init__(output_path, seed, local_rank, dataset_name)
        print(f"loading at dataset_name {dataset_name}")
        self.raw_datasets = load_dataset(
            "parquet",
            data_files={
                "train": f"{dataset_name}/data/train-00000-of-00001-2a1df75c6bce91ab.parquet",
                "test": f"{dataset_name}/data/test-00000-of-00001-8c7c51afc6d45980.parquet",
            },
        )
        self.dataset_name_clean = "Dahoas_rm_static"
        self.dataset_name = "Dahoas/rm-static"
        print("init rm-static dataset finished")

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample["prompt"]

    def get_chosen(self, sample):
        return sample["chosen"]

    def get_rejected(self, sample):
        return sample["rejected"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"] + sample["chosen"]

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"] + sample["rejected"]


class LocalJsonFileDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name, chat_path):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "local/jsonfile"
        self.dataset_name_clean = "jsonfile"
        self.raw_datasets = load_dataset(
            "json",
            data_files={
                "train": f"{chat_path}/data/train.json",
                "eval": f"{chat_path}/data/eval.json",
            },
        )

    def get_train_data(self):
        if self.raw_datasets["train"] is not None:
            return self.raw_datasets["train"]
        return None

    def get_eval_data(self):
        return None if self.raw_datasets["eval"] is None else self.raw_datasets["eval"]

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return " " + sample["prompt"] if sample["prompt"] is not None else None

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return " " + sample["chosen"] if sample["chosen"] is not None else None

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return " " + sample["rejected"] if sample["rejected"] is not None else None

    def get_prompt_and_chosen(self, sample):
        if sample["prompt"] is not None and sample["chosen"] is not None:
            return " " + sample["prompt"] + " " + sample["chosen"]
        return None

    def get_prompt_and_rejected(self, sample):
        if sample["prompt"] is not None and sample["rejected"] is not None:
            return " " + sample["prompt"] + " " + sample["rejected"]
        return None


class YiDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name, chat_path):
        super().__init__(output_path, seed, local_rank, dataset_name)
        print(f"data path is {chat_path}")
        self.dataset_name = "yi"
        self.dataset_name_clean = "yi"
        self.raw_datasets = load_dataset(
            "json",
            data_files={
                "train": f"{chat_path}/data/train.jsonl",
                "eval": f"{chat_path}/data/eval.jsonl",
            },
        )

    def get_train_data(self):
        if self.raw_datasets["train"] is not None:
            return self.raw_datasets["train"]
        return None

    def get_eval_data(self):
        return None if self.raw_datasets["eval"] is None else self.raw_datasets["eval"]
    
    def get_prompt(self, sample):
        return " " + sample["prompt"] if sample["prompt"] is not None else None

    def get_prompt_and_chosen(self, sample):
        if sample["prompt"] is not None and sample["chosen"] is not None:
            return " " + sample["prompt"] + " " + sample["chosen"]
        return None
