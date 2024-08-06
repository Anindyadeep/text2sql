import torch
from typing import Sequence
from text2sql.dataset.bird.common import BirdDevInstance
from text2sql.dataset.base import BaseDataset
from text2sql.dataset.utils import download_and_process_bird_dataset
from text2sql.logger import setup_console_logger

logger = setup_console_logger(name="[BIRD-TRAINSET]")

# Todo: Add different types of prompt style


class BirdTrainSet(BaseDataset):
    def __init__(self, path: str, system_prompt: str, num_cot: int):
        self.path = path

        train_data = download_and_process_bird_dataset(
            download_folder=self.path, force=False, split="train"
        )
        self.data = BirdDevInstance(data=train_data).apply_prompt(
            system_prompt=system_prompt, 
            num_cot=num_cot
        )

        # Main training dataset logic 
        sources, target = [], []
        for instance in self.data:
            sources.append(instance["prompt"])
            target.append(instance["sql"])
        
        logger.info("Formatting inputs to Alpaca style")



    def __len__(self):
        return len(self.data)
    

    
