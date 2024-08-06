from pathlib import Path
from typing import Optional
from text2sql.dataset.base import BaseDataset
from text2sql.dataset.bird.common import BirdDevInstance
from text2sql.dataset.utils import download_and_process_bird_dataset


class BirdDevDataset(BaseDataset):
    def __init__(
        self,
        path: str,
        system_prompt: Optional[str] = None,
        num_cot: Optional[int] = None,
    ) -> None:

        self.path = Path(path)
        dev_data = download_and_process_bird_dataset(
            download_folder=self.path, force=False, split="validation"
        )
        bird_instance = BirdDevInstance(data=dev_data)
        self.data = bird_instance.apply_prompt(
            system_prompt=system_prompt, num_cot=num_cot
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
