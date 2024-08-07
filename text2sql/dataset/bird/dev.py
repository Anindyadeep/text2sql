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
        filter_by: Optional[tuple] = None,
        num_rows: Optional[int] = None,
    ) -> None:
        """BirdBench 
        """

        self.path = Path(path)
        dev_data = download_and_process_bird_dataset(
            download_folder=self.path, force=False, split="validation"
        )

        if filter_by is not None:
            filter_key, filter_value = filter_by
            assert filter_key in ["db_id", "difficulty"], ValueError(
                "Filtering is supported for keys: 'db_id' and 'difficulty'"
            )
            if filter_by == "difficulty":
                assert filter_value in [
                    "simple",
                    "moderate",
                    "challenging",
                ], ValueError(
                    "difficulty can either be: 'simple' or 'moderate' or 'challenging'"
                )
            else:
                available_dbs = set([content["db_id"] for content in dev_data])
                assert filter_value in available_dbs, ValueError(
                    f"available_dbs: {', '.join(available_dbs)}"
                )

            dev_data = [
                content for content in dev_data if content[filter_key] == filter_value
            ]

        if num_rows is not None:
            assert 0 < num_rows < len(dev_data), ValueError(
                f"num_rows should be more than 0 and less than {len(dev_data)}"
            )
            dev_data = dev_data[:num_rows]

        bird_instance = BirdDevInstance(data=dev_data)
        self.data = bird_instance.apply_prompt(
            system_prompt=system_prompt, num_cot=num_cot
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
