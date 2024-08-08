from typing import Optional

from text2sql.dataset.bird.common import BirdDatasetBase


class BirdDevDataset(BirdDatasetBase):
    def __init__(
        self,
        data_path: str,
        system_prompt: Optional[str] = None,
        num_cot: Optional[int] = None,
        filter_by: Optional[tuple] = None,
        num_rows: Optional[int] = None,
        model_name_or_path: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        super().__init__(
            data_path=data_path,
            split="validation",
            system_prompt=system_prompt,
            num_cot=num_cot,
            filter_by=filter_by,
            num_rows=num_rows,
            model_name_or_path=model_name_or_path,
            hf_token=hf_token,
            logger_name="[BIRD-DEVSET]",
        )
