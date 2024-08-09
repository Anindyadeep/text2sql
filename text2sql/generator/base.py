import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional

from tqdm import tqdm

from text2sql.logger import setup_console_logger

logger = setup_console_logger(name="[GENERATOR]")


class BaseGenerator(ABC):
    def __init__(
        self, experiment_name: str, type: str, experiment_folder: Optional[str] = None
    ) -> None:
        """BaseGenerator is a base abstract class that can be extended for
        any kind of model / workflow based inferences. Each generation session
        is treated as a experiment and by default goes inside a ./experiment folder.

        Args:
            experiment_name (str): The name of the experiment
            type (str): The type of the experiment
            experiment_folder (Optional[str]): The folder in which all the generation results will be stored.
        """
        self.experiment_folder = (
            Path(experiment_folder)
            if experiment_folder is not None
            else Path("./experiments")
        )
        self.experiment_path = self.experiment_folder / type / experiment_name

        self.client = None

        if not self.experiment_path.exists():
            self.experiment_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created new experiment folder: {self.experiment_path}")
        else:
            logger.info(f"Experiment folder found in: {self.experiment_path}")

    @abstractmethod
    def generate(self, data_blob: dict, **kwargs: Optional[Any]) -> dict:
        """The main generation logic

        Arguments
            data_blob (dict): Single blob of the dataset which should contain atleast the following keywords:
                - db_path (str): The path in which db file exists to connect
                - prompt (str): The main prompt
        """
        raise NotImplementedError

    def postprocess(self, output_string: str):
        return output_string

    def generate_and_save_results(
        self, data: List[dict], **kwargs: Optional[Any]
    ) -> dict:
        existing_response = self.load_results_from_folder()
        if existing_response is None:
            for content in tqdm(data, total=len(data), desc="Generating results"):
                sql = self.postprocess(
                    self.generate(prompt=content["prompt"], **kwargs)
                )
                content["generated"] = sql

            json.dump(data, open(self.experiment_path / "predict.json", "w"), indent=4)
            logger.info(f"All responses are written to: {self.experiment_path}")
            return data

        logger.info("Already results found")
        return existing_response

    def load_results_from_folder(self):
        if self.experiment_path.exists():
            return json.load(open(self.experiment_path / "predict.json", "r"))
        return None
