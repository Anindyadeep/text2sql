import json
import torch
from typing import Optional
from abc import abstractmethod, ABC

import torch.utils
import torch.utils.data


class BaseDataInstance(ABC):
    """A DataInstance is a particular blob of data which should contain
    all the information necessary to train / evaluate the model for that
    single instance.Particularly it should have the following attributes:

    - question: The input text to the model
    - sql: The target SQL query
    - db_path: The sqlite database file path (relevant for evaluation)

    For now we only support .sqlite databases. A BaseDataInstance
    should always have a self.data attribute which is a list of dictionary
    containing the above attributes.

    For split set to "train" the following attributes are required:
    - question
    - sql
    - table_name (optional)
    - knowledge (optional)

    For split set to "dev" the following attributes are required:
    - question
    - sql
    - db_path
    - table_name

    If you want to extend for your own data, make sure it has the following
    attributes otherwise it would be incompatible with other parts of the
    library.
    """

    def __repr__(self):
        self.data = self.data[:3] if len(self.data) > 3 else self.data
        return str(json.dumps(self.data, indent=4))

    @abstractmethod
    def schema_prompt(self) -> str:
        """This method should return a prompt for the schema of the database
        which is used in the prompt.
        """
        return NotImplementedError

    @abstractmethod
    def question_prompt(self, question: str) -> str:
        """This method should return a prompt for the instruction of the
        question which is used in the prompt.
        """
        return NotImplementedError

    @abstractmethod
    def add_few_shot_examples(self, k: int) -> str:
        """This method should return a prompt for the few shot examples which is used in the prompt."""
        return NotImplementedError

    def additional_prompt(self) -> str:
        """This method should return a prompt for the additional knowledge
        which is used in the prompt. This will go inside the end of the prompt
        after instruction and schema.
        """
        return ""

    @abstractmethod
    def apply_prompt(self, system_prompt: Optional[str] = ""):
        """This method should return the final prompt which will be used"""
        return NotImplementedError


class BaseDataset(torch.utils.data.Dataset, ABC):
    """A Base dataset class should be able to download the model and make it
    possible to either iterate over the dataset or get a particular instance
    and compatible for both training and evaluation.
    """

    @abstractmethod
    def __len__(self):
        """This method should return the length of the dataset"""
        return NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        """This method should return a particular instance of the dataset"""
        return NotImplementedError
