import os 
from typing import Optional, Any 
from premai import Prem
from text2sql.generator.base import BaseGenerator

class GeneratorPremAI(BaseGenerator):
    def __init__(
        self, 
        project_id: str, 
        experiment_name: str, type: str,
        model_name: str, experiment_folder: Optional[str]=None, api_key: Optional[str]=None
    ) -> None:
        self.project_id, self.model_name = project_id, model_name
        self.api_key = os.environ.get("PREMAI_API_KEY", None) if api_key is None else api_key
        super().__init__(
            experiment_name=experiment_name, 
            type=type, 
            experiment_folder=experiment_folder
        )

        self.client = Prem(api_key=self.api_key)

    def generate(self, data_blob: dict, **kwargs: Optional[Any]) -> dict:
        # TODO: Filter the kwargs with only prem only parameters 
        prompt = data_blob["prompt"]
        data_blob["generated"] = self.client.chat.completions.create(
            project_id=self.project_id,
            messages=[{
                "role":"user", "content": prompt
            }],
            **kwargs
        )
        return data_blob
