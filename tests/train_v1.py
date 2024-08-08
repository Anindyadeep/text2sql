import torch
import transformers
from typing import Sequence, Optional 
from dataclasses import dataclass, field
from text2sql.dataset.bird.train import BirdTrainSet
from text2sql.logger import setup_console_logger

logger = setup_console_logger("[TEST-V1-TRAIN-RUN]")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="google/gemma-2-2b-it")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(
        self, instances: Sequence[dict]
    ) -> dict[str, torch.Tensor]:
        input_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")
        )
        input_ids = torch.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        labels = torch.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )



def train():
    """Training Loop for SQL-LLaMA using LLaMA2 with HF-Transformers and Alpaca Instruction-Style"""

    logger.warning("Parsing HF-Transformers Arguments")

    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments)
    )
    model_args, training_args = parser.parse_args_into_dataclasses()

    logger.warning(
        "Setting up Pretrained-Model: " + str(model_args.model_name_or_path)
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    train_dataset = BirdTrainSet(
        data_path="./data",
        model_name_or_path=model_args.model_name_or_path,
        hf_token="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",
        num_rows=10
    )
    tokenizer = train_dataset.tokenizer
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    data_module = dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
    device = torch.device("cuda:0")

    trainer = transformers.Trainer(
        model=model.to(device),
        tokenizer=tokenizer,
        args=training_args,
        **data_module

    )
    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
