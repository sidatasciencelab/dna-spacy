from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os


def build_tokenizer(input_path, output_directory):

    paths = [str(x) for x in Path(".").glob(input_path)]

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])


    os.mkdir(output_directory)
    tokenizer.save_model("dnaBERTo")

    tokenizer = ByteLevelBPETokenizer(
        "./dnaBERTo/vocab.json",
        "./dnaBERTo/merges.txt",
    )

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)


def train_bert(model_path, corpus_path):

    config = RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path, max_len=512)
    model = RobertaForMaskedLM(config=config)
    dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=corpus_path,
            block_size=128,
        )
    data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )


    training_args = TrainingArguments(
        output_dir="./dnaBERTo",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(model_path)