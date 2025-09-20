import os
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset


def train_tokenizer(data_file: str, output_dir: str):
    """Trains a new BPE tokenizer on the filtered dataset."""
    print(f"Training tokenizer on {data_file}...")

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=[data_file], vocab_size=52000, min_frequency=2, special_tokens=[
        "<unk>",
        "<eos>",
    ])

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save files to the specified directory
    tokenizer.save_model(output_dir)
    print(f"Tokenizer trained and saved to {output_dir}")


def find_latest_checkpoint(output_dir: str):
    """Return the path to the latest checkpoint in output_dir, or None if none exist."""
    try:
        return get_last_checkpoint(output_dir)
    except Exception:
        # If the directory doesn't exist or is not a HF checkpoint dir
        return None


def main():
    # --- Configuration ---
    FILTERED_DATA_FILE = "/Users/jia/Documents/Thesis/experiments/filtered_dataset_only_and.txt"
    TOKENIZER_PATH = "./filtered_tokenizer"
    MODEL_OUTPUT_PATH = "./filtered_gpt2_model"

    # --- Step 1: Train the Tokenizer ---
    # Check if the tokenizer has already been trained
    if not os.path.exists(os.path.join(TOKENIZER_PATH, "vocab.json")):
        if not os.path.exists(FILTERED_DATA_FILE):
            print(
                f"Error: Filtered data file '{FILTERED_DATA_FILE}' not found.")
            print("Please run the filtering script first to generate the data.")
            return
        train_tokenizer(FILTERED_DATA_FILE, TOKENIZER_PATH)
    else:
        print("Tokenizer already found, skipping training.")

    # --- Step 2: Load the Trained Tokenizer ---
    print("Loading trained tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_PATH)
    # Ensure special tokens are set for the model config
    tokenizer.add_special_tokens({
        "eos_token": "<eos>",
        "unk_token": "<unk>"
    })

    # Set padding token (required for batching)
    tokenizer.pad_token = tokenizer.eos_token  # Use <eos> as padding token

    # --- Step 3: Configure and Initialize the GPT-2 Model ---
    print("Configuring and initializing GPT-2 model from scratch...")
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        n_positions=1024,  # Max sequence length
        n_embd=1280,       # Embedding size (GPT-2-Large)
        n_layer=36,        # Number of layers (GPT-2-Large)
        n_head=20          # Number of attention heads (GPT-2-Large)
    )
    model = GPT2LMHeadModel(config)
    print(f"Model created with {model.num_parameters():,} parameters.")

    # --- Step 4: Prepare the Dataset for Training ---
    print("Loading and preparing dataset for training...")
    # Load the filtered text file as a dataset object
    dataset = load_dataset('text', data_files={'train': FILTERED_DATA_FILE})

    # Split into train/validation (90/10 split)
    train_test_split = dataset['train'].train_test_split(
        test_size=0.1, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=1024)

    tokenized_train_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=['text'])
    tokenized_eval_dataset = eval_dataset.map(
        tokenize_function, batched=True, remove_columns=['text'])

    # Use a data collator to handle batching and padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal language modeling (not masked)
    )

    # --- Step 5: Set Up the Trainer ---
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_PATH,
        overwrite_output_dir=True,
        num_train_epochs=1,  # More epochs for better convergence
        per_device_train_batch_size=2,  # Reduced for GPT-2-Large (36 layers)
        gradient_accumulation_steps=8,  # Effective batch size = 2 * 8 = 16
        learning_rate=2.5e-4,  # Lower learning rate for large model
        warmup_steps=1000,   # Gradual warmup
        weight_decay=0.01,   # Regularization
        max_grad_norm=1.0,   # Gradient clipping
        save_steps=5000,     # More frequent saves
        save_total_limit=3,  # Keep more checkpoints
        eval_strategy="steps",
        eval_steps=5000,
        logging_steps=100,   # More frequent logging
        prediction_loss_only=True,
        dataloader_drop_last=True,  # Avoid uneven batches
        # fp16=True,  # Mixed precision not supported on MPS
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
    )

    # --- Step 6: Start Training ---
    latest_ckpt = find_latest_checkpoint(MODEL_OUTPUT_PATH)
    if latest_ckpt:
        print(f"Resuming training from checkpoint: {latest_ckpt}")
    else:
        print("No checkpoint found. Starting fresh training...")

    print("Starting model training...")
    trainer.train(resume_from_checkpoint=latest_ckpt)
    print("Training complete.")

    # --- Step 7: Save the Final Model and Tokenizer ---
    print(f"Saving final model to {MODEL_OUTPUT_PATH}...")
    trainer.save_model(MODEL_OUTPUT_PATH)
    tokenizer.save_pretrained(MODEL_OUTPUT_PATH)
    print("Model and tokenizer saved successfully.")
    print(
        f"\nTo use the model, you can load it using: pipeline('text-generation', model='{MODEL_OUTPUT_PATH}')")


if __name__ == '__main__':
    main()
