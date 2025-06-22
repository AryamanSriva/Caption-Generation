from transformers import GPT2Tokenizer
from train import train_model
from evaluate import evaluate_model
from app import launch_gradio
from model import CaptionModel
from config import *

def main():
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens(
        {"bos_token": "<|startoftext|>", "unk_token": "<|unk|>", "pad_token": "[PAD]"}
    )
    
    # Train the model
    model = train_model(tokenizer)
    
    # Evaluate the model
    evaluate_model(model, tokenizer)
    
    # Launch Gradio app
    launch_gradio(model, tokenizer)

if __name__ == "__main__":
    main()