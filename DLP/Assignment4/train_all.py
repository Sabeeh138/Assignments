import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.getcwd())

from a5_helper import load_coco_captions, train_captioner, get_toy_data
from a5_helper import train as train_transformer
from rnn_lstm_captioning import CaptioningRNN
from transformers import Transformer, CrossEntropyLoss, position_encoding_sinusoid, prepocess_input_sequence, generate_token_dict, AddSubDataset

# Logger to write to both stdout and a file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def train_part1(device):
    print("\n" + "="*50)
    print("STARTING PART 1: IMAGE CAPTIONING (ATTENTION LSTM)")
    print("="*50)
    
    dataset_path = "./datasets/coco.pt"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    # Check if file is still growing or wait for full size
    # Based on notebook, expected size is ~378MB (396,555,616 bytes)
    expected_size = 396555616
    current_size = os.path.getsize(dataset_path)
    if current_size < expected_size:
        print(f"Waiting for {dataset_path} to complete download ({current_size}/{expected_size})...")
        while current_size < expected_size:
            time.sleep(10)
            current_size = os.path.getsize(dataset_path)
            print(f"Progress: {current_size}/{expected_size}")

    data_dict = load_coco_captions(path=dataset_path)
    num_train = data_dict["train_images"].size(0)
    NULL_index = data_dict["vocab"]["token_to_idx"]["<NULL>"]

    # Training parameters
    BATCH_SIZE = 256
    learning_rate = 1e-3
    num_epochs = 10

    model = CaptioningRNN(
        cell_type="attn",
        word_to_idx=data_dict["vocab"]["token_to_idx"],
        input_dim=400,
        hidden_dim=512,
        wordvec_dim=256,
        ignore_index=NULL_index,
    ).to(device)

    print(f"Training Attention LSTM on {num_train} images for {num_epochs} epochs...")
    
    train_image_data = data_dict["train_images"]
    train_caption_data = data_dict["train_captions"]

    attn_model, loss_history = train_captioner(
        model,
        train_image_data,
        train_caption_data,
        num_epochs=num_epochs,
        batch_size=BATCH_SIZE,
        learning_rate=learning_rate,
        device=device,
    )
    
    torch.save(attn_model.state_dict(), "attn_lstm_model.pt")
    print("Part 1 training complete. Model saved to attn_lstm_model.pt")

def train_part2(device):
    print("\n" + "="*50)
    print("STARTING PART 2: TRANSFORMER ARITHMETIC")
    print("="*50)

    if not os.path.exists("two_digit_op.json"):
        print("Error: two_digit_op.json not found.")
        return

    data = get_toy_data("two_digit_op.json")
    
    # Split data (use 80/20 split)
    total_len = len(data["inp_expression"])
    train_len = int(0.8 * total_len)
    
    X_train, X_test = data["inp_expression"][:train_len], data["inp_expression"][train_len:]
    y_train, y_test = data["out_expression"][:train_len], data["out_expression"][train_len:]

    SPECIAL_TOKENS = ["POSITIVE", "NEGATIVE", "add", "subtract", "BOS", "EOS"]
    vocab = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] + SPECIAL_TOKENS
    convert_str_to_tokens = generate_token_dict(vocab)

    num_heads = 4
    emb_dim = 32
    dim_feedforward = 32
    dropout = 0.2
    num_enc_layers = 4
    num_dec_layers = 4
    vocab_len = len(vocab)
    BATCH_SIZE = 32
    num_epochs = 10
    lr = 1e-3

    model = Transformer(
        num_heads,
        emb_dim,
        dim_feedforward,
        dropout,
        num_enc_layers,
        num_dec_layers,
        vocab_len,
    ).to(device)

    train_data = AddSubDataset(
        X_train,
        y_train,
        convert_str_to_tokens,
        SPECIAL_TOKENS,
        emb_dim,
        position_encoding_sinusoid,
    )
    valid_data = AddSubDataset(
        X_test,
        y_test,
        convert_str_to_tokens,
        SPECIAL_TOKENS,
        emb_dim,
        position_encoding_sinusoid,
    )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    print(f"Training Transformer on {len(X_train)} expressions for {num_epochs} epochs...")

    trained_model = train_transformer(
        model,
        train_loader,
        valid_loader,
        CrossEntropyLoss,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=BATCH_SIZE,
        warmup_interval=None,
        device=device,
    )

    torch.save(trained_model.state_dict(), "transformer_model.pt")
    print("Part 2 training complete. Model saved to transformer_model.pt")

if __name__ == "__main__":
    # Re-open in append mode if needed, or overwrite for new run
    sys.stdout = Logger("training_log.txt")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # try:
    #     train_part2(device)
    # except Exception as e:
    #     print(f"Error in Part 2 training: {e}")

    try:
        train_part1(device)
    except Exception as e:
        print(f"Error in Part 1 training: {e}")

    print("\nALL TRAINING COMPLETED.")
