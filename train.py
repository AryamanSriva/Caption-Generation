import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from config import *
from model import CaptionModel
from data_loading import CustomImageDataset, tfms
from utils import create_mask, generate_square_subsequent_mask

def initialize_model(tokenizer):
    model = CaptionModel(
        emb_size=EMB_DIM,
        nhead=NHEAD,
        num_decoder_layers=NUM_LAYERS,
        tgt_vocab_size=len(tokenizer),
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        activation=ACTIVATION,
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    optimizer_cnn = torch.optim.AdamW(
        model.image_encoder.parameters(), lr=CNN_LR
    )
    optimizer_transformer = torch.optim.AdamW(
        model.text_decoder.parameters(), lr=TRANSFORMER_LR
    )

    scheduler_cnn = CosineAnnealingLR(optimizer_cnn, T_max=NUM_EPOCHS)
    scheduler_transformer = CosineAnnealingLR(optimizer_transformer, T_max=NUM_EPOCHS)
    
    return model, criterion, optimizer_cnn, optimizer_transformer, scheduler_cnn, scheduler_transformer

def train_model(tokenizer):
    # Initialize datasets
    train_dataset = CustomImageDataset(
        root_dir="/usercode/flickr-8k", 
        data_split="train", 
        transform=tfms, 
        tokenizer=tokenizer
    )
    eval_dataset = CustomImageDataset(
        root_dir="/usercode/flickr-8k", 
        data_split="dev", 
        transform=tfms, 
        tokenizer=tokenizer
    )
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)
    
    model, criterion, optimizer_cnn, optimizer_transformer, scheduler_cnn, scheduler_transformer = initialize_model(tokenizer)
    
    writer = SummaryWriter(EXP_DIR)
    
    for e in range(NUM_EPOCHS):
        model.train()
        train_running_loss = 0.0
        i = 0

        for idx, data in enumerate(train_dataloader):
            imgs = data["image"].to(device=DEVICE)
            target = data["caption"].to(device=DEVICE)

            target_in = target[:, :-1]
            target_out = target[:, 1:]

            model.zero_grad(set_to_none=True)
            tgt_mask, tgt_padding_mask = create_mask(
                imgs, target_in, pad_idx=tokenizer.pad_token_id)

            logits = model(imgs, target_in, tgt_mask.to(
                DEVICE), tgt_padding_mask.to(DEVICE))

            T, B, D = logits.shape
            loss = criterion(
                logits.permute(1, 0, 2).reshape(T*B, D), target_out.reshape(T*B))
            writer.add_scalar("loss/train", loss, (e * NUM_EPOCHS + idx))

            loss.backward()
            optimizer_cnn.step()
            optimizer_transformer.step()

            train_running_loss += loss.item()
            i += 1

        scheduler_cnn.step()
        scheduler_transformer.step()

        train_epoch_loss = train_running_loss / i
        writer.add_scalar("epoch_loss/train", train_epoch_loss, e)
        print(f"train loss after epoch {e+1}: {train_epoch_loss}")

        if (e + 1) % CHECKPOINT_EPOCH == 0:
            model.eval()
            eval_running_loss = 0.0
            i = 0

            with torch.no_grad():
                for idx, data in enumerate(eval_dataloader):
                    imgs = data["image"].to(device=DEVICE)
                    target = data["caption"].to(device=DEVICE)

                    target_in = target[:, :-1]
                    target_out = target[:, 1:]

                    tgt_mask, tgt_padding_mask = create_mask(
                        imgs, target_in, pad_idx=tokenizer.pad_token_id)

                    logits = model(imgs, target_in, tgt_mask.to(
                        DEVICE), tgt_padding_mask.to(DEVICE))

                    T, B, D = logits.shape
                    loss = criterion(
                        logits.permute(1, 0, 2).reshape(T*B, D), target_out.reshape(T*B))

                    eval_running_loss += loss.item()
                    i += 1

                eval_epoch_loss = eval_running_loss / i
                writer.add_scalar("epoch_loss/eval", eval_epoch_loss, e)
                print(f"eval loss after epoch {e+1}: {eval_epoch_loss}")

            torch.save(model.state_dict(),
                       f"{EXP_DIR}/model_epoch_{e}.pt")
    
    return model