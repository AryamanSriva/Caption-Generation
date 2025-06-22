from collections import defaultdict
from torch.utils.data import DataLoader
from torchmetrics.text import ROUGEScore
from config import DEVICE, EMB_DIM
from data_loading import CustomImageDataset, tfms

def eval_collate(batch_data):
    images = batch_data[0]["image"].unsqueeze(0)
    caption = batch_data[0]["caption"]
    image_name = batch_data[0]["image_name"]
    return {"image": images, "caption": caption, "image_name": image_name}

def evaluate_model(model, tokenizer):
    eval_dataset = CustomImageDataset(
        root_dir="/usercode/flickr-8k",
        data_split="dev",
        transform=tfms,
        tokenizer=tokenizer,
        phase="test",
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=eval_collate)

    pred_results = defaultdict(list)
    model = model.eval()

    with torch.no_grad():
        for idx, data in enumerate(eval_dataloader):
            imgs = data["image"].to(DEVICE)
            references = data["caption"]
            image_name = data["image_name"]

            img_ftr = model.image_encoder(imgs).reshape(1, EMB_DIM, -1).permute(2, 0, 1)

            pred_tokens = [tokenizer.bos_token_id]

            while (
                pred_tokens[-1] != tokenizer.eos_token_id
                if len(pred_tokens) > 1
                else True
            ) and len(pred_tokens) <= 50:
                cur_tokens = torch.tensor(pred_tokens).unsqueeze(0).to(DEVICE)
                pred = model.text_decoder.generate(img_ftr, cur_tokens)[-1]
                pred_tokens.append(pred.detach().cpu().argmax(-1).item())

            pred_caption = tokenizer.decode(pred_tokens)

            pred_results["image_name"].append(image_name)
            pred_results["prediction"].append(pred_caption)
            pred_results["references"].append(references)

    predictions = [x.replace("<|startoftext|>", "").replace("<|endoftext|>", "") for x in pred_results["prediction"]]
    references = pred_results["references"]

    rouge_score = ROUGEScore()
    rouge_score = rouge_score(predictions, references)["rougeL_fmeasure"]
    print(f"ROUGE-L F1 Score: {rouge_score}")
    
    return pred_results