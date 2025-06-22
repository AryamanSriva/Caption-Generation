import gradio as gr
from pathlib import Path
from config import DEVICE, EMB_DIM
from data_loading import tfms

def predict(img, model, tokenizer):
    tokens = [tokenizer.bos_token_id]
    img = tfms(img).unsqueeze(0)
    img_ftr = model.image_encoder(img).reshape(1, EMB_DIM, -1).permute(2, 0, 1)
    img_ftr = model.text_decoder.positional_encoding(img_ftr)

    c = 0
    while (
        tokens[-1] != tokenizer.eos_token_id if len(tokens) > 1 else True
    ) and c <= 50:
        pred = model.text_decoder.generate(img_ftr, torch.tensor(tokens).unsqueeze(0))[
            -1
        ]
        tokens.append(pred.argmax(-1).item())
        c += 1

    return (
        tokenizer.decode(tokens)
        .replace(tokenizer.bos_token, "")
        .replace(tokenizer.eos_token, "")
        .strip()
    )

def launch_gradio(model, tokenizer):
    image_dir = "/usercode/flickr-8k/images"
    examples = [
        Path(image_dir, "1377668044_36398401dd.jpg"),
        Path(image_dir, "2094323311_27d58b1513.jpg"),
        Path(image_dir, "299181827_8dc714101b.jpg")
    ]

    model = model.eval()
    with torch.no_grad():
        demo = gr.Interface(
            fn=lambda img: predict(img, model, tokenizer),
            inputs=gr.Image(type="pil"),
            outputs="text",
            examples=examples
        )
        demo.launch(share=True)