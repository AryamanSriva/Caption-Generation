# Image Captioning

A deep learning project that generates captions for images using a ConvNeXt encoder and Transformer decoder architecture, trained on the Flickr8k dataset.

## Features

- **State-of-the-art Architecture**: Combines ConvNeXt (visual encoder) with Transformer decoder for caption generation
- **Flickr8k Dataset**: Trained on 8,000 images with 40,000 captions
- **Custom Transformations**: Includes resize-pad transformation to maintain aspect ratios
- **Gradio Interface**: Interactive web interface for real-time caption generation
- **Comprehensive Evaluation**: Uses ROUGE-L score for performance assessment


## Dataset Setup

1. Download the Flickr8k dataset:
   ```bash
   wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
   wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
   ```

2. Extract and organize the dataset:
   ```
   flickr-8k/
   ├── images/
   │   ├── 1000268201_693b08cb0e.jpg
   │   └── ...
   └── text/
       ├── Flickr8k.token.txt
       ├── Flickr_8k.trainImages.txt
       ├── Flickr_8k.devImages.txt
       └── Flickr_8k.testImages.txt
   ```

## Model Architecture

### Visual Encoder (ConvNeXt)
- **Base Model**: ConvNeXt-Small pretrained on ImageNet
- **Feature Extraction**: Removes classification layers, outputs 768-dimensional features
- **Input Size**: 224×224 RGB images

### Text Decoder (Transformer)
- **Architecture**: 6-layer Transformer decoder
- **Embedding Dimension**: 768
- **Attention Heads**: 8
- **Vocabulary**: GPT-2 tokenizer with special tokens
- **Positional Encoding**: Sinusoidal encoding for sequence modeling


## Training

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 8 |
| Epochs | 35 |
| CNN Learning Rate | 1e-5 |
| Transformer Learning Rate | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Loss Function | CrossEntropyLoss |
| Max Caption Length | 50 tokens |

### Training Process

```bash
python train.py
```

The training script includes:
- Automatic mixed precision support
- Gradient clipping for stability
- Learning rate scheduling
- Checkpoint saving every 5 epochs
- TensorBoard logging

### Monitoring

View training progress with TensorBoard:
```bash
tensorboard --logdir caption_generation_experimentation
```

## Evaluation

The model is evaluated using:
- **ROUGE-L Score**: Measures longest common subsequence
- **BLEU Score**: N-gram precision metric
- **Qualitative Assessment**: Visual inspection of generated captions

### Running Evaluation

```python
# Load pretrained model
model.load_state_dict(torch.load('model_checkpoint.pt'))

# Generate captions for test set
python evaluate.py
```

## Usage

### Interactive Interface

Launch the Gradio interface:

```python
python app.py
```

This creates a web interface where you can:
- Upload images
- Get real-time caption generation
- View example results

### Programmatic Usage

```python
from model import CaptionModel
from utils import predict

# Load model
model = CaptionModel(...)
model.load_state_dict(torch.load('checkpoint.pt'))

# Generate caption
caption = predict(image)
print(f"Generated caption: {caption}")
```

## Future Improvements

- Implement beam search for better caption quality
- Add attention visualization
- Support for multiple languages
- Fine-tune on domain-specific datasets
- Implement CLIP-based evaluation metrics
- Add data augmentation techniques


## References

- [ConvNeXt: A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Flickr8k Dataset](https://forms.illinois.edu/sec/1713398)
- [Show, Attend and Tell](https://arxiv.org/abs/1502.03044)


## Acknowledgments

- Flickr8k dataset
- PyTorch
- Hugging Face
- Gradio
