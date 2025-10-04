# Image Captioning AI

A comprehensive deep learning project that combines computer vision and natural language processing to generate descriptive captions for images. This project uses pre-trained ResNet50 for image feature extraction and LSTM networks for caption generation.

## Overview

This project implements an end-to-end image captioning system that:
- Extracts visual features from images using pre-trained ResNet50
- Generates natural language captions using LSTM networks
- Supports both greedy and beam search decoding
- Provides comprehensive evaluation metrics (BLEU, METEOR, ROUGE-L)

## Architecture

The system consists of several key components:

1. **Data Preprocessing Pipeline** (`data_preprocessing.py`)
   - Loads and cleans image captions
   - Creates tokenizers for text processing
   - Extracts ResNet50 features from images
   - Prepares training sequences

2. **Image Captioning Model** (`image_captioning_model.py`)
   - ResNet50 feature extraction (2048-dimensional vectors)
   - LSTM-based caption generation
   - Attention mechanism for combining image and text features
   - Support for both training and inference

3. **Training Script** (`train_model.py`)
   - Complete training pipeline
   - Automatic model checkpointing
   - Early stopping and learning rate scheduling
   - Progress tracking and logging

4. **Inference Script** (`generate_captions.py`)
   - Generate captions for new images
   - Interactive mode for testing
   - Batch processing capabilities
   - Command-line interface

5. **Evaluation Metrics** (`evaluate_model.py`)
   - BLEU score calculation
   - METEOR score computation
   - ROUGE-L evaluation
   - Comprehensive model assessment

## Installation

### Prerequisites

- Python 3.7+
- TensorFlow 2.8+
- CUDA-compatible GPU (recommended for training)

### Setup

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Place images in the `images/` directory
   - Create a captions file in `captions/captions.txt` with format: `image_filename,caption`

## Dataset Format

Your dataset should follow this structure:

```
project_root/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── captions/
│   └── captions.txt
└── [model files will be generated here]
```

The `captions.txt` file should contain one line per image-caption pair:
```
image1.jpg,A dog playing in the park
image1.jpg,The brown dog is running
image2.jpg,A cat sleeping on the couch
image2.jpg,The white cat is resting
```

Each image can have multiple captions (one per line).

## Usage

### Training the Model

Train the model on your dataset:

```bash
python train_model.py
```

This will:
1. Preprocess your data (if not already done)
2. Train the model with automatic checkpointing
3. Save the trained model and tokenizer
4. Test the model on sample images

Optional parameters:
```bash
python train_model.py --epochs 100 --batch-size 32 --max-length 50
```

### Generating Captions

#### Interactive Mode

Run the interactive caption generator:

```bash
python generate_captions.py --interactive
```

This opens an interactive session where you can:
- Enter image paths to generate captions
- Use 'batch' mode for multiple images
- Get real-time caption generation

#### Single Image

Generate a caption for a single image:

```bash
python generate_captions.py --image path/to/your/image.jpg
```

#### Multiple Images

Generate captions for multiple images:

```bash
python generate_captions.py --images path1.jpg path2.jpg path3.jpg
```

#### Advanced Options

```bash
python generate_captions.py --image test.jpg --beam-width 5 --max-length 30
```

### Evaluating the Model

Evaluate your trained model:

```bash
python evaluate_model.py
```

This calculates:
- BLEU scores (1-4 gram)
- METEOR scores
- ROUGE-L scores
- Statistical analysis

Optional parameters:
```bash
python evaluate_model.py --max-samples 200 --save-results evaluation_results.pkl
```

## Model Architecture Details

### Feature Extraction
- Uses pre-trained ResNet50 (ImageNet weights)
- Extracts 2048-dimensional feature vectors
- Global average pooling for fixed-size output

### Caption Generation
- LSTM-based sequence-to-sequence model
- Word embedding layer (256 dimensions)
- Two LSTM layers (512 units each)
- Dropout for regularization (0.5)
- Attention mechanism for image-text fusion

### Training Configuration
- Optimizer: Adam (learning rate: 0.001)
- Loss: Sparse Categorical Crossentropy
- Early stopping with patience of 5 epochs
- Learning rate reduction on plateau
- Model checkpointing for best validation loss

## Performance Metrics

The evaluation script provides comprehensive metrics:

### BLEU Score
- Measures n-gram overlap between generated and reference captions
- Ranges from 0 to 1 (higher is better)
- Calculates BLEU-1 through BLEU-4

### METEOR Score
- Considers synonyms and stemming
- Better correlation with human judgment
- Ranges from 0 to 1 (higher is better)

### ROUGE-L
- Based on longest common subsequence
- Measures both precision and recall
- Good for evaluating fluency

## Example Results

After training, you can expect captions like:

```
Generated Caption: "a dog playing in the grass"
Ground Truth: "a brown dog running in the park"

Generated Caption: "a person riding a bicycle"
Ground Truth: "a man cycling on a mountain trail"
```

## Tips for Better Results

1. **Data Quality**: Use high-quality, diverse captions
2. **Dataset Size**: Larger datasets generally produce better results
3. **Training Time**: Allow sufficient training epochs
4. **Hyperparameters**: Experiment with learning rates and model sizes
5. **Beam Search**: Use beam width of 3-5 for better captions

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or image resolution
2. **Poor Captions**: Increase training time or dataset size
3. **Slow Training**: Use GPU acceleration
4. **Tokenizer Errors**: Check caption file format

### Performance Optimization

1. Use GPU for training and inference
2. Pre-extract image features for faster training
3. Use smaller models for faster inference
4. Implement batch processing for multiple images

## Project Structure

```
image-captioning-ai/
├── data_preprocessing.py      # Data preprocessing pipeline
├── image_captioning_model.py  # Main model architecture
├── train_model.py            # Training script
├── generate_captions.py      # Inference script
├── evaluate_model.py         # Evaluation metrics
├── requirements.txt          # Dependencies
├── README.md               # This file
├── captions/               # Caption files
├── images/                 # Image dataset
├── preprocessed_data.pkl   # Generated: preprocessed data
├── image_captioning_model.h5  # Generated: trained model
├── tokenizer.pkl           # Generated: tokenizer
└── training_history.pkl    # Generated: training history
```

## Contributing

Feel free to contribute improvements:

1. Add new model architectures
2. Implement additional evaluation metrics
3. Improve preprocessing pipeline
4. Add support for different datasets
5. Optimize inference speed

## License

This project is open source and available under the MIT License.

## Acknowledgments

- ResNet50 pre-trained weights from ImageNet
- TensorFlow/Keras for deep learning framework
- Research papers on image captioning architectures

## References

1. "Show and Tell: A Neural Image Caption Generator" - Vinyals et al.
2. "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" - Xu et al.
3. "Deep Visual-Semantic Alignments for Generating Image Descriptions" - Karpathy et al.

---

For questions or issues, please check the troubleshooting section or create an issue in the repository.