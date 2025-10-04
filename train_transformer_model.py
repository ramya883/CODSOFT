import os
import pickle
import numpy as np
from data_preprocessing import DataPreprocessor
from transformer_captioning_model import TransformerCaptioningModel, TransformerCaptionGenerator

def main():
    print("=== Transformer Image Captioning Model Training ===")
    
    # Configuration
    config = {
        'captions_file': 'captions/captions.txt',
        'images_dir': 'images',
        'preprocessed_data_path': 'preprocessed_data.pkl',
        'model_save_path': 'transformer_captioning_model.h5',
        'tokenizer_save_path': 'tokenizer.pkl',
        'max_length': 40,
        'vocab_size': 10000,
        'embedding_dim': 256,
        'num_heads': 8,
        'ff_dim': 512,
        'num_transformer_blocks': 4,
        'batch_size': 64,
        'epochs': 50,
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
        'cnn_model_type': 'resnet50'  # or 'vgg16'
    }
    
    # Check if preprocessed data exists
    if os.path.exists(config['preprocessed_data_path']):
        print("Loading preprocessed data...")
        with open(config['preprocessed_data_path'], 'rb') as f:
            data_dict = pickle.load(f)
        print("Preprocessed data loaded successfully!")
    else:
        print("Preprocessing data...")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(
            captions_file=config['captions_file'],
            images_folder=config['images_dir'],
            max_caption_length=config['max_length'],
            vocab_size=config['vocab_size'],
            cnn_model_type=config['cnn_model_type']
        )
        
        # Preprocess data
        data_dict = preprocessor.preprocess_data(
            test_size=config['test_split'] + config['val_split'],
            val_size=config['val_split']
        )
        
        print("Data preprocessing completed!")
    
    # Extract key information
    tokenizer = data_dict['tokenizer']
    vocab_size = len(tokenizer.word_index) + 1
    max_length = data_dict['max_length']
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Maximum sequence length: {max_length}")
    print(f"Training samples: {len(data_dict['train'][0])}")
    print(f"Validation samples: {len(data_dict['val'][0]) if data_dict['val'][0] is not None else 0}")
    print(f"Test samples: {len(data_dict['test'][0]) if data_dict['test'][0] is not None else 0}")
    
    # Initialize model
    print("\nBuilding transformer model...")
    model = TransformerCaptioningModel(
        vocab_size=vocab_size,
        max_length=max_length,
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        num_transformer_blocks=config['num_transformer_blocks']
    )
    
    # Build and display model summary
    model.build_model()
    model.get_model_summary()
    
    # Train model
    print("\nStarting training...")
    history = model.train(
        data_dict=data_dict,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        model_save_path=config['model_save_path']
    )
    
    # Save tokenizer
    model.tokenizer = tokenizer
    model.save_model(
        model_path=config['model_save_path'],
        tokenizer_path=config['tokenizer_save_path']
    )
    
    print("\nTraining completed!")
    print(f"Model saved to: {config['model_save_path']}")
    print(f"Tokenizer saved to: {config['tokenizer_save_path']}")
    
    # Test the model with a few examples
    print("\nTesting the model...")
    
    # Create caption generator
    generator = TransformerCaptionGenerator(
        model=model.model,
        tokenizer=tokenizer,
        max_length=max_length,
        image_features_dict=data_dict['image_features']
    )
    
    # Get some test images
    test_images = list(data_dict['image_features'].keys())[:5]
    
    for img_name in test_images:
        caption = generator.generate_caption(img_name)
        print(f"Image: {img_name}")
        print(f"Generated caption: {caption}")
        print()

if __name__ == "__main__":
    main()