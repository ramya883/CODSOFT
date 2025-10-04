import os
import pickle
import numpy as np
from data_preprocessing import DataPreprocessor
from image_captioning_model import ImageCaptioningModel, CaptionGenerator

def main():
    print("=== Image Captioning Model Training ===")
    
    # CLI overrides
    import argparse
    parser = argparse.ArgumentParser(description='Train Image Captioning Model')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--lstm-units', type=int, default=512)
    parser.add_argument('--max-length', type=int, default=40)
    parser.add_argument('--vocab-size', type=int, default=10000)
    parser.add_argument('--max-images', type=int, default=None, help='Limit number of images for quick runs')
    parser.add_argument('--preprocess-fresh', action='store_true', help='Ignore existing preprocessed_data.pkl and rebuild')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'captions_file': 'captions/captions.txt',
        'images_dir': 'images',
        'preprocessed_data_path': 'preprocessed_data.pkl',
        'model_save_path': 'image_captioning_model.h5',
        'tokenizer_save_path': 'tokenizer.pkl',
        'max_length': args.max_length,
        'vocab_size': args.vocab_size,
        'embedding_dim': args.embedding_dim,
        'lstm_units': args.lstm_units,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
        'max_images': args.max_images
    }
    
    # Check if preprocessed data exists
    if os.path.exists(config['preprocessed_data_path']) and not args.preprocess_fresh:
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
            vocab_size=config['vocab_size']
        )
        
        # Preprocess data
        data_dict = preprocessor.preprocess_data(
            test_size=config['test_split'] + config['val_split'],
            val_size=config['val_split'],
            max_images=config['max_images']
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
    print("\nBuilding model...")
    model = ImageCaptioningModel(
        vocab_size=vocab_size,
        max_length=max_length,
        embedding_dim=config['embedding_dim'],
        lstm_units=config['lstm_units']
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
    caption_generator = CaptionGenerator(
        model=model.model,
        tokenizer=tokenizer,
        max_length=max_length,
        image_features_dict=data_dict['image_features']
    )
    
    # Test on a few training examples
    test_images = list(data_dict['image_features'].keys())[:5]
    
    print("\nGenerated captions (Training examples):")
    for img_name in test_images:
        try:
            generated_caption = caption_generator.generate_caption(img_name, beam_width=3)
            print(f"{img_name}: {generated_caption}")
        except Exception as e:
            print(f"Error generating caption for {img_name}: {e}")
    
    # Save training history
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("\nTraining history saved to training_history.pkl")
    
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()