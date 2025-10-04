import os
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from image_captioning_model import ImageCaptioningModel, CaptionGenerator

def get_model_max_length(model):
    """Extract max_length from model architecture"""
    try:
        # Try to get from input shape of the second input (text sequences)
        input_shape = model.inputs[1].shape
        if len(input_shape) >= 2:
            return int(input_shape[1])
    except:
        pass
    
    # Default fallback
    return 20

def load_model_and_tokenizer(model_path='image_captioning_model.h5', tokenizer_path='tokenizer.pkl'):
    """Load the trained model and tokenizer"""
    print("Loading model and tokenizer...")
    
    # Create model instance with default parameters (will be overridden by loaded model)
    model_instance = ImageCaptioningModel(vocab_size=10000, max_length=40)
    
    # Load the trained model using the model instance's load method
    model_instance.load_model(model_path, tokenizer_path)
    model = model_instance.model
    tokenizer = model_instance.tokenizer if hasattr(model_instance, 'tokenizer') else None
    
    # Get max_length from the loaded model architecture
    max_length = get_model_max_length(model)
    
    return model, tokenizer, max_length

def extract_features_from_image(image_path, target_size=(224, 224)):
    """Extract features from a single image using ResNet50"""
    # Load and preprocess image
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extract features using ResNet50
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    features = model.predict(img_array, verbose=0)
    
    return features[0]  # Return single feature vector

def generate_caption_for_image(image_path, model, tokenizer, max_length=20, beam_width=3):
    """Generate caption for a single image"""
    # Extract features from image
    image_features = extract_features_from_image(image_path)
    
    # Create temporary feature dictionary
    image_features_dict = {'temp_image': image_features}
    
    # Create caption generator
    caption_generator = CaptionGenerator(
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        image_features_dict=image_features_dict
    )
    
    # Generate caption
    caption = caption_generator.generate_caption('temp_image', beam_width=beam_width)
    
    return caption

def batch_generate_captions(image_paths, model, tokenizer, max_length=40, beam_width=3):
    """Generate captions for multiple images"""
    captions = []
    
    # Extract features for all images
    print("Extracting features from images...")
    image_features_dict = {}
    
    for i, image_path in enumerate(image_paths):
        if os.path.exists(image_path):
            features = extract_features_from_image(image_path)
            image_features_dict[os.path.basename(image_path)] = features
            print(f"Processed {i+1}/{len(image_paths)} images")
        else:
            print(f"Warning: Image not found - {image_path}")
    
    # Create caption generator
    caption_generator = CaptionGenerator(
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        image_features_dict=image_features_dict
    )
    
    # Generate captions
    print("Generating captions...")
    for image_path in image_paths:
        if os.path.exists(image_path):
            image_name = os.path.basename(image_path)
            caption = caption_generator.generate_caption(image_name, beam_width=beam_width)
            captions.append({
                'image': image_path,
                'caption': caption
            })
    
    return captions

def interactive_caption_generator(model, tokenizer, max_length=20):
    """Interactive mode for generating captions"""
    print("=== Interactive Image Caption Generator ===")
    print("Model loaded successfully!")
    print("Enter 'quit' to exit")
    print("Enter 'batch' to process multiple images")
    print("Enter image path to generate caption")
    
    if model is None or tokenizer is None:
        print("Error: Could not load model or tokenizer. Please train the model first using train_model.py")
        return
    
    print("Model loaded successfully!")
    print("Enter 'quit' to exit")
    print("Enter 'batch' to process multiple images")
    print("Enter image path to generate caption")
    
    while True:
        user_input = input("\nEnter image path or command: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'batch':
            # Batch processing
            image_paths = []
            print("Enter image paths (one per line, empty line to finish):")
            while True:
                path = input().strip()
                if path == '':
                    break
                image_paths.append(path)
            
            if image_paths:
                captions = batch_generate_captions(image_paths, model, tokenizer)
                print("\nGenerated Captions:")
                for item in captions:
                    print(f"{item['image']}: {item['caption']}")
        else:
            # Single image processing
            if os.path.exists(user_input):
                try:
                    caption = generate_caption_for_image(user_input, model, tokenizer)
                    print(f"Generated Caption: {caption}")
                except Exception as e:
                    print(f"Error generating caption: {e}")
            else:
                print(f"Error: Image not found - {user_input}")

def main():
    """Main function with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate captions for images')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--images', type=str, nargs='+', help='Paths to multiple images')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--model', type=str, default='image_captioning_model.h5', help='Path to model file')
    parser.add_argument('--tokenizer', type=str, default='tokenizer.pkl', help='Path to tokenizer file')
    parser.add_argument('--beam-width', type=int, default=3, help='Beam width for caption generation')
    parser.add_argument('--max-length', type=int, default=40, help='Maximum caption length')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_result = load_model_and_tokenizer(args.model, args.tokenizer)
    
    if len(model_result) == 3:
        model, tokenizer, max_length = model_result
    else:
        model, tokenizer = model_result
        max_length = args.max_length
    
    if model is None or tokenizer is None:
        print("Error: Could not load model or tokenizer. Please train the model first using train_model.py")
        return
    
    print("Model loaded successfully!")
    print(f"Using max_length: {max_length}")
    
    # Handle different modes
    if args.interactive:
        interactive_caption_generator(model, tokenizer, max_length)
    elif args.image:
        # Single image
        if os.path.exists(args.image):
            caption = generate_caption_for_image(
                args.image, model, tokenizer, 
                max_length=max_length, beam_width=args.beam_width
            )
            print(f"Generated Caption: {caption}")
        else:
            print(f"Error: Image not found - {args.image}")
    elif args.images:
        # Multiple images
        captions = batch_generate_captions(
            args.images, model, tokenizer,
            max_length=max_length, beam_width=args.beam_width
        )
        print("\nGenerated Captions:")
        for item in captions:
            print(f"{item['image']}: {item['caption']}")
    else:
        # Default to interactive mode
        print("No images specified. Running in interactive mode...")
        interactive_caption_generator(model, tokenizer, max_length)

if __name__ == "__main__":
    main()