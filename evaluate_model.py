import numpy as np
from collections import Counter
import pickle
import os
from image_captioning_model import ImageCaptioningModel, CaptionGenerator

def n_grams(tokens, n):
    """Generate n-grams from a list of tokens"""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def bleu_score(candidate, references, max_n=4):
    """
    Calculate BLEU score for a candidate caption against multiple references
    
    Args:
        candidate: Generated caption string
        references: List of reference caption strings
        max_n: Maximum n-gram order to consider
    
    Returns:
        BLEU score (float)
    """
    # Tokenize
    candidate_tokens = candidate.lower().split()
    reference_tokens = [ref.lower().split() for ref in references]
    
    # Calculate precision for each n-gram order
    precisions = []
    
    for n in range(1, max_n + 1):
        # Get n-grams for candidate
        candidate_ngrams = n_grams(candidate_tokens, n)
        
        # Get n-grams for all references
        reference_ngrams = []
        for ref_tokens in reference_tokens:
            reference_ngrams.extend(n_grams(ref_tokens, n))
        
        # Count n-grams
        candidate_counts = Counter(candidate_ngrams)
        reference_counts = Counter(reference_ngrams)
        
        # Calculate clipped precision
        clipped_matches = 0
        for ngram, count in candidate_counts.items():
            clipped_matches += min(count, reference_counts.get(ngram, 0))
        
        # Calculate precision
        if len(candidate_ngrams) > 0:
            precision = clipped_matches / len(candidate_ngrams)
        else:
            precision = 0.0
        
        precisions.append(precision)
    
    # Calculate brevity penalty
    candidate_length = len(candidate_tokens)
    reference_lengths = [len(ref_tokens) for ref_tokens in reference_tokens]
    
    if candidate_length == 0:
        return 0.0
    
    # Find closest reference length
    closest_ref_length = min(reference_lengths, key=lambda x: abs(x - candidate_length))
    
    if candidate_length > closest_ref_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = np.exp(1 - closest_ref_length / candidate_length)
    
    # Calculate geometric mean of precisions
    if any(p == 0 for p in precisions):
        bleu = 0.0
    else:
        geometric_mean = np.exp(np.mean(np.log(precisions)))
        bleu = brevity_penalty * geometric_mean
    
    return bleu

def meteor_score(candidate, references, alpha=0.9, beta=3.0, gamma=0.5):
    """
    Simplified METEOR score calculation
    
    Args:
        candidate: Generated caption string
        references: List of reference caption strings
        alpha: Parameter for harmonic mean
        beta: Parameter for penalty
        gamma: Parameter for fragmentation penalty
    
    Returns:
        METEOR score (float)
    """
    def word_matches(candidate_tokens, reference_tokens):
        """Count word matches between candidate and reference"""
        candidate_counts = Counter(candidate_tokens)
        reference_counts = Counter(reference_tokens)
        
        matches = 0
        for word in candidate_counts:
            matches += min(candidate_counts[word], reference_counts.get(word, 0))
        
        return matches
    
    # Tokenize
    candidate_tokens = candidate.lower().split()
    reference_tokens_list = [ref.lower().split() for ref in references]
    
    if not candidate_tokens:
        return 0.0
    
    # Find best matching reference
    best_matches = 0
    best_reference = None
    
    for ref_tokens in reference_tokens_list:
        matches = word_matches(candidate_tokens, ref_tokens)
        if matches > best_matches:
            best_matches = matches
            best_reference = ref_tokens
    
    if best_reference is None:
        return 0.0
    
    # Calculate precision and recall
    precision = best_matches / len(candidate_tokens)
    recall = best_matches / len(best_reference)
    
    if precision + recall == 0:
        return 0.0
    
    # Calculate F-mean
    f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
    
    # Calculate penalty (simplified)
    penalty = beta * (len(candidate_tokens) - best_matches) / len(candidate_tokens)
    penalty = 1 - penalty / (penalty + 1)
    
    # Calculate fragmentation penalty (simplified)
    frag_penalty = gamma * (len(candidate_tokens) - best_matches) / len(candidate_tokens)
    frag_penalty = 1 - frag_penalty
    
    meteor = f_mean * penalty * frag_penalty
    
    return meteor

def rouge_l_score(candidate, references):
    """
    Calculate ROUGE-L score based on longest common subsequence
    
    Args:
        candidate: Generated caption string
        references: List of reference caption strings
    
    Returns:
        ROUGE-L score (float)
    """
    def lcs_length(seq1, seq2):
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    # Tokenize
    candidate_tokens = candidate.lower().split()
    reference_tokens_list = [ref.lower().split() for ref in references]
    
    if not candidate_tokens:
        return 0.0
    
    # Calculate LCS for each reference
    lcs_scores = []
    for ref_tokens in reference_tokens_list:
        lcs_len = lcs_length(candidate_tokens, ref_tokens)
        
        if len(ref_tokens) > 0:
            recall = lcs_len / len(ref_tokens)
        else:
            recall = 0.0
            
        if len(candidate_tokens) > 0:
            precision = lcs_len / len(candidate_tokens)
        else:
            precision = 0.0
        
        if precision + recall > 0:
            f_measure = 2 * precision * recall / (precision + recall)
        else:
            f_measure = 0.0
        
        lcs_scores.append(f_measure)
    
    # Return best score
    return max(lcs_scores) if lcs_scores else 0.0

def evaluate_model(model_path, tokenizer_path, test_data_path, max_samples=100):
    """
    Evaluate the trained model on test data
    
    Args:
        model_path: Path to trained model
        tokenizer_path: Path to tokenizer
        test_data_path: Path to test data
        max_samples: Maximum number of samples to evaluate
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("Loading model and test data...")
    
    # Load model and tokenizer
    model_instance = ImageCaptioningModel(vocab_size=10000, max_length=40)
    model = model_instance.load_model(model_path, tokenizer_path)
    
    # Load test data
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    
    # Extract test data
    X_test_seq, y_test, img_test = test_data['test']
    image_features = test_data['image_features']
    tokenizer = test_data['tokenizer']
    
    # Create caption generator
    caption_generator = CaptionGenerator(
        model=model,
        tokenizer=tokenizer,
        max_length=test_data['max_length'],
        image_features_dict=image_features
    )
    
    # Get ground truth captions
    captions_dict = test_data['captions_dict']
    
    # Evaluate on subset of test data
    bleu_scores = []
    meteor_scores = []
    rouge_scores = []
    
    print(f"Evaluating on {min(max_samples, len(img_test))} test samples...")
    
    for i, img_name in enumerate(img_test[:max_samples]):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{min(max_samples, len(img_test))}")
        
        try:
            # Generate caption (beam search)
            generated_caption = caption_generator.generate_caption(img_name)
            
            # Get ground truth captions
            ground_truth_captions = captions_dict.get(img_name, [])
            
            if not ground_truth_captions:
                continue
            
            # Calculate metrics
            bleu = bleu_score(generated_caption, ground_truth_captions)
            meteor = meteor_score(generated_caption, ground_truth_captions)
            rouge = rouge_l_score(generated_caption, ground_truth_captions)
            
            bleu_scores.append(bleu)
            meteor_scores.append(meteor)
            rouge_scores.append(rouge)
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue
    
    # Calculate average scores
    results = {
        'bleu_score': np.mean(bleu_scores) if bleu_scores else 0.0,
        'meteor_score': np.mean(meteor_scores) if meteor_scores else 0.0,
        'rouge_l_score': np.mean(rouge_scores) if rouge_scores else 0.0,
        'num_samples': len(bleu_scores),
        'bleu_scores': bleu_scores,
        'meteor_scores': meteor_scores,
        'rouge_scores': rouge_scores
    }
    
    return results

def print_evaluation_results(results):
    """Print evaluation results in a formatted way"""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Number of samples evaluated: {results['num_samples']}")
    print(f"BLEU Score: {results['bleu_score']:.4f}")
    print(f"METEOR Score: {results['meteor_score']:.4f}")
    print(f"ROUGE-L Score: {results['rouge_l_score']:.4f}")
    print("="*50)
    
    # Print score distributions
    if results['bleu_scores']:
        print(f"\nBLEU Score Statistics:")
        print(f"  Min: {min(results['bleu_scores']):.4f}")
        print(f"  Max: {max(results['bleu_scores']):.4f}")
        print(f"  Std: {np.std(results['bleu_scores']):.4f}")
    
    if results['meteor_scores']:
        print(f"\nMETEOR Score Statistics:")
        print(f"  Min: {min(results['meteor_scores']):.4f}")
        print(f"  Max: {max(results['meteor_scores']):.4f}")
        print(f"  Std: {np.std(results['meteor_scores']):.4f}")
    
    if results['rouge_scores']:
        print(f"\nROUGE-L Score Statistics:")
        print(f"  Min: {min(results['rouge_scores']):.4f}")
        print(f"  Max: {max(results['rouge_scores']):.4f}")
        print(f"  Std: {np.std(results['rouge_scores']):.4f}")

def main():
    """Main function for evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate image captioning model')
    parser.add_argument('--model', type=str, default='image_captioning_model.h5', help='Path to model file')
    parser.add_argument('--tokenizer', type=str, default='tokenizer.pkl', help='Path to tokenizer file')
    parser.add_argument('--data', type=str, default='preprocessed_data.pkl', help='Path to preprocessed data')
    parser.add_argument('--max-samples', type=int, default=100, help='Maximum number of samples to evaluate')
    parser.add_argument('--save-results', type=str, help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found - {args.model}")
        return
    
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer file not found - {args.tokenizer}")
        return
    
    if not os.path.exists(args.data):
        print(f"Error: Data file not found - {args.data}")
        return
    
    # Evaluate model
    results = evaluate_model(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        test_data_path=args.data,
        max_samples=args.max_samples
    )
    
    # Print results
    print_evaluation_results(results)
    
    # Save results if requested
    if args.save_results:
        with open(args.save_results, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {args.save_results}")

if __name__ == "__main__":
    main()