import stanza
import collections
import math
import re
import sys

def analyze_collocations():
    # 1. Setup Stanza
    print("Downloading/Loading Stanza model for Ancient Greek...")
    try:
        stanza.download('grc')
    except Exception as e:
        print(f"Model download warning (might already be present): {e}")
    
    # Initialize pipeline
    # use_gpu=False is safer for some environments, set to True if GPU is available
    nlp = stanza.Pipeline('grc', processors='tokenize,lemma,pos', use_gpu=False, verbose=False)

    # 2. Read and Clean Corpus
    print("Reading corpus...")
    corpus_path = 'homer.iliad.tess'
    clean_lines = []
    
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Format: <hom. il. 1.1>\tTEXT...
                # We want to remove the tag and the tab
                # Regex to remove <...> and whitespace at start
                cleaned = re.sub(r'<[^>]+>\s*', '', line).strip()
                if cleaned:
                    clean_lines.append(cleaned)
    except FileNotFoundError:
        print(f"Error: {corpus_path} not found.")
        return

    print(f"Read {len(clean_lines)} lines.")

    # 3. Process Text and Extract Lemmas
    print("Processing text (this may take a minute)...")
    
    all_lemmas = []
    
    # Process in chunks to show progress and manage memory
    chunk_size = 100
    total_chunks = (len(clean_lines) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(clean_lines), chunk_size):
        chunk = clean_lines[i:i+chunk_size]
        text_chunk = " ".join(chunk)
        
        # Run NLP pipeline
        doc = nlp(text_chunk)
        
        for sentence in doc.sentences:
            for word in sentence.words:
                # Filter out punctuation and numbers, keep only words
                if word.upos not in ['PUNCT', 'SYM', 'NUM'] and word.lemma:
                    all_lemmas.append(word.lemma)
        
        if (i // chunk_size) % 10 == 0:
            print(f"Processed chunk {i // chunk_size + 1}/{total_chunks}")

    print(f"Total lemmas extracted: {len(all_lemmas)}")

    # 4. Count Frequencies and Co-occurrences
    print("Calculating statistics...")
    
    lemma_counts = collections.Counter(all_lemmas)
    pair_counts = collections.Counter()
    
    window_size = 5
    total_pairs = 0
    
    # Sliding window
    # We look at pairs (word_i, word_j) where j > i and j < i + window_size
    for i in range(len(all_lemmas)):
        current_word = all_lemmas[i]
        
        end_window = min(i + window_size, len(all_lemmas))
        for j in range(i + 1, end_window):
            next_word = all_lemmas[j]
            
            # Sort pair to be order-independent (optional, but good for "A and B" vs "B and A")
            # For strict collocation "have luck" vs "luck have", order matters? 
            # Usually strict collocation preserves order, but co-occurrence usually doesn't.
            # Let's preserve order for now to see "verb object" structures better.
            pair = (current_word, next_word)
            pair_counts[pair] += 1
            total_pairs += 1

    # 5. Calculate PMI
    # PMI(x, y) = log2( P(x,y) / ( P(x) * P(y) ) )
    # P(x,y) = count(x,y) / total_pairs
    # P(x) = count(x) / total_lemmas
    
    total_lemmas = len(all_lemmas)
    min_occurrence = 3 # Filter noise
    
    pmi_scores = []
    
    for pair, count in pair_counts.items():
        if count < min_occurrence:
            continue
            
        w1, w2 = pair
        c1 = lemma_counts[w1]
        c2 = lemma_counts[w2]
        
        p_pair = count / total_pairs
        p_w1 = c1 / total_lemmas
        p_w2 = c2 / total_lemmas
        
        pmi = math.log2(p_pair / (p_w1 * p_w2))
        
        pmi_scores.append((pair, pmi, count))

    # Sort by PMI
    pmi_scores.sort(key=lambda x: x[1], reverse=True)

    # 6. Output Results
    print("\n" + "="*60)
    print(f"{'Collocation (Lemma A, Lemma B)':<30} | {'PMI':<10} | {'Count':<5}")
    print("="*60)
    
    # Show top 50
    for pair, pmi, count in pmi_scores[:50]:
        print(f"{str(pair):<30} | {pmi:>10.2f} | {count:>5}")

    print("="*60)
    
    # Also show top by raw frequency (often interpretable differently)
    print("\nTop Frequent Pairs (Raw Count):")
    freq_scores = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    for pair, count in freq_scores[:20]:
         print(f"{str(pair):<30} | {count:>5}")

if __name__ == "__main__":
    analyze_collocations()
