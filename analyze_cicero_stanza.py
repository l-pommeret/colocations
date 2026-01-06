
import os
import stanza
import collections
import time
import math
import sys

def analyze_full_cicero_stanza():
    print("Initializing Stanza for Latin (Full Corpus Analysis)...")
    
    # Configure logging to suppress noisy output if possible
    
    # Initialize Pipeline (CPU is fine for this size, might take a few mins)
    # processors: tokenize, mwt (multi-word token), pos, lemma
    # We need POS to filter stop words!
    nlp = stanza.Pipeline('la', processors='tokenize,mwt,pos,lemma', use_gpu=False, verbose=False)
    
    cicero_dir = 'latin_library_corpus/cicero'
    files = [f for f in os.listdir(cicero_dir) if f.endswith('.txt')]
    
    print(f"Found {len(files)} files in {cicero_dir}. Starting processing...")
    
    # FILTERING CONFIGURATION (Same as analyze_cicero.py)
    # Exclude functional words to focus on content collocations
    STOP_POS = {'PUNCT', 'SYM', 'NUM', 'X', 'PRON', 'AUX', 'ADP', 'SCONJ', 'PART', 'DET', 'CCONJ'}
    BLACKLIST = {'calendar', 'expression', 'monetary', 'kal.', 'non.', 'id.', 'c.', 'cn.', 'm.', 'l.'}
    
    lemma_counts = collections.Counter()
    pair_counts = collections.Counter()
    total_pairs = 0
    window_size = 5 # Standard window
    
    processed_files = 0
    start_time = time.time()
    
    # Process files
    for filename in files:
        filepath = os.path.join(cicero_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
            
        # Optimization: Split by paragraphs to avoid Stanza OOM or slowdown on huge texts
        paragraphs = text.split('\n\n')
        
        file_lemmas = []
        
        for para in paragraphs:
            if not para.strip(): continue
            
            # Process paragraph with Stanza
            # Stanza can be slow on very long paragraphs, but usually OK.
            try:
                doc = nlp(para)
            except Exception as e:
                # Fallback or skip
                continue
            
            for sent in doc.sentences:
                for word in sent.words:
                    # Filter logic
                    upos = word.upos
                    lemma = word.lemma
                    
                    if lemma:
                        lemma = lemma.lower()
                        
                        # Apply Filtering
                        if upos not in STOP_POS:
                            if lemma not in BLACKLIST and len(lemma) > 1 and lemma != '_':
                                file_lemmas.append(lemma)
                                lemma_counts[lemma] += 1
        
        # Calculate pairs for this file (intra-file collocations)
        # Using a sliding window on the filtered lemma stream
        for i in range(len(file_lemmas)):
            current_word = file_lemmas[i]
            
            # Look ahead in window
            end_window = min(i + window_size + 1, len(file_lemmas))
            
            for j in range(i + 1, end_window):
                next_word = file_lemmas[j]
                
                if current_word == next_word: continue
                
                # Order independent pairs
                pair = tuple(sorted((current_word, next_word)))
                pair_counts[pair] += 1
                total_pairs += 1

        processed_files += 1
        if processed_files % 5 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / processed_files
            remaining = (len(files) - processed_files) * avg_time
            print(f"Processed {processed_files}/{len(files)} files... (Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.0f}s)")
            
            # Intermediate save every 20 files? No, might corrupt. Just wait.

    print("Processing complete.")
    print(f"Total pairs counted: {total_pairs}")
    
    # Calculate PMI
    # PMI(x, y) = log2( P(x,y) / (P(x)*P(y)) )
    # P(x, y) = count(pair) / total_pairs
    # P(x) = count(x) / total_lemmas
    
    print("Calculating PMI...")
    total_lemmas = sum(lemma_counts.values())
    
    pmi_scores = []
    min_occurrence = 5 # Filter noise
    
    for pair, count in pair_counts.items():
        if count < min_occurrence: continue
        
        w1, w2 = pair
        
        p_pair = count / total_pairs
        p_w1 = lemma_counts[w1] / total_lemmas
        p_w2 = lemma_counts[w2] / total_lemmas
        
        try:
            pmi = math.log2(p_pair / (p_w1 * p_w2))
        except ValueError:
            pmi = 0
            
        pmi_scores.append((pair, pmi, count))
    
    pmi_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Output Results
    output_filename = 'collocations_cicero_stanza_full.txt'
    print(f"Writing results to {output_filename}...")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CICERO FULL CORPUS ANALYSIS (Stanza Lemmatization)\n")
        f.write("Source: Latin Library (Full Text)\n")
        f.write("Method: Stanza NLP (UD Model), Window Size: 5, Stopwords Removed\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Files Processed: {processed_files}\n")
        f.write(f"Total Lemmas: {total_lemmas}\n")
        f.write(f"Unique Pairs Found: {len(pair_counts)}\n\n")
        
        # Section 1: Top PMI
        f.write(f"{'TOP COLLOCATIONS by PMI (Min count: {min_occurrence})':<50}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Collocation':<40} | {'PMI':<10} | {'Count':<5}\n")
        f.write("-" * 80 + "\n")
        
        for pair, pmi, count in pmi_scores[:5000]:
             f.write(f"{str(pair):<40} | {pmi:>10.2f} | {count:>5}\n")

        # Section 2: Top Frequency
        f.write("\n" + "="*80 + "\n")
        f.write("TOP FREQUENT PAIRS (Raw Count)\n")
        f.write("-" * 80 + "\n")
        for pair, count in pair_counts.most_common(5000):
             f.write(f"{str(pair):<40} | {count:>5}\n")
             
    print(f"Results written to {output_filename}")

if __name__ == "__main__":
    analyze_full_cicero_stanza()
