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
    # use_gpu=False is safer for some environments
    nlp = stanza.Pipeline('grc', processors='tokenize,lemma,pos', use_gpu=False, verbose=False)

    # 2. Read and Clean Corpus
    print("Reading corpus...")
    corpus_path = 'homer.iliad.tess'
    clean_lines = []
    
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Format: <hom. il. 1.1>\tTEXT...
                cleaned = re.sub(r'<[^>]+>\s*', '', line).strip()
                if cleaned:
                    clean_lines.append(cleaned)
    except FileNotFoundError:
        print(f"Error: {corpus_path} not found.")
        return

    print(f"Read {len(clean_lines)} lines.")

    # 3. Process Text and Extract Lemmas (with Stopword Filter)
    print("Processing text (filtering stopwords)...")
    
    # Define stopwords by POS tags
    STOP_POS = {'DET', 'CCONJ', 'SCONJ', 'PART', 'PRON', 'PUNCT', 'SYM', 'NUM'}
    
    all_lemmas = []
    
    chunk_size = 100
    total_chunks = (len(clean_lines) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(clean_lines), chunk_size):
        chunk = clean_lines[i:i+chunk_size]
        text_chunk = " ".join(chunk)
        
        doc = nlp(text_chunk)
        
        for sentence in doc.sentences:
            for word in sentence.words:
                # Logic Improvement: Filter stopwords
                if word.upos not in STOP_POS and word.lemma:
                    all_lemmas.append(word.lemma)
        
        if (i // chunk_size) % 10 == 0:
            print(f"Processed chunk {i // chunk_size + 1}/{total_chunks}")

    print(f"Total lemmas extracted (filtered): {len(all_lemmas)}")

    # 4. Count Frequencies and Co-occurrences (Pairs & Trigrams)
    print("Calculating statistics...")
    
    lemma_counts = collections.Counter(all_lemmas)
    pair_counts = collections.Counter()
    trigram_counts = collections.Counter()
    
    window_size = 5
    total_pairs = 0
    
    # 4a. Pairs (Bigrams) with Window
    for i in range(len(all_lemmas)):
        current_word = all_lemmas[i]
        
        end_window = min(i + window_size, len(all_lemmas))
        for j in range(i + 1, end_window):
            next_word = all_lemmas[j]
            
            # Logic Improvement: Remove self-collocations
            if current_word == next_word:
                continue
            
            pair = (current_word, next_word)
            pair_counts[pair] += 1
            total_pairs += 1

    # 4b. Trigrams (Consecutive 3 words)
    # Note: For trigrams, we usually look at strict adjacency or small window.
    # Here let's do strict adjacency 3-grams from the filtered list (which means they were close in semantic space).
    for i in range(len(all_lemmas) - 2):
        w1 = all_lemmas[i]
        w2 = all_lemmas[i+1]
        w3 = all_lemmas[i+2]
        
        # Avoid self-repetition in trigrams too? (e.g. w1==w2 or w2==w3)
        if w1 != w2 and w2 != w3 and w1 != w3:
             trigram = (w1, w2, w3)
             trigram_counts[trigram] += 1

    # 5. Calculate PMI for Pairs
    total_lemmas = len(all_lemmas)
    min_occurrence = 3 
    
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
        
        try:
             pmi = math.log2(p_pair / (p_w1 * p_w2))
        except ValueError:
             pmi = 0
        
        pmi_scores.append((pair, pmi, count))

    # Sort by PMI
    pmi_scores.sort(key=lambda x: x[1], reverse=True)

    # 6. Output Results
    output_filename = 'collocations_iliade_v2.txt'
    print(f"Writing results to {output_filename}...")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("ANALYSIS V2: Stopwords Removed, No Self-Collocations\n")
        f.write("="*60 + "\n\n")
        
        f.write("TOP COLLOCATIONS (PAIRS) by PMI\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Collocation':<30} | {'PMI':<10} | {'Count':<5}\n")
        f.write("-" * 60 + "\n")
        
        for pair, pmi, count in pmi_scores[:100]:
            f.write(f"{str(pair):<30} | {pmi:>10.2f} | {count:>5}\n")

        f.write("\n" + "="*60 + "\n")
        f.write("TOP TRIGRAMS (3-Word Phrases) by FREQUENCY\n")
        f.write("(Derived from filtered lemma sequence)\n")
        f.write("="*60 + "\n")
        
        # Sort trigrams by frequency
        sorted_trigrams = sorted(trigram_counts.items(), key=lambda x: x[1], reverse=True)
        
        for trigram, count in sorted_trigrams[:50]:
             f.write(f"{str(trigram):<45} | {count:>5}\n")
             
    print("Done.")

if __name__ == "__main__":
    analyze_collocations()
