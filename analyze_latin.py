
import stanza
import collections
import math
import re
import sys

def analyze_collocations_latin():
    # 1. Setup Stanza for Latin
    print("Downloading/Loading Stanza model for Latin...")
    try:
        stanza.download('la')
    except Exception as e:
        print(f"Model download warning (might already be present): {e}")
    
    # Initialize pipeline
    # use_gpu=False is safer for some environments
    nlp = stanza.Pipeline('la', processors='tokenize,lemma,pos', use_gpu=False, verbose=False)

    # 2. Read and Clean Corpus
    print("Reading corpus...")
    import html
    import os

    # 2. Read and Clean Corpus (Walker)
    print("Reading corpus from 'latin_library_corpus'...")
    corpus_root = 'latin_library_corpus'
    clean_lines = []
    
    # Optional: Filter for specific authors to avoid medieval/neo-latin if desired, 
    # or just process everything. User said "gros corpus". 
    # Let's process everything but maybe limit file count if it's too huge?
    # No, let's go for it, but showing progress is key.
    
    file_count = 0
    
    # Filter for major Classical authors to ensure quality and reasonable runtime
    # Cicero alone is huge, so this is already a "gros corpus".
    TARGET_AUTHORS = [
        'vergil', 'cicero', 'caesar', 'ovid', 'horace', 'seneca', 
        'tacitus', 'livy', 'catullus', 'lucretius', 'sallust'
    ]
    
    for root, dirs, files in os.walk(corpus_root):
        for file in files:
            if file.endswith('.txt'):
                # Check if file or folder path matches target authors
                # Normalized path for checking
                path_lower = os.path.join(root, file).lower()
                
                if not any(author in path_lower for author in TARGET_AUTHORS):
                    continue
                    
                file_path = os.path.join(root, file)
                file_count += 1
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            # 1. Unescape
                            line = html.unescape(line)
                            # 2. Remove tags
                            cleaned = re.sub(r'<[^>]+>', ' ', line)
                            cleaned = cleaned.strip()
                            
                            # Filter
                            if cleaned and "The Latin Library" not in cleaned and "Latin Library" not in cleaned:
                                 clean_lines.append(cleaned)
                                 
                except Exception as e:
                    print(f"Skipping {file}: {e}")

    print(f"Read {len(clean_lines)} lines from {file_count} files.")

    print(f"Read {len(clean_lines)} lines.")

    # 3. Process Text and Extract Lemmas (Restricted Stopwords)
    print("Processing text...")
    
    # User requested to keep key grammatical words (kai/atque/determinants)
    # The default Stanford POS tags include: DET, CCONJ, SCONJ, etc.
    # We will ONLY filter punctuation and numbers, and maybe very mostly useless particles if needed
    # But user specifically asked to keep DET and CCONJ. 
    # Let's filter minimal set: PUNCT, NUM, SYM, X (foreign/unknown)
    
    STOP_POS = {'PUNCT', 'SYM', 'NUM', 'X'}
    
    all_lemmas = []
    
    chunk_size = 100
    total_chunks = (len(clean_lines) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(clean_lines), chunk_size):
        chunk = clean_lines[i:i+chunk_size]
        text_chunk = " ".join(chunk)
        
        doc = nlp(text_chunk)
        
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos not in STOP_POS and word.lemma:
                    lemma = word.lemma.lower()
                    # Filter out non-alphabetic chars (e.g. Greek, +, numbers)
                    # Simple regex: only keep if it contains at least one latin letter 
                    # and no weird symbols. 
                    # Actually, strict filter: must match ^[a-z\u00C0-\u00FF]+$ (including accented chars?)
                    # Latin Library uses standard chars closer to ASCII usually, but let's be safe.
                    # Or simpler: if re.match(r'^[a-zA-Z]+$', lemma): (but we need to handle accents if any)
                    # Let's clean it:
                    
                    # Remove punctuation attached to lemma if any
                    lemma_clean = re.sub(r'[^a-z]+', '', lemma)
                    
                    if len(lemma_clean) > 1: # Skip single letters/artifacts
                        all_lemmas.append(lemma_clean)
        
        if (i // chunk_size) % 10 == 0:
            print(f"Processed chunk {i // chunk_size + 1}/{total_chunks}")

    print(f"Total lemmas extracted: {len(all_lemmas)}")

    # 4. Count Frequencies and Co-occurrences
    print("Calculating statistics...")
    
    lemma_counts = collections.Counter(all_lemmas)
    pair_counts = collections.Counter()
    trigram_counts = collections.Counter()
    
    window_size = 5
    total_pairs = 0
    
    # Pairs (Bigrams) with Window
    for i in range(len(all_lemmas)):
        current_word = all_lemmas[i]
        
        end_window = min(i + window_size, len(all_lemmas))
        for j in range(i + 1, end_window):
            next_word = all_lemmas[j]
            
            if current_word == next_word:
                continue
            
            pair = (current_word, next_word)
            pair_counts[pair] += 1
            total_pairs += 1

    # Trigrams (Consecutive)
    for i in range(len(all_lemmas) - 2):
        w1 = all_lemmas[i]
        w2 = all_lemmas[i+1]
        w3 = all_lemmas[i+2]
        
        if w1 != w2 and w2 != w3 and w1 != w3:
             trigram = (w1, w2, w3)
             trigram_counts[trigram] += 1

    # 5. Calculate PMI
    total_lemmas = len(all_lemmas)
    min_occurrence = 10 # Higher threshold for large corpus to avoid hapax noise
    
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
    output_filename = 'collocations_latin.txt'
    print(f"Writing results to {output_filename}...")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("ANALYSIS LATIN: Minimal Stopword Filtering (Kept DET/CONJ)\n")
        f.write("="*60 + "\n\n")
        
        f.write("TOP COLLOCATIONS (PAIRS) by PMI (Min count: 10)\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Collocation':<30} | {'PMI':<10} | {'Count':<5}\n")
        f.write("-" * 60 + "\n")
        
        for pair, pmi, count in pmi_scores[:100]:
            f.write(f"{str(pair):<30} | {pmi:>10.2f} | {count:>5}\n")

        f.write("\n" + "="*60 + "\n")
        f.write("TOP FREQUENT PAIRS (Raw Count)\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Pair':<30} | {'Count':<5}\n")
        f.write("-" * 60 + "\n")
        
        # Sort by frequency
        freq_scores = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
        for pair, count in freq_scores[:100]:
             f.write(f"{str(pair):<30} | {count:>5}\n")

        f.write("\n" + "="*60 + "\n")
        f.write("TOP TRIGRAMS (3-Word Phrases) by FREQUENCY\n")
        f.write("="*60 + "\n")
        
        sorted_trigrams = sorted(trigram_counts.items(), key=lambda x: x[1], reverse=True)
        
        for trigram, count in sorted_trigrams[:50]:
             f.write(f"{str(trigram):<45} | {count:>5}\n")

    print("Done.")

if __name__ == "__main__":
    analyze_collocations_latin()
