
import os
import collections
import math
import sys

def parse_conllu(file_path):
    """
    Parses a CoNLL-U file and returns a list of sentences.
    Each sentence is a list of token dicts (with 'lemma', 'upos', etc.).
    """
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            
            parts = line.split('\t')
            
            # CoNLL-U format has 10 columns. 
            # 0: ID, 1: FORM, 2: LEMMA, 3: UPOS, ...
            if len(parts) >= 4:
                token_id = parts[0]
                
                # Skip multi-word ranges (e.g. 1-2) if handled by sub-tokens, 
                # but in Latin UD they are rare or simple. 
                # We mainly want valid integer IDs for words.
                if '-' in token_id or '.' in token_id:
                    continue
                    
                token = {
                    'form': parts[1],
                    'lemma': parts[2],
                    'upos': parts[3]
                }
                current_sentence.append(token)
                
    if current_sentence:
        sentences.append(current_sentence)
        
    return sentences

def analyze_treebank_collocations():
    print("Analyzing Latin Treebank Data (Perseus + PROIEL)...")
    
    # Corpus directories
    corpora = ['ud_latin_perseus', 'ud_latin_proiel']
    
    # 1. Collect all lemmas
    # Filter: Keep DET/CCONJ as requested? 
    # USER UPDATE: "grammatical things" like "is qui", "sum et" are not exploitable yet.
    # So we MUST filter PRON (is, qui), AUX (sum), ADP (in, ad), SCONJ (ut, si).
    # We will keep NOUN, VERB, ADJ, ADV, PROPN.
    # Keeping DET/CCONJ? "res publica" is NOUN ADJ. "pater et filius" is NOUN CCONJ NOUN.
    # Let's be stricter to remove functional noise.
    
    STOP_POS = {'PUNCT', 'SYM', 'NUM', 'X', 'PRON', 'AUX', 'ADP', 'SCONJ', 'PART', 'DET'}
    # Keeping CCONJ for now? "et" is very frequent, maybe filter it too for pure lexical pairs?
    # User said "celui qui" (PRON PRON) and "il est" (PRON AUX/VERB). 
    # Let's filter CCONJ too to focus on "content words" (NOUN, ADJ, VERB).
    STOP_POS.add('CCONJ') 
    
    # Artifact blacklist
    BLACKLIST = {'calendar', 'expression', 'monetary', 'kal.', 'non.', 'id.'}

    all_lemmas = []
    
    file_count = 0
    
    for corpus_dir in corpora:
        if not os.path.exists(corpus_dir):
            print(f"Warning: {corpus_dir} not found.")
            continue
            
        for root, dirs, files in os.walk(corpus_dir):
            for file in files:
                if file.endswith('.conllu'):
                    file_path = os.path.join(root, file)
                    file_count += 1
                    
                    sentences = parse_conllu(file_path)
                    
                    for sent in sentences:
                        for token in sent:
                            upos = token['upos']
                            lemma = token['lemma']
                            
                            if upos not in STOP_POS and lemma:
                                # Clean lemma
                                lemma = lemma.lower()
                                
                                # Artifact filter
                                if lemma in BLACKLIST:
                                    continue
                                
                                # Should be clean already in Treebank, but good measure
                                # Some treebanks use '_' for missing lemma, check for likely valid words
                                if len(lemma) > 1 and lemma != '_': 
                                    all_lemmas.append(lemma)
    
    print(f"Processed {file_count} CoNLL-U files.")
    print(f"Total filtered lemmas: {len(all_lemmas)}")
    
    # 2. Count Collocations (Pairs) - Moving Window
    print("Counting pairs and trigrams...")
    
    lemma_counts = collections.Counter(all_lemmas)
    pair_counts = collections.Counter()
    trigram_counts = collections.Counter()
    
    window_size = 5
    total_pairs = 0
    
    # Pairs
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

    # Trigrams (Strict Adjacency)
    for i in range(len(all_lemmas) - 2):
        w1 = all_lemmas[i]
        w2 = all_lemmas[i+1]
        w3 = all_lemmas[i+2]
        
        # Simple distinct filter
        if w1 != w2 and w2 != w3 and w1 != w3:
             trigram = (w1, w2, w3)
             trigram_counts[trigram] += 1
             
    # 3. Calculate PMI
    total_lemmas = len(all_lemmas)
    min_occurrence = 5 # Good threshold for treebank size
    
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

    pmi_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 4. Output
    output_filename = 'collocations_latin_treebank.txt'
    print(f"Writing results to {output_filename}...")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("LATIN TREEBANK ANALYSIS (Perseus + PROIEL)\n")
        f.write("Source: High-quality hand-annotated lemmas (CoNLL-U)\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total Filtered Tokens: {len(all_lemmas)}\n\n")
        
        f.write("TOP COLLOCATIONS (PAIRS) by PMI (Min count: 5)\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Collocation':<30} | {'PMI':<10} | {'Count':<5}\n")
        f.write("-" * 60 + "\n")
        
        for pair, pmi, count in pmi_scores[:1000]:
            f.write(f"{str(pair):<30} | {pmi:>10.2f} | {count:>5}\n")

        f.write("\n" + "="*60 + "\n")
        f.write("TOP FREQUENT PAIRS (Raw Count)\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Pair':<30} | {'Count':<5}\n")
        f.write("-" * 60 + "\n")
        
        freq_scores = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
        for pair, count in freq_scores[:1000]:
             f.write(f"{str(pair):<30} | {count:>5}\n")

        f.write("\n" + "="*60 + "\n")
        f.write("TOP TRIGRAMS (3-Word Phrases) by FREQUENCY\n")
        f.write("="*60 + "\n")
        
        sorted_trigrams = sorted(trigram_counts.items(), key=lambda x: x[1], reverse=True)
        
        for trigram, count in sorted_trigrams[:200]:
             f.write(f"{str(trigram):<45} | {count:>5}\n")

    print("Done.")

if __name__ == "__main__":
    analyze_treebank_collocations()
