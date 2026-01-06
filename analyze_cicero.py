
import os
import collections
import math
import sys

def parse_conllu_cicero(file_path):
    """
    Parses a CoNLL-U file and returns a list of sentences that belong to Cicero.
    Filtering based on metadata comments (# sent_id, # source, # newdoc).
    """
    sentences = []
    current_sentence = []
    current_metadata = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                # End of sentence
                if current_sentence:
                    # Check metadata for Cicero
                    is_cicero = False
                    meta_text = " ".join(current_metadata)
                    
                    # Filter Logic
                    # Perseus: phi0474
                    # PROIEL: Epistulae ad Atticum, De Officiis, In Catilinam, etc.
                    # Also generic "Cicero" check
                    if 'phi0474' in meta_text or \
                       'Cicero' in meta_text or \
                       'Atticum' in meta_text or \
                       'Officiis' in meta_text or \
                       'Catilinam' in meta_text or \
                       'Archia' in meta_text or \
                       'Murena' in meta_text or \
                       'Sestio' in meta_text or \
                       'Verr' in meta_text:
                        is_cicero = True
                    
                    if is_cicero:
                        sentences.append(current_sentence)
                        
                    current_sentence = []
                    current_metadata = []
                continue
                
            if line.startswith('#'):
                current_metadata.append(line)
                continue
            
            parts = line.split('\t')
            
            if len(parts) >= 4:
                token_id = parts[0]
                if '-' in token_id or '.' in token_id:
                    continue
                    
                token = {
                    'form': parts[1],
                    'lemma': parts[2],
                    'upos': parts[3]
                }
                current_sentence.append(token)
                
    # Handle last sentence
    if current_sentence:
        meta_text = " ".join(current_metadata)
        if 'phi0474' in meta_text or \
           'Cicero' in meta_text or \
           'Atticum' in meta_text or \
           'Officiis' in meta_text: # etc
             sentences.append(current_sentence)
        
    return sentences

def analyze_cicero_collocations():
    print("Analyzing CICERO (Treebank Data)...")
    
    corpora = ['ud_latin_perseus', 'ud_latin_proiel']
    
    # Lexical Focus Filters from previous step
    STOP_POS = {'PUNCT', 'SYM', 'NUM', 'X', 'PRON', 'AUX', 'ADP', 'SCONJ', 'PART', 'DET', 'CCONJ'}
    BLACKLIST = {'calendar', 'expression', 'monetary', 'kal.', 'non.', 'id.'}

    all_lemmas = []
    file_count = 0
    
    for corpus_dir in corpora:
        if not os.path.exists(corpus_dir):
            continue
            
        for root, dirs, files in os.walk(corpus_dir):
            for file in files:
                if file.endswith('.conllu'):
                    file_path = os.path.join(root, file)
                    file_count += 1
                    
                    sentences = parse_conllu_cicero(file_path)
                    
                    for sent in sentences:
                        for token in sent:
                            upos = token['upos']
                            lemma = token['lemma']
                            
                            if upos not in STOP_POS and lemma:
                                lemma = lemma.lower()
                                if lemma in BLACKLIST: continue
                                if len(lemma) > 1 and lemma != '_': 
                                    all_lemmas.append(lemma)
    
    print(f"Processed {file_count} files.")
    print(f"Total Cicero lemmas: {len(all_lemmas)}")
    
    if len(all_lemmas) == 0:
        print("No Cicero data found! Check filters.")
        return

    # Count
    # Analyze Pairs with VARYING WINDOW SIZES (3, 4, 5)
    windows_to_test = [3, 4, 5]
    
    for w_size in windows_to_test:
        print(f"Analyzing Pairs with Window Size = {w_size}...")
        
        pair_counts = collections.Counter()
        total_pairs = 0
        
        for i in range(len(all_lemmas)):
            current_word = all_lemmas[i]
            end_window = min(i + w_size, len(all_lemmas))
            for j in range(i + 1, end_window):
                next_word = all_lemmas[j]
                if current_word == next_word: continue
                
                # Normalize pair (A, B) == (B, A)
                pair = tuple(sorted((current_word, next_word)))
                pair_counts[pair] += 1
                total_pairs += 1
        
        # Output for this window size
        filename = f'collocations_cicero_pairs_window{w_size}.txt'
        print(f"Writing results to {filename}...")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write(f"CICERO PAIRS - WINDOW SIZE {w_size}\n")
            f.write("="*60 + "\n\n")
            f.write(f"{'Pair':<30} | {'Count':<5}\n")
            f.write("-" * 60 + "\n")
            for pair, count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5000]:
                 f.write(f"{str(pair):<30} | {count:>5}\n")

    print("Done.")

if __name__ == "__main__":
    analyze_cicero_collocations()
