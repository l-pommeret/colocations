
import os
import math
import collections

def get_author(metadata_lines):
    """
    Determines the author based on metadata lines.
    Returns 'Cicero', 'Caesar', 'Vergil', 'Ovid', 'Jerome', or 'Other'.
    """
    meta_text = " ".join(metadata_lines).lower()
    
    if 'phi0474' in meta_text or 'cicero' in meta_text or 'atticum' in meta_text or 'officiis' in meta_text:
        return 'Cicero'
    if 'phi0448' in meta_text or 'caesar' in meta_text or 'gallico' in meta_text:
        return 'Caesar'
    if 'phi0690' in meta_text or 'vergil' in meta_text or 'aeneid' in meta_text:
        return 'Vergil'
    if 'phi0959' in meta_text or 'ovid' in meta_text or 'metamorphoses' in meta_text:
        return 'Ovid'
    if 'jerome' in meta_text or 'vulgate' in meta_text or 'testamentum' in meta_text:
        return 'Jerome (Vulgate)'
        
    return 'Other'

def calculate_entropy(tokens):
    """
    Calculates Shannon Entropy (in bits) for a list of tokens.
    H(X) = - sum(p(x) * log2(p(x)))
    """
    if not tokens:
        return 0.0
    
    counts = collections.Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
        
    return entropy

def analyze_entropy():
    print("Analyzing Language Entropy by Author...")
    
    corpora = ['ud_latin_perseus', 'ud_latin_proiel']
    
    # Store lemmas by author
    author_lemmas = collections.defaultdict(list)
    
    STOP_POS = {'PUNCT', 'SYM', 'NUM', 'X'} # Basic filter only
    
    file_count = 0
    
    for corpus_dir in corpora:
        if not os.path.exists(corpus_dir):
            continue
            
        for root, dirs, files in os.walk(corpus_dir):
            for file in files:
                if file.endswith('.conllu'):
                    file_path = os.path.join(root, file)
                    file_count += 1
                    
                    # Parse file
                    current_metadata = []
                    current_lemmas = []
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            
                            if not line:
                                # End of sentence
                                if current_lemmas:
                                    author = get_author(current_metadata)
                                    if author != 'Other':
                                        author_lemmas[author].extend(current_lemmas)
                                    
                                    current_lemmas = []
                                    current_metadata = [] # Keep metadata? usually per doc, but sometimes repeated. 
                                    # Actually in CoNLLU, metadata often precedes sentence. 
                                    # We reset metadata after use, assuming it's per-sentence block.
                                continue
                                
                            if line.startswith('#'):
                                current_metadata.append(line)
                                continue
                            
                            parts = line.split('\t')
                            if len(parts) >= 4:
                                upos = parts[3]
                                lemma = parts[2].lower()
                                
                                # Basic filtering
                                if upos not in STOP_POS and lemma != '_':
                                    current_lemmas.append(lemma)
                    
                    # Handle last sentence
                    if current_lemmas and current_metadata:
                         author = get_author(current_metadata)
                         if author != 'Other':
                             author_lemmas[author].extend(current_lemmas)

    print(f"Processed {file_count} files.")
    print("-" * 65)
    print(f"{'Author':<20} | {'Tokens':<10} | {'Unique':<8} | {'Entropy':<8} | {'TTR':<6}")
    print("-" * 65)
    
    results = []
    
    for author, lemmas in author_lemmas.items():
        if len(lemmas) < 1000: continue # Skip small samples
        
        entropy = calculate_entropy(lemmas)
        unique_count = len(set(lemmas))
        total_count = len(lemmas)
        ttr = unique_count / total_count if total_count > 0 else 0
        
        results.append((author, total_count, unique_count, entropy, ttr))

    # Sort by Entropy desc
    results.sort(key=lambda x: x[3], reverse=True)
    
    for r in results:
        print(f"{r[0]:<20} | {r[1]:<10} | {r[2]:<8} | {r[3]:<8.4f} | {r[4]:.4f}")

    print("-" * 65)
    print("Key:")
    print("Entropy: Higher = More unpredictable / Richer vocabulary distribution")
    print("TTR (Type-Token Ratio): Higher = More unique words per text length")

if __name__ == "__main__":
    analyze_entropy()
