import re
import argparse
from collections import Counter, defaultdict


def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        chars = word.split()
        for i in range(len(chars)-1):
            pairs[chars[i], chars[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = Counter()
    bigram = ' '.join(pair)
    pattern = re.compile(bigram)
    for word in v_in:
        w_out = pattern.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def main():
    ap = argparse.ArgumentParser(description="")
    ap.add_argument('--txtfile', type=str, required=True, help='input txt corpus')
    ap.add_argument('--num_iter', type=int, default=100, help='number of iteration for BPE merge')
    ap.add_argument('--bpefile', type=str, default="bpe.codes", help='output bpe file for BPE codes')
    args = ap.parse_args()


    fout = open(args.bpefile, 'w')
    lines = open(args.txtfile, 'r', encoding='utf-8').read()
    words = re.split('\s+', lines)
    words = [' '.join([char for char in word] + ["</w>"]) for word in words ]
    vocab = Counter(words)

    for i in range(args.num_iter):
        pairs = get_stats (vocab)
        # all possible subword bi-gram have been merged
        if len(pairs) == 0:
            break;
        best = max(pairs, key=pairs.get)
        freq = pairs[best]
        vocab = merge_vocab(best, vocab)
        fout.write('{} : {}\n'.format(best, freq))
        print (best, freq)

    fout.close()


if __name__ == '__main__':
    main();
