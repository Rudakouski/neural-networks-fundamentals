from collections import Counter
import re

import torch


class SkipGramDataset:
    def __init__(self, corpus, window_size=2, min_count=1):
        self.window_size = int(window_size)
        self.min_count = int(min_count)

        self.tokens = self._tokenize(corpus)

        freq = Counter(self.tokens)

        filtered = {word: count for word, count in freq.items() if count >= self.min_count}

        sorted_words = sorted(filtered.keys(), key=lambda w: (-filtered[w], w))

        self.vocab = sorted_words
        self.word_to_index = {word: i for i, word in enumerate(self.vocab)}
        self.index_to_word = {i: word for i, word in enumerate(self.vocab)}
        self.word_counts = {word: filtered[word] for word in self.vocab}

        indexed_tokens = [self.word_to_index.get(token, -1) for token in self.tokens]
        pairs = []
        for center_pos, center_idx in enumerate(indexed_tokens):
            if center_idx == -1:
                continue
            left = max(0, center_pos - self.window_size)
            right = min(len(indexed_tokens), center_pos + self.window_size + 1)
            for context_pos in range(left, right):
                if context_pos == center_pos:
                    continue
                context_idx = indexed_tokens[context_pos]
                if context_idx != -1:
                    pairs.append((center_idx, context_idx))
        self.pairs = pairs

    def _tokenize(self, corpus):
        text = ' '.join(corpus) if isinstance(corpus, (list, tuple)) else str(corpus)
        return [
            self._normalize_token(token)
            for token in re.findall(r'[a-z]+', text, flags=re.IGNORECASE)
        ]

    def _normalize_token(self, token):
        token = token.lower()
        original = token
        if len(token) > 5 and token.endswith('ing'):
            token = token[:-3]
        elif len(token) > 4 and token.endswith('ed'):
            token = token[:-2]
        elif len(token) > 4 and token.endswith('es') and not token.endswith('ses'):
            token = token[:-2]
        elif len(token) > 3 and token.endswith('s') and not token.endswith('ss'):
            token = token[:-1]
        if len(token) < 2:
            return original
        return token

    def _make_pairs(self, indexed_tokens, window_size):
        pairs = []
        for center_position, center_word in enumerate(indexed_tokens):
            left = max(0, center_position - window_size)
            right = min(len(indexed_tokens), center_position + window_size + 1)
            for context_position in range(left, right):
                if context_position != center_position:
                    pairs.append((center_word, indexed_tokens[context_position]))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        center_word, context_word = self.pairs[index]
        return {
            'center_word': torch.tensor(center_word, dtype=torch.long),
            'context_word': torch.tensor(context_word, dtype=torch.long),
        }
