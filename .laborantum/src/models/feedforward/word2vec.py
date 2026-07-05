import heapq

import torch
import torch.nn.functional as F


class HierarchicalSoftmaxTargets:
    def __init__(self, word_counts, word_to_index):
        active_counts = {
            word_to_index[word]: int(count)
            for word, count in word_counts.items()
            if word in word_to_index
        }
        if len(active_counts) <= 1:
            self.paths = {0: [0]}
            self.codes = {0: [1]}
            self.num_internal_nodes = 1
            self.max_path_length = 1
            return

        next_internal_id = 0
        serial = 0
        heap = []
        for word_index, count in active_counts.items():
            heapq.heappush(heap, (count, serial, {'word': word_index}))
            serial += 1

        while len(heap) > 1:
            left_count, _, left = heapq.heappop(heap)
            right_count, _, right = heapq.heappop(heap)
            node = {
                'node': next_internal_id,
                'left': left,
                'right': right,
            }
            next_internal_id += 1
            heapq.heappush(heap, (left_count + right_count, serial, node))
            serial += 1

        paths = {}
        codes = {}

        def walk(node, path, code):
            if 'word' in node:
                paths[node['word']] = path.copy()
                codes[node['word']] = code.copy()
                return
            walk(node['left'], path + [node['node']], code + [0])
            walk(node['right'], path + [node['node']], code + [1])

        walk(heap[0][2], [], [])
        self.paths = paths
        self.codes = codes
        self.num_internal_nodes = next_internal_id
        self.max_path_length = max(len(path) for path in self.paths.values())

    def __call__(self, context_word):
        device = context_word.device
        context_word = context_word.detach().cpu().view(-1).tolist()
        paths = []
        codes = []
        masks = []
        for word_index in context_word:
            path = self.paths[int(word_index)]
            code = self.codes[int(word_index)]
            padding = self.max_path_length - len(path)
            paths.append(path + [0] * padding)
            codes.append(code + [0] * padding)
            masks.append([1.0] * len(path) + [0.0] * padding)
        return {
            'path': torch.tensor(paths, dtype=torch.long, device=device),
            'code': torch.tensor(codes, dtype=torch.float32, device=device),
            'mask': torch.tensor(masks, dtype=torch.float32, device=device),
        }


class HierarchicalSoftmaxLoss(torch.nn.Module):
    def __init__(self, model, targets):
        super().__init__()
        self.model = model
        self.targets = targets

    def forward(self, batch):
        target_tensors = self.targets(batch['data']['context_word'])
        batch['data'].update(target_tensors)
        embedding = batch['signals']['embedding']
        node_vectors = self.model.decoder(batch['data']['path'])
        logits = torch.einsum('bd,bld->bl', embedding, node_vectors)
        batch['signals']['logits'] = logits
        batch['signals']['probabilities'] = torch.sigmoid(logits)
        batch['postprocessed']['code'] = (batch['signals']['probabilities'] >= 0.5).long()
        per_node_loss = F.binary_cross_entropy_with_logits(
            logits,
            batch['data']['code'],
            reduction='none',
        )
        masked_loss = per_node_loss * batch['data']['mask']
        return masked_loss.sum() / batch['data']['mask'].sum().clamp_min(1.0)


class Word2VecHierarchicalSoftmax(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_internal_nodes):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.num_internal_nodes = int(num_internal_nodes)
        self.encoder = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.decoder = torch.nn.Embedding(self.num_internal_nodes, self.embedding_dim)

        torch.nn.init.xavier_uniform_(self.encoder.weight)
        torch.nn.init.xavier_uniform_(self.decoder.weight)

    def __forward_kernel(self, center_word, path):
        embedding = self.encoder(center_word)
        node_vectors = self.decoder(path)
        logits = torch.einsum('bd,bld->bl', embedding, node_vectors)
        return embedding, logits

    def forward(self, batch):
        if 'signals' not in batch:
            batch['signals'] = {
                'embedding': self.encoder(batch['data']['center_word']),
            }
            batch['postprocessed'] = {}

        if 'path' in batch['data']:
            embedding, logits = self.__forward_kernel(
                batch['data']['center_word'],
                batch['data']['path']
            )
            batch['signals']['embedding'] = embedding
            batch['signals']['logits'] = logits
            batch['signals']['probabilities'] = torch.sigmoid(logits)
            batch['postprocessed']['code'] = (batch['signals']['probabilities'] >= 0.5).long()

        return batch


class BinaryIndexTree:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.depth = max(1, (vocab_size - 1).bit_length())
        self.max_path_length = self.depth
        self.num_internal_nodes = (1 << self.depth) - 1 

    def targets_for_index(self, word_index):
        return format(word_index, f'0{self.depth}b')

    def node_id_from_prefix(self, prefix_bits):
        if not prefix_bits:
            return 0
        val = int(prefix_bits, 2)
        d = len(prefix_bits)
        return (1 << d) - 1 + val 

    def path_and_targets(self, word_index):
        bits = self.targets_for_index(word_index)
        path = []
        targets = []
        prefix = ""
        for bit in bits:
            path.append(self.node_id_from_prefix(prefix))
            targets.append(int(bit))
            prefix += bit
        return path, targets

    def __call__(self, context_word):
        device = context_word.device
        indices = context_word.detach().cpu().view(-1).tolist()
        paths = []
        targets = []
        for idx in indices:
            p, t = self.path_and_targets(idx)
            paths.append(p)
            targets.append(t)
        path_tensor = torch.tensor(paths, dtype=torch.long, device=device)
        targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
        mask = torch.ones((len(indices), self.max_path_length), dtype=torch.float32, device=device)
        return {"path": path_tensor, "targets": targets_tensor, "mask": mask}
    


class HierarchicalSoftmax(torch.nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.vocab_size = int(vocab_size)
        self.targets = BinaryIndexTree(self.vocab_size)
        self.decoder = torch.nn.Embedding(
            self.targets.num_internal_nodes, self.embedding_dim
        )
        torch.nn.init.normal_(self.decoder.weight, mean=0.0, std=0.02)
        self.num_internal_nodes = self.targets.num_internal_nodes
        self.max_path_length = self.targets.max_path_length

    def forward(self, embedding, target_word):

        tree_data = self.targets(target_word)
        path = tree_data['path']
        targets = tree_data['targets']
        mask = tree_data['mask']

        node_vectors = self.decoder(path) 

        logits = torch.einsum('bd,bld->bl', embedding, node_vectors)

        probabilities = torch.sigmoid(logits) 

        target_probabilities = torch.where(
            targets == 1.0,
            probabilities,
            1.0 - probabilities
        )

        masked_target_probs = target_probabilities * mask + (1.0 - mask)
        total_probability = masked_target_probs.prod(dim=1)

        per_node_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        masked_loss = per_node_loss * mask
        per_word_loss = masked_loss.sum(dim=1)

        loss = per_word_loss.mean()

        return {
            'path': path,
            'targets': targets,
            'mask': mask,
            'logits': logits,
            'probabilities': probabilities,
            'target_probabilities': target_probabilities,
            'total_probability': total_probability,
            'per_node_loss': per_node_loss,
            'per_word_loss': per_word_loss,
            'loss': loss,
        }
    



class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.encoder = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.hierarchical_softmax = HierarchicalSoftmax(self.embedding_dim, self.vocab_size)
        self.decoder = self.hierarchical_softmax.decoder
        self.num_internal_nodes = self.hierarchical_softmax.targets.num_internal_nodes
        torch.nn.init.normal_(self.encoder.weight, mean=0.0, std=0.02)

    def forward(self, batch):
        center_word = batch['data']['center_word']
        embedding = self.encoder(center_word)

        batch['signals'] = {'embedding': embedding}
        batch['postprocessed'] = {}

        if 'context_word' in batch['data']:
            hs_output = self.hierarchical_softmax(embedding, batch['data']['context_word'])

            batch['data']['path'] = hs_output['path']
            batch['data']['targets'] = hs_output['targets']
            batch['data']['mask'] = hs_output['mask']

            batch['signals']['logits'] = hs_output['logits']
            batch['signals']['probabilities'] = hs_output['probabilities']
            batch['signals']['target_probabilities'] = hs_output['target_probabilities']
            batch['signals']['total_probability'] = hs_output['total_probability']
            batch['signals']['loss'] = hs_output['loss']

            batch['postprocessed']['targets'] = (hs_output['probabilities'] >= 0.5).long()

        return batch