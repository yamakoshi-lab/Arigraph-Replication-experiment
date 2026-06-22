import torch
import numpy as np
from FlagEmbedding import BGEM3FlagModel

class BGEM3Retriever:
    def __init__(self, model_name_or_path='BAAI/bge-m3', device=None, use_fp16=True, pooling='cls', hybrid_alpha=0.6):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Initializing BGEM3FlagModel on {device}...")
        self.model = BGEM3FlagModel(model_name_or_path, use_fp16=use_fp16, device=device)
        self.model.model.to(device)
        self.device = device
        self.pooling = pooling
        self.hybrid_alpha = hybrid_alpha
        
        # Cache for embeddings to avoid re-computation
        self.cache_dense_cls = {}
        self.cache_dense_mean = {}
        self.cache_sparse = {}

    def get_device(self):
        return next(self.model.model.parameters()).device

    def _encode_keys(self, keys, mode):
        """Batch encode keys that are not in the cache."""
        device = self.get_device()
        
        # Find which keys need encoding
        uncached_keys = []
        if mode == 'cls':
            uncached_keys = [k for k in keys if k not in self.cache_dense_cls]
        elif mode == 'mean':
            uncached_keys = [k for k in keys if k not in self.cache_dense_mean]
        elif mode == 'sparse':
            uncached_keys = [k for k in keys if k not in self.cache_sparse]
        elif mode == 'hybrid':
            uncached_keys = [k for k in keys if (k not in self.cache_dense_mean or k not in self.cache_sparse)]

        if uncached_keys:
            # Batch encode
            # For sparse or hybrid, we need FlagEmbedding's encode
            if mode in ('cls', 'sparse', 'hybrid'):
                # FlagEmbedding's encode returns a dict
                outputs = self.model.encode(
                    uncached_keys,
                    return_dense=(mode in ('cls', 'hybrid')),
                    return_sparse=(mode in ('sparse', 'hybrid')),
                    return_colbert_vecs=False,
                    verbose=False
                )
                
                if mode in ('cls', 'hybrid'):
                    dense_vecs = outputs['dense_vecs']
                    for k, vec in zip(uncached_keys, dense_vecs):
                        self.cache_dense_cls[k] = vec / (np.linalg.norm(vec) + 1e-9)
                
                if mode in ('sparse', 'hybrid'):
                    lexical_weights = outputs['lexical_weights']
                    for k, weights in zip(uncached_keys, lexical_weights):
                        self.cache_sparse[k] = weights

            if mode in ('mean', 'hybrid'):
                # Compute mean pooling manually in batches
                batch_size = 256
                all_mean = []
                for i in range(0, len(uncached_keys), batch_size):
                    batch = uncached_keys[i:i+batch_size]
                    inputs = self.model.tokenizer(
                        batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
                    ).to(device)
                    with torch.no_grad():
                        outputs = self.model.model.model(**inputs)
                    last_hidden_state = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    mean_embeddings = sum_embeddings / sum_mask
                    normalized_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=-1)
                    all_mean.append(normalized_embeddings.cpu().numpy())
                
                if all_mean:
                    mean_vecs = np.concatenate(all_mean, axis=0)
                    for k, vec in zip(uncached_keys, mean_vecs):
                        self.cache_dense_mean[k] = vec

    def embed(self, list_of_str, pooling=None):
        """Embed function matching the contriever API. Returns torch tensor."""
        if pooling is None:
            pooling = self.pooling
            
        if isinstance(list_of_str, str):
            list_of_str = [list_of_str]
        
        self._encode_keys(list_of_str, pooling)
        
        if pooling == 'cls':
            vecs = [self.cache_dense_cls[s] for s in list_of_str]
        else:
            vecs = [self.cache_dense_mean[s] for s in list_of_str]
            
        return torch.tensor(np.array(vecs), dtype=torch.float32, device=self.get_device())

    def search(
        self,
        key_strings,
        query_strings,
        topk: int=None,
        similarity_threshold: float=None,
        return_embeds=False,
        return_scores=False,
        pooling=None,  # 'cls' or 'mean' or 'sparse' or 'hybrid'
        hybrid_alpha=None # weight for dense, 1 - alpha for sparse
    ):
        if pooling is None:
            pooling = self.pooling
        if hybrid_alpha is None:
            hybrid_alpha = self.hybrid_alpha

        batch_request = True
        if isinstance(query_strings, str):
            batch_request = False
            query_strings = [query_strings]

        num_q = len(query_strings)
        num_keys = len(key_strings)

        # 1. Encode all keys and queries
        self._encode_keys(key_strings, pooling)
        self._encode_keys(query_strings, pooling)

        # 2. Compute similarity scores
        scores_matrix = np.zeros((num_q, num_keys))

        if pooling in ('cls', 'mean'):
            # Dense dot product
            if pooling == 'cls':
                query_dense = np.array([self.cache_dense_cls[q] for q in query_strings])
                key_dense = np.array([self.cache_dense_cls[k] for k in key_strings])
            else:
                query_dense = np.array([self.cache_dense_mean[q] for q in query_strings])
                key_dense = np.array([self.cache_dense_mean[k] for k in key_strings])
            scores_matrix = query_dense @ key_dense.T

        elif pooling == 'sparse':
            # Sparse lexical matching score
            for q_idx, q in enumerate(query_strings):
                q_sparse = self.cache_sparse[q]
                for k_idx, k in enumerate(key_strings):
                    k_sparse = self.cache_sparse[k]
                    scores_matrix[q_idx, k_idx] = self.model.compute_lexical_matching_score(q_sparse, k_sparse)

        elif pooling == 'hybrid':
            # Hybrid: Alpha * Dense(Mean) + (1 - Alpha) * Sparse
            # First get dense (mean) score
            query_dense = np.array([self.cache_dense_mean[q] for q in query_strings])
            key_dense = np.array([self.cache_dense_mean[k] for k in key_strings])
            dense_scores = query_dense @ key_dense.T

            # Next get sparse scores
            sparse_scores = np.zeros((num_q, num_keys))
            for q_idx, q in enumerate(query_strings):
                q_sparse = self.cache_sparse[q]
                for k_idx, k in enumerate(key_strings):
                    k_sparse = self.cache_sparse[k]
                    sparse_scores[q_idx, k_idx] = self.model.compute_lexical_matching_score(q_sparse, k_sparse)

            scores_matrix = hybrid_alpha * dense_scores + (1 - hybrid_alpha) * sparse_scores

        # 3. Format results matching original contriever.py
        selected_idx = []
        for q_idx in range(num_q):
            scores = scores_matrix[q_idx]
            if topk is not None:
                # Get topk indices
                sorted_idx = np.argsort(scores)[::-1]
                selected_idx.append(sorted_idx[:topk].tolist())
            elif similarity_threshold is not None:
                # Get indices above threshold
                idx_above = np.where(scores >= similarity_threshold)[0].tolist()
                selected_idx.append(idx_above)
            else:
                raise ValueError("You must specify either topk or similarity_threshold!")

        result = dict(idx=selected_idx)

        if return_embeds:
            # For compat with search_in_embeds, return dense embeds
            if pooling in ('cls', 'sparse'):
                result['embeds'] = [[self.cache_dense_cls[key_strings[k_id]] for k_id in selected_idx[q_idx]] for q_idx in range(num_q)]
            else:
                result['embeds'] = [[self.cache_dense_mean[key_strings[k_id]] for k_id in selected_idx[q_idx]] for q_idx in range(num_q)]
                
        if return_scores:
            result['scores'] = [[scores_matrix[q_idx, k_id] for k_id in selected_idx[q_idx]] for q_idx in range(num_q)]

        result['strings'] = [[key_strings[k_id] for k_id in selected_idx[q_idx]] for q_idx in range(num_q)]

        if not batch_request:
            result = {k: v[0] for k, v in result.items()}

        return result

    @staticmethod
    @torch.no_grad()
    def search_in_embeds(
        key_embeds,
        query_embeds,
        topk: int=None,
        similarity_threshold: float=None,
        return_embeds=False,
        return_scores=False,
    ):
        if int(topk is None) + int(similarity_threshold is None) != 1:
            raise ValueError("You should specify either topk or similarity_threshold but not both!")

        scores = query_embeds  @ key_embeds.T # shape: (num_keys,) or (num_queries, num_keys)
        batch_request = len(query_embeds.shape) > 1

        if not batch_request:
            scores = scores.reshape(1, -1) # shape: (num_queries, num_keys)
        num_q = scores.shape[0]

        if topk:
            sorted_idx = scores.argsort(-1, descending=True)  # sort for each query
            selected_idx = sorted_idx[:,:topk]
            selected_idx = selected_idx.tolist()
        else:
            selected_idx = [[] for i in range(num_q)]
            for (q_id, k_id) in (scores >= similarity_threshold).nonzero():
                selected_idx[q_id].append(k_id)

        result = dict(idx=selected_idx)

        if return_embeds:
            result['embeds'] = [
                 [key_embeds[k_id] for k_id in selected_idx[q_id]]
                 for q_id in range(num_q)
            ]
        if return_scores:
            result['scores'] = [
                 [scores[q_id, k_id] for k_id in selected_idx[q_id]]
                 for q_id in range(num_q)
            ]
        if not batch_request:
            result = {k: v[0] for k,v in result.items()}

        return result
