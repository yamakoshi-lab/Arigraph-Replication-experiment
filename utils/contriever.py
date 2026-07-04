import os
import torch
import numpy as np
from FlagEmbedding import BGEM3FlagModel

class Retriever:
    def __init__(self, device="cpu", **kwargs):
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    def embed(self, list_of_str):
        if isinstance(list_of_str, str):
            list_of_str = [list_of_str]
        # BGE-M3の3種のベクトルを同時出力（長文対応 8192）
        embeddings = self.model.encode(list_of_str, batch_size=12, max_length=8192, 
                                       return_dense=True, return_sparse=True, return_colbert_vecs=True)
        return embeddings

    @staticmethod
    def concat_embeds(embed_list):
        if not embed_list:
            return None
        dense_vecs = np.concatenate([emb['dense_vecs'] for emb in embed_list], axis=0)
        lexical_weights = []
        for emb in embed_list:
            lexical_weights.extend(emb['lexical_weights'])
        colbert_vecs = []
        for emb in embed_list:
            colbert_vecs.extend(emb['colbert_vecs'])
        return {
            'dense_vecs': dense_vecs,
            'lexical_weights': lexical_weights,
            'colbert_vecs': colbert_vecs
        }

    def search(self, key_strings, query_strings, topk=1, return_embeds=False, return_scores=False):
        batch_request = True
        if isinstance(query_strings, str):
            batch_request = False
            query_strings = [query_strings]

        num_q = len(query_strings)
        key_embeds = self.embed(key_strings)
        query_embeds = self.embed(query_strings)
        result = self.search_in_embeds(key_embeds, query_embeds, topk, return_embeds, return_scores)
        
        result['strings'] = [[key_strings[k_id] for k_id in result['idx'][q_id]] for q_id in range(num_q)]
        if not batch_request:
            result = {k: v[0] for k, v in result.items()}
            
        return result

    def search_in_embeds(self, key_embeds, query_embeds, topk=1, return_embeds=False, return_scores=False, similarity_threshold=None):
        if (topk is None and similarity_threshold is None) or (topk is not None and similarity_threshold is not None):
            raise ValueError("You should specify either topk or similarity_threshold but not both!")

        dense_q = torch.tensor(query_embeds['dense_vecs'])
        dense_k = torch.tensor(key_embeds['dense_vecs'])
        dense_scores = dense_q @ dense_k.T

        num_q = dense_scores.shape[0]
        num_k = dense_scores.shape[1]
        
        hybrid_scores = torch.zeros((num_q, num_k))

        mode = os.environ.get("RETRIEVAL_MODE", "dense")

        if mode == "hybrid":
            # ハイブリッドスコアの合成
            for i in range(num_q):
                for j in range(num_k):
                    d_score = dense_scores[i, j].item()
                    l_score = self.model.compute_lexical_matching_score(
                        query_embeds['lexical_weights'][i],
                        key_embeds['lexical_weights'][j]
                    )
                    c_score = self.model.colbert_score(
                        query_embeds['colbert_vecs'][i],
                        key_embeds['colbert_vecs'][j]
                    ).item()

                    h_score = (d_score * 0.4) + (l_score * 0.2) + (c_score * 0.4)
                    hybrid_scores[i, j] = float(h_score)
            scores = hybrid_scores
        else:
            # Dense単独（exp-001で検証済み。200問でF1 0.4072/EM 0.335、
            # ハイブリッド(F1 0.3613/EM 0.245)を上回った）
            scores = dense_scores

        batch_request = len(dense_q.shape) > 1
        
        if topk is not None:
            topk_vals, selected_idx = torch.topk(scores, min(topk, num_k), dim=-1)
        else:
            selected_idx = [torch.where(scores[i] > similarity_threshold)[0] for i in range(num_q)]

        if not batch_request:
            selected_idx = [selected_idx]

        result = dict(idx=selected_idx)
        if return_embeds:
            result['embeds'] = [[key_embeds['dense_vecs'][k_id] for k_id in selected_idx[q_id]] for q_id in range(num_q)]
        if return_scores:
            result['scores'] = [[scores[q_id, k_id].item() for k_id in selected_idx[q_id]] for q_id in range(num_q)]
            
        return result