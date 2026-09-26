import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class GPUEmbeddingBlocker:

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 256,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load a compact multilingual Apache 2.0 / MIT model
        self.model = SentenceTransformer(model_name, device=self.device)
        # Enable FP16 (half-precision) to halve memory and leverage Tensor Cores
        if self.device == "cuda":
            self.model.half()
        self.batch_size = batch_size

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encodes texts into L2-normalized FP16 tensors on GPU."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=True,  # Allows dot-product to equal cosine similarity
        )
        return embeddings

    def find_top_k(
        self,
        s1_ids: list[str],
        pool_ids: list[str],
        s1_texts: list[str],
        pool_texts: list[str],
        top_k: int = 20,
        min_cosine: float = 0.60,
    ) -> dict[str, list[str]]:
        """Computes chunked top-k cosine similarity on GPU."""
        candidates = {s1_id: [] for s1_id in s1_ids}
        if not s1_ids or not pool_ids:
            return candidates

        # 1. Encode both sets on GPU
        with torch.no_grad():
            s1_embs = self.encode_texts(s1_texts)  # Shape: (N, D)
            pool_embs = self.encode_texts(pool_texts)  # Shape: (M, D)

            pool_ids_arr = np.array(pool_ids)

            # 2. Chunked GPU matrix multiplication to stay under RTX 3050 VRAM limit
            chunk_size = 2048
            for i in range(0, s1_embs.shape[0], chunk_size):
                s1_chunk = s1_embs[i : i + chunk_size]

                # Cosine sim via matrix multiplication (N_chunk, M)
                sim_matrix = torch.matmul(s1_chunk, pool_embs.T)

                # Retrieve top-k on GPU
                top_vals, top_indices = torch.topk(
                    sim_matrix, k=min(top_k, pool_embs.shape[0]), dim=1
                )

                top_vals_cpu = top_vals.cpu().numpy()
                top_indices_cpu = top_indices.cpu().numpy()

                for row_idx in range(len(s1_chunk)):
                    global_s1_id = s1_ids[i + row_idx]
                    mask = top_vals_cpu[row_idx] >= min_cosine
                    valid_cands = pool_ids_arr[top_indices_cpu[row_idx][mask]]
                    candidates[global_s1_id] = valid_cands.tolist()

            # Clean VRAM cache
            del s1_embs, pool_embs
            torch.cuda.empty_cache()

        return candidates