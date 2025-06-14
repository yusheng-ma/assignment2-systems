import math
import torch
import einx
from torch import Tensor
from jaxtyping import Float
from einops import einsum, rearrange
from cs336_basics.nn_utils import softmax

B_QUERY = 16
B_KEY = 16

def pad_and_rearrange(x: Tensor, block: int, name="") -> tuple[Tensor, int, int]:
    *_, seq_len, dim = x.shape
    tile = math.ceil(seq_len / block)
    pad_len = tile * block - seq_len
    if pad_len > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
    x = rearrange(x, "... (tile B) dim -> ... tile B dim", tile=tile)
    return x, tile, pad_len


class MyFlashAttentionAutogradFunctionClass(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "... n_queries dim_key"],
        K: Float[Tensor, "... n_keys dim_key"],
        V: Float[Tensor, "... n_keys dim_val"],
        is_causal=False
    ) -> Float[Tensor, "... n_queries dim_val"]: # -> output O
        *batch_shape, n_queries, dim_key = Q.shape
        device = Q.device
        dim_val = V.shape[-1]
        Q_tile, tile_query, pad_q = pad_and_rearrange(Q, B_QUERY, "query")
        K_tile, tile_key, pad_k = pad_and_rearrange(K, B_KEY, "key")
        V_tile, _, _ = pad_and_rearrange(V, B_KEY, "value")

        Oi_list = []
        Li_list = []

        for i in range(tile_query):
            Oi = torch.zeros(*batch_shape, B_QUERY, dim_val, device=device)
            li = torch.zeros(*batch_shape, B_QUERY, device=device)
            mi = torch.full((*batch_shape, B_QUERY), float("-inf"), device=device)

            for j in range(tile_key):
                Sij = einsum(Q_tile[..., i, :, :], K_tile[..., j, :, :],
                             "... B_QUERY dim_key, ... B_KEY dim_key -> ... B_QUERY B_KEY") / math.sqrt(dim_key)
                
                if is_causal:
                    #  masked out, add the constant value of -1e6 to the corresponding elements of the attention score matrix S(j)
                    seq_qi = torch.arange(B_QUERY * i, B_QUERY * i + B_QUERY, device=device)
                    seq_kj = torch.arange(B_KEY * j, B_KEY * j + B_KEY, device=device)
                    
                    qi = einx.rearrange('query -> query 1', seq_qi)   # (B_QUERY, 1)
                    kj = einx.rearrange('key -> 1 key', seq_kj)       # (1, B_KEY)

                    causal_mask = qi >= kj                            # (B_QUERY, B_KEY)
                    Sij = torch.where(causal_mask, Sij, float("-inf"))

                    # seq = torch.arange(sequence_length, device=x.device)
                    # qi = einx.rearrange('query -> b... 1 query 1', seq, b=[1] * len(b))
                    # kj = einx.rearrange('key   -> b... 1 1   key', seq, b=[1] * len(b))
                    # causal_mask = qi >= kj  # (query, key)
                    # attention_scores = torch.where(mask, attention_scores, float("-inf"))

                # --- Assert shapes ---
                assert Sij.shape[-2:] == (B_QUERY, B_KEY), f"Unexpected shape for Sij: {Sij.shape}"
                assert mi.shape[-1] == B_QUERY, f"Unexpected shape for mi: {mi.shape}"

                mij = torch.maximum(mi, torch.max(Sij, dim=-1).values)
                assert mij.shape == mi.shape, f"mij shape mismatch: {mij.shape} != {mi.shape}"

                Pij_hat = torch.exp(Sij - mij[..., None])
                assert Pij_hat.shape[-2:] == (B_QUERY, B_KEY), f"Pij_hat shape mismatch: {Pij_hat.shape}"

                lij = torch.exp(mi - mij) * li + torch.sum(Pij_hat, dim=-1)
                assert lij.shape == li.shape, f"lij shape mismatch: {lij.shape} != {li.shape}"

                diag = einsum(torch.exp(mi - mij), Oi,
                              "... B_QUERY, ... B_QUERY dim_val -> ... B_QUERY dim_val")
                assert diag.shape == Oi.shape, f"diag shape mismatch: {diag.shape} != {Oi.shape}"

                Oij = diag + einsum(Pij_hat, V_tile[..., j, :, :],
                                    "... B_QUERY B_KEY, ... B_KEY dim_val -> ... B_QUERY dim_val")
                assert Oij.shape == Oi.shape, f"Oij shape mismatch: {Oij.shape} != {Oi.shape}"

                li = lij
                mi = mij
                Oi = Oij

            Oi = einsum(1.0 / li, Oi, "... B_QUERY , ... B_QUERY dim_val -> ... B_QUERY dim_val")
            Li = mi + torch.log(li)

            Oi_list.append(Oi)
            Li_list.append(Li)

        O = rearrange(Oi_list, "tile_query ... B_QUERY dim_val -> ... (tile_query B_QUERY) dim_val")
        L = rearrange(Li_list, "tile_query ... B_QUERY -> ... (tile_query B_QUERY)")
        # Remove padding if any
        if pad_q > 0:
            O = O[..., :-pad_q, :]
            L = L[..., :-pad_q]

        ctx.save_for_backward(O, Q, K, V, L)
        return O

    @staticmethod
    def backward(ctx):
        pass
