import math
import torch
import triton
import triton.language as tl
from torch import Tensor
from jaxtyping import Float


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd, # stride: batch, (query or key), dim
    stride_kb, stride_kq, stride_kd,
    stride_vb, stride_vq, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # mutliplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb, # global info
        shape=(N_QUERIES, D), # global info
        strides=(stride_qq, stride_qd), # global info
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # local info
        block_shape=(Q_TILE_SIZE, D), # local info
        order=(1, 0), # local info
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kq, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vq, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # this tile i
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

    Q_tilei = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    # for all tile j
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_tilej = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_tilej = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        Sij = tl.dot(Q_tilei, tl.trans(K_tilej)) * scale

        if is_causal:
            seq_qi = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            seq_kj = K_TILE_SIZE * j + tl.arange(0, K_TILE_SIZE)

            qi = seq_qi[:, None]
            kj = seq_kj[None, :]

            causal_mask = qi >= kj
            Sij = tl.where(causal_mask, Sij, -1e6)

        mij = tl.maximum(mi, tl.max(Sij, axis=-1))

        Pij_hat = tl.exp(Sij - mij[:, None])

        lij = tl.exp(mi - mij) * li + tl.sum(Pij_hat, axis=-1)

        diag = tl.exp(mi - mij)[:, None] * Oi

        Pij_hat_cast = Pij_hat.to(V_tilej.dtype)
        Oij = tl.dot(Pij_hat_cast, V_tilej, acc=diag)

        li = lij
        mi = mij
        Oi = Oij

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    Oi = (1.0 / li)[:, None] * Oi
    Li = mi + tl.log(li)

    Oi_cast = Oi.to(O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, Oi_cast, boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))


class MyTritonFlashAttentionAutogradFunctionClass(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "batch_size n_queries dim_key"],
        K: Float[Tensor, "batch_size n_keys dim_key"],
        V: Float[Tensor, "batch_size n_keys dim_val"],
        is_causal=False
    ) -> Float[Tensor, "batch_size n_queries dim_val"]: # -> output O
        batch_size, n_queries, dim_key = Q.shape
        _, n_keys, _ = V.shape
        device = Q.device

        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        ctx.is_causal = is_causal

        O = torch.empty(batch_size, n_queries, dim_key, dtype=torch.float32, device=device)
        L = torch.empty(batch_size, n_queries, dtype=torch.float32, device=device)

        flash_fwd_kernel[(math.ceil(n_queries / ctx.Q_TILE_SIZE), batch_size)](
            Q, K, V,
            O, L,
            Q.stride(-3), Q.stride(-2), Q.stride(-1),
            K.stride(-3), K.stride(-2), K.stride(-1),
            V.stride(-3), V.stride(-2), V.stride(-1),
            O.stride(-3), O.stride(-2), O.stride(-1),
            L.stride(-2), L.stride(-1),
            N_QUERIES=n_queries, N_KEYS=n_keys,
            scale = 1 / math.sqrt(dim_key),
            D=dim_key,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=ctx.is_causal,
        )

        ctx.save_for_backward(O, Q, K, V, L)
        return O

    @staticmethod
    def backward(ctx):
        pass
