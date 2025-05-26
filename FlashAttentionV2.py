import torch
import triton
import triton.language as tl
# import os
# os.environ["TRITON_INTERPRET"] = "1"

@triton.jit
def fa_kernerl(q,k,v,o,B,H,N,d,qk_scala,
               stride_qb,stride_qh,stride_qn,stride_qd,
               stride_kb,stride_kh,stride_kn,stride_kd,
               stride_vb,stride_vh,stride_vn,stride_vd,
               stride_ob,stride_oh,stride_on,stride_od,
               BLOCK_SIZE_S:tl.constexpr,
               BLOCK_DMODEL:tl.constexpr
):
    # 每个线程需要计算 N x d 个元素
    start_B = tl.program_id(0)
    start_H = tl.program_id(1)
    q_ptr = q + start_B * stride_qb + start_H * stride_qh
    k_ptr = k + start_B * stride_kb + start_H * stride_kh
    v_ptr = v + start_B * stride_vb + start_H * stride_vh
    o_ptr = o + start_B * stride_ob + start_H * stride_oh

    for block_start_qo in range(0, N, BLOCK_SIZE_S):
        # load q
        Q_block_ptr = tl.make_block_ptr(
            base = q_ptr,
            shape=(N, d),
            strides=(stride_qn, stride_qd),
            offsets=(block_start_qo * BLOCK_SIZE_S, 0),
            block_shape=(BLOCK_SIZE_S, BLOCK_DMODEL),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base = o_ptr ,
            shape=(N, d),
            strides=(stride_on, stride_od),
            offsets=(block_start_qo * BLOCK_SIZE_S, 0),
            block_shape=(BLOCK_SIZE_S, BLOCK_DMODEL),
            order=(1, 0),
        )
        Q = tl.load(Q_block_ptr, boundary_check=(0, 1) )
        O = tl.load(O_block_ptr, boundary_check=(0, 1) )
        mi = tl.zeros([BLOCK_SIZE_S],dtype=tl.float32) - float('inf') # 每行最大值
        li = tl.zeros([BLOCK_SIZE_S],dtype=tl.float32) # 每行分母
        # 遍历KV 小块
        for block_start_kv in range(0,N,BLOCK_SIZE_S):
            # load kv
            K_block_ptr = tl.make_block_ptr(
                base = k_ptr ,
                shape=(N, d),
                strides=(stride_kn, stride_kd),
                offsets=(block_start_kv * BLOCK_SIZE_S, 0),
                block_shape=(BLOCK_DMODEL, BLOCK_SIZE_S),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base = v_ptr,
                shape=(N, d),
                strides=(stride_vn, stride_vd),
                offsets=(block_start_kv * BLOCK_SIZE_S, 0),
                block_shape=(BLOCK_SIZE_S, BLOCK_DMODEL),
                order=(1, 0),
            )
            K = tl.load(K_block_ptr, boundary_check=(0, 1))
            V = tl.load(V_block_ptr, boundary_check=(0, 1))
            QK = tl.zeros([BLOCK_SIZE_S,BLOCK_SIZE_S],dtype=tl.float32)
            QK += tl.dot(Q,K)
            QK = QK * qk_scala
            mi_new = tl.maximum(mi,tl.max(QK, axis=1))
            QK -= mi_new[:,None]
            p = tl.exp(QK)
            lij = tl.sum(p, axis=1)
            alpha = tl.exp(mi - mi_new)
            li = li * alpha + lij
            O = O * alpha[:, None]
            O = tl.dot(p,V, acc = O)
            mi = mi_new
        O = O / li[:,None]
        tl.store(O_block_ptr,O, boundary_check=(0, 1))

def flashAttention(q,k,v):
    out = torch.empty_like(q)
    B,H,N,d = 4,8,16,32
    BLOCK_SIZE_S = 16
    BLOCK_DMODEL = d
    qk_scala = 1.0 / d ** 0.5
    grid = (B, H, 1)
    fa_kernerl[grid](q,k,v,out,B,H,N,d,qk_scala,
                     q.stride(0),q.stride(1),q.stride(2),q.stride(3),
                     k.stride(0),k.stride(1),k.stride(2),k.stride(3),
                     v.stride(0),v.stride(1),v.stride(2),v.stride(3),
                     out.stride(0),out.stride(1),out.stride(2),out.stride(3),
                     BLOCK_SIZE_S,BLOCK_DMODEL)
    return out


if __name__=="__main__":
    # batch num_head seq_len head_dim
    B,H,N,d = 4,8,16,32
    q = torch.randn((B,H,N,d),device='cuda',dtype=torch.float32)
    k = torch.randn((B,H,N,d),device='cuda',dtype=torch.float32)
    v = torch.randn((B,H,N,d),device='cuda',dtype=torch.float32)
    out = flashAttention(q,k,v)
    print(out.shape)
