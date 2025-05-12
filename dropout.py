import torch
import triton
import triton.language as tl

@triton.jit
def _dropout(X,OUT,P,SEED,N,BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0,BLOCK_SIZE)

    x = tl.load(X+offs,mask=offs<N)
    random = tl.rand(SEED,offs)
    x_mask = random > P

    out = tl.where(x_mask,x/(1-P),0.0)
    tl.store(OUT+offs,out,mask=offs<N)
    


def dropout(x,p,seed):
    out = torch.empty_like(x)
    N = x.shape[0]
    BLOCK_SIZE = 128
    M = triton.cdiv(N,BLOCK_SIZE)
    _dropout[(M,)](x,out,p,seed,N,BLOCK_SIZE)
    return out



if __name__=="__main__":
    N = 16
    x = torch.randn((N),device='cuda')
    print(x)
    p = 0.5
    seed = 1
    print(dropout(x,p,seed))
