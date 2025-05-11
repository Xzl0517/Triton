import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(X,Y,OUT,N,BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0,BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X+offs,mask=mask)
    y = tl.load(Y+offs,mask=mask)
    out = x + y
    tl.store(OUT + offs,out,mask=mask)


def vec_add(x,y):
    out = torch.empty_like(x)
    N = x.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(N) // 4
    M = N // BLOCK_SIZE
    vector_add_kernel[(M,)](x,y,out,N,BLOCK_SIZE)
    return out



if __name__=="__main__":
    N = 128
    x = torch.randn(N,device='cuda')
    y = torch.randn(N,device='cuda')
    # print(x.shape[0])
    print((x+y)[0:10])
    out = vec_add(x,y)
    print(out[0:10])
