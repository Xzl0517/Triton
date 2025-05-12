import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x,out,stride,N,BLOCK_SIZE:tl.constexpr):
    row = tl.program_id(0)
    x += row * stride
    out += row * stride
    offs = tl.arange(0,BLOCK_SIZE)
    X = tl.load(x+offs,mask= offs < N,other=-float('inf'))
    X =  X - tl.max(X,axis=0)
    X = tl.exp(X)
    sum = tl.sum(X,axis=0)
    tl.store(out+offs,X/sum,mask=offs<N)

def softmax(x):
    N, M = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(M)
    # print(BLOCK_SIZE)
    # print(x.stride(0))
    softmax_kernel[(N,)](x,out,x.stride(0),M,BLOCK_SIZE)
    return out


if __name__=="__main__":
    N,M = 32,5
    x = torch.ones((N,M),device='cuda').float()
    # print(x)
    # print(x)
    softmax_torch = torch.softmax(x,dim=1)
    print(softmax_torch[1,:])
    print(softmax(x)[1,:])

    