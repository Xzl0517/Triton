import torch
import triton
import triton.language as tl

@triton.jit
def _RMSNorm(X,OUT,stride,M,eps,BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(0)
    x_ptr = X + pid * stride
    out_ptr = OUT + pid * stride
    offs = tl.arange(0,BLOCK_SIZE)
    mask = offs < M
    x = tl.load(x_ptr + offs,mask)
    x_2 = x * x
    x_mean = tl.sum(x_2, axis=0) / M
    x_sqrt = tl.sqrt(x_mean)
    tl.store(out_ptr+offs,x/(x_sqrt+eps),mask)


def RMSNorm(X):
    N,M = X.shape
    out = torch.empty_like(X)
    BLOCK_SIZE = triton.next_power_of_2(M)
    eps = 1e-6
    _RMSNorm[(N,)](X,out,X.stride(0),M,eps,BLOCK_SIZE)
    return out


if __name__=="__main__":
    N,M = 32,100
    X = torch.randn((N,M),device='cuda')
    triton_out = RMSNorm(X)
    print(triton_out)
    RMSNorm_torch = torch.nn.RMSNorm(normalized_shape=M,eps=1e-6).cuda()
    torch_out = RMSNorm_torch(X)
    print(torch_out)
