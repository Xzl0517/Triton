import torch
import triton
import triton.language as tl


@triton.jit
def _matmul(X,Y,OUT, 
            M,N,K,
            stride_X1,stride_X2,
            stride_Y1,stride_Y2,
            stride_OUT1,stride_OUT2,
            BLOCK_SIZE_M:tl.constexpr,
            BLOCK_SIZE_N:tl.constexpr,
            BLOCK_SIZE_K:tl.constexpr):
    pid = tl.program_id(0)
    m = tl.cdiv(M,BLOCK_SIZE_M)
    n = tl.cdiv(N,BLOCK_SIZE_N)
    row = pid // m # X OUT 的row行
    col = pid % n # Y OUT 的col列
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_x = row * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)
    off_y = col * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)
    x_ptr = X + (off_x[:,None]*stride_X1 + off_k[None,:]*stride_X2)
    y_ptr = Y + (off_k[:,None]*stride_Y1 + off_y[None,:]*stride_Y2)

    x = tl.load(x_ptr)
    y = tl.load(y_ptr)
    result = tl.zeros([BLOCK_SIZE_M,BLOCK_SIZE_N],dtype=tl.float32)
    result = tl.dot(x,y,result)

    c_ptr = OUT + off_x[:,None] * stride_OUT1 + off_y[None,:] * stride_OUT2
    c_mask = (off_x[:, None] < M) & (off_y[None, :] < N)

    tl.store(c_ptr, result,c_mask)



    

def matmul(x,y):
    M,N,K = x.shape[0],y.shape[1],x.shape[1]
    out = torch.empty((M,N),device='cuda',dtype=torch.float32)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = x.shape[1]
    m = triton.cdiv(M,BLOCK_SIZE_M)
    n = triton.cdiv(N,BLOCK_SIZE_N)
    # print(M,N,K)
    # print(x.stride(0),x.stride(1))
    # print(y.stride(0),y.stride(1))
    # print(out.stride(0),out.stride(1))
    # print(BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K)
    # print(m,n)
    _matmul[(n*m,)](x,y,out,
                    M,N,K,
                    x.stride(0),x.stride(1),
                    y.stride(0),y.stride(1),
                    out.stride(0),out.stride(1), 
                    BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K)
    return out


if __name__=="__main__":
    N, k = 64,16
    k, M = 16,64
    # x = torch.randn((N,k),device='cuda',dtype=torch.float32)
    # y = torch.randn((k,M),device='cuda',dtype=torch.float32)
    x = torch.randint(low=0,high=10,size=(N,k),device='cuda').float()
    y = torch.randint(low=0,high=10,size=(k,M),device='cuda').float()
    # print(x)
    # print(y)
    torch_out = x @ y
    triton_out = matmul(x,y)
    print(torch_out)
    print(triton_out)
