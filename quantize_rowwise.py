import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_int8_quantize(X,Y,OUT, 
            M,N,K,
            factor,
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
    f_ptr = factor + off_k


    x = tl.load(x_ptr) # fp32
    y = tl.load(y_ptr) # int8
    f = tl.load(f_ptr) # 取缩放因子 
    y = (y / 127.) * f 
    result = tl.zeros([BLOCK_SIZE_M,BLOCK_SIZE_N],dtype=tl.float32)
    result = tl.dot(x,y,acc=result)

    c_ptr = OUT + off_x[:,None] * stride_OUT1 + off_y[None,:] * stride_OUT2
    c_mask = (off_x[:, None] < M) & (off_y[None, :] < N)

    tl.store(c_ptr, result,c_mask)

def matmul_int8_quantize(x,y,factor):
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
    _matmul_int8_quantize[(n*m,)](x,y,out,
                    M,N,K,
                    factor,
                    x.stride(0),x.stride(1),
                    y.stride(0),y.stride(1),
                    out.stride(0),out.stride(1), 
                    BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K)
    return out

@triton.jit
def _quantize_rowwise(
    x_ptr,
    out_ptr,
    factor_ptr,
    N,
    stride,
    BLOCK_SIZE:tl.constexpr
):
    row = tl.program_id(0)
    x_ptr += row * stride
    out_ptr += row * stride
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask)

    max_x = tl.max(x, axis=0)
    min_x = tl.min(x, axis=0)
    factor = max_x - min_x
    output = tl.ceil(127. * x / factor)
    
    tl.store(out_ptr + offs, output, mask)
    tl.store(factor_ptr + row, factor)

def quantize_rowwise(x):
    M,N = x.shape
    BlOCK_SIZE = triton.next_power_of_2(N)
    out = torch.empty_like(x,device=x.device,dtype=torch.int8)
    factor = torch.empty(M,device=x.device,dtype=torch.float32)
    _quantize_rowwise[(M,)](x,out,factor,N,x.stride(0),BlOCK_SIZE)
    return out,factor


if __name__ == "__main__":
    M,N = 60,32
    x = torch.randn((M,N),device='cuda') # input embedding 
    w = torch.randn((N,M),device='cuda') # weigt matrix
    out_w,factor = quantize_rowwise(w)
    print(out_w)
    print(factor)

    triton_out = matmul_int8_quantize(x,out_w,factor)
    torch_out = x @ w
    print(torch_out)
    print(triton_out)
    print(torch.sum(abs(abs(triton_out)-abs(torch_out))))