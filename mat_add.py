import torch
import triton
import triton.language as tl

@triton.jit
def matadd_kernel(a_ptr,b_ptr,c_ptr,stride,BLOCK_SIZE:tl.constexpr):

    row = tl.program_id(axis=0)
    a_ptr += row * stride
    b_ptr += row * stride
    c_ptr += row * stride
    a_offsets = 0 + tl.arange(0, BLOCK_SIZE)
    b_offsets = 0 + tl.arange(0, BLOCK_SIZE)
    c_offsets = 0 + tl.arange(0, BLOCK_SIZE)
    
    x = tl.load(a_ptr + a_offsets,mask=a_offsets<stride)
    y = tl.load(b_ptr + b_offsets,mask=a_offsets<stride)
    c = x + y 
    tl.store(c_ptr + c_offsets, c,mask=a_offsets<stride)



def matadd(x,y):
    n,m = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(m)
    matadd_kernel[(n,)](x,y,out,x.stride(0),BLOCK_SIZE)

    return out



if __name__=="__main__":
    n,m = 32,100
    x = torch.rand((n,m),device='cuda',dtype=torch.float16)
    y = torch.rand((n,m),device='cuda',dtype=torch.float16)
    # print(x)
    # print(y)
    print(x+y)
    out = matadd(x,y)
    print(out)

