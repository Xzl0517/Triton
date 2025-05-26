import torch
import triton
import triton.language as tl
import math

@triton.jit
def _ReLU(x,out,stride,BLOCK_SIZE:tl.constexpr):
    row = tl.program_id(0)
    x += row * stride
    out += row * stride
    offs = tl.arange(0,BLOCK_SIZE)
    mask = offs < stride
    X = tl.load(x + offs, mask)
    X = tl.maximum(X,0)
    tl.store(out + offs, X, mask)

def Relu(x):
    M,N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    _ReLU[grid](x,out,x.stride(0),BLOCK_SIZE)
    return out

@triton.jit
def _leaky_relu(x,out,alpha,stride,BLOCK_SIZE:tl.constexpr):
    row = tl.program_id(0)
    x += row * stride
    out += row * stride
    offs = tl.arange(0,BLOCK_SIZE)
    mask = offs < stride

    X = tl.load(x + offs, mask)
    X = tl.where(X>=0,X,alpha * X)
    tl.store(out + offs,X,mask)

def leaky_relu(x,alpha):
    M,N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    _leaky_relu[grid](x,out,alpha,x.stride(0),BLOCK_SIZE)
    return out

@triton.jit
def _sigmoid(x,out,stride,BLOCK_SIZE:tl.constexpr):
    row = tl.program_id(0)
    x += row * stride
    out += row * stride
    offs = tl.arange(0,BLOCK_SIZE)
    mask = offs < stride
    X = tl.load(x + offs,mask)
    X = 1.0 / (1.0 + tl.exp(-X))
    tl.store(out + offs,X, mask)

def sigmoid(x):
    M,N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    _sigmoid[grid](x,out,x.stride(0),BLOCK_SIZE)
    return out


# Tanh
@triton.jit
def _tanh(x,out,stride,BLOCK_SIZE:tl.constexpr):
    row = tl.program_id(0)
    x += row * stride
    out += row * stride
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < stride
    X = tl.load(x + offs,mask)
    X_p = tl.exp(X)
    X_n = tl.exp(-X)
    Y = (X_p - X_n) / (X_p + X_n)
    tl.store(out+offs,Y,mask)

def Tanh(x):
    M,N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    _tanh[grid](x,out,x.stride(0),BLOCK_SIZE)
    return out

# Gelu
@triton.jit
def _gelu(x,out,stride,BLOCK_SIZE:tl.constexpr):
    row = tl.program_id(0)
    x += row * stride
    out += row * stride
    offs = tl.arange(0,BLOCK_SIZE)
    mask = offs < stride
    X = tl.load(x + offs,mask)
    tanh_x = tl.sqrt(2 / math.pi) * (X + 0.044715 * X * X * X)
    X_p = tl.exp(tanh_x)
    X_n = tl.exp(-tanh_x)
    Y = 0.5 * X * (1.0 + (X_p - X_n) / (X_p + X_n))
    tl.store(out + offs, Y ,mask)

def Gelu(x):
    M,N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    _gelu[grid](x,out,x.stride(0),BLOCK_SIZE)
    return out    
# Silu
@triton.jit
def _silu(x,out,stride,BLOCK_SIZE:tl.constexpr):
    row = tl.program_id(0)
    x += row * stride
    out += row * stride
    offs = tl.arange(0,BLOCK_SIZE)
    mask = offs < stride
    X = tl.load(x + offs,mask)
    Y = X / (1 + tl.exp(-X))
    tl.store(out + offs,Y,mask)

def Silu(x):
    M,N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    _silu[grid](x,out,x.stride(0),BLOCK_SIZE)
    return out 


if __name__=="__main__":
    M,N = 8,6
    x = torch.randn((M,N),device='cuda',dtype=torch.float32)
    print(x)

    # Relu
    # triton_out = Relu(x)
    # Relu_torch = torch.nn.ReLU()
    # torch_out = Relu_torch(x)
    # print(triton_out)
    # print(torch_out)


    # Leaky_relu
    # triton_out = leaky_relu(x,0.01)
    # LeakyRelu_torch = torch.nn.LeakyReLU()
    # torch_out = LeakyRelu_torch(x)
    # print(triton_out)
    # print(torch_out)


    # Sigmoid
    # triton_out = sigmoid(x)
    # Sigmoid_torch = torch.nn.Sigmoid()
    # torch_out = Sigmoid_torch(x)
    # print(triton_out)
    # print(torch_out)

    # Tanh
    # triton_out = Tanh(x)
    # Tanh_torch = torch.nn.Tanh()
    # torch_out = Tanh_torch(x)
    # print(triton_out)
    # print(torch_out)

    # GeLU
    # triton_out = Gelu(x)
    # Gelu_torch = torch.nn.GELU()
    # torch_out = Gelu_torch(x)
    # print(triton_out)
    # print(torch_out)

    # SiLU
    # triton_out = Silu(x)
    # Silu_torch = torch.nn.SiLU()
    # torch_out = Silu_torch(x)
    # print(triton_out)
    # print(torch_out)
