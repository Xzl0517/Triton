import torch
import triton
import triton.language as tl

@triton.jit
def _conv(x,out,kernel,B,C,H,W,kernels,kernel_h,kernel_w,
          stride_xb,stride_xc,stride_xh,stride_xw,
          stride_ob,stride_oc,stride_oh,stride_ow,
          stride_ks,stride_kc,stride_kh,stride_kw,
          BLOCK_SIZE_H:tl.constexpr,
          BLOCK_SIZE_W:tl.constexpr,
          OUT_H:tl.constexpr,
          OUT_W:tl.constexpr
):
    bid = tl.program_id(0)
    kid = tl.program_id(1)
    row = tl.program_id(2)

    x_B_offs = bid * stride_xb
    
    o_B_offs = bid * stride_ob
    o_C_offs = kid * stride_oc
    o_R_offs = row * stride_oh

    k_B_offs = kid * stride_ks

    k_row_offs = tl.arange(0, BLOCK_SIZE_H)
    k_row_mask = k_row_offs[:,None] < kernel_h
    k_row_offs = k_row_offs[:,None] * stride_kh
    k_col_offs = tl.arange(0, BLOCK_SIZE_W)
    k_col_mask = k_col_offs[None,:] < kernel_w
    k_col_offs = k_col_offs[None,:] * stride_kw
    kernel_mask = k_row_mask & k_col_mask 

    for col in range(0,OUT_W):
        Y = 0.0

        x_row_offs = row * kernel_h + tl.arange(0, BLOCK_SIZE_H)
        x_row_mask = x_row_offs[:,None] < H
        x_row_offs = x_row_offs[:,None] * stride_xh

        x_col_offs = col * kernel_w + tl.arange(0, BLOCK_SIZE_W)
        x_col_mask = x_col_offs[None,:] < W
        x_col_offs = x_col_offs[None,:] * stride_xw
        x_mask =  x_row_mask & x_col_mask

        for c in range(0,C):
            x_offs =  x_B_offs + c* stride_xc + x_row_offs + x_col_offs
            X = tl.load(x + x_offs, x_mask)

            k_offs = k_B_offs + c * stride_kc + k_row_offs + k_col_offs
            Kernel = tl.load(kernel + k_offs,kernel_mask)

            Y += tl.sum( X * Kernel)
        o_offs = o_B_offs + o_C_offs + o_R_offs + col
        tl.store(out + o_offs,Y)


def Conv(x,kernel):
    B,C,H,W = x.shape
    kernels, C, kernel_h,kernel_w = kernel.shape
    assert H % kernel_h == 0 and W % kernel_w == 0
    out = torch.empty((B,kernels,H//kernel_h, W//kernel_w),device='cuda',dtype=torch.float32)
    grid = (B,kernels,H//kernel_h)
    BLOCK_SIZE_H = triton.next_power_of_2(kernel_h)
    BLOCK_SIZE_W = triton.next_power_of_2(kernel_w)
    OUT_H = H//kernel_h
    OUT_W = W//kernel_w
    _conv[grid](x,out,kernel,B,C,H,W,kernels, kernel_h,kernel_w,
                x.stride(0),x.stride(1),x.stride(2),x.stride(3),
                out.stride(0),out.stride(1),out.stride(2),out.stride(3),
                kernel.stride(0),kernel.stride(1),kernel.stride(2),kernel.stride(3),
                BLOCK_SIZE_H,BLOCK_SIZE_W,OUT_H,OUT_W)
    return out

if __name__=="__main__":
    B,C,H,W = 4,3,32,32
    kernels, kernel_h,kernel_w = 10, 4, 4
    x = torch.randint(0,10,(B,C,H,W),device='cuda',dtype=torch.float32)
    kernel = torch.randint(0,10,(kernels,C,kernel_h,kernel_w),device='cuda',dtype=torch.float32)
    
    conv_layer = torch.nn.Conv2d(
        in_channels= C,
        out_channels= kernels,
        kernel_size= (kernel_h,kernel_w),
        stride=(kernel_h,kernel_w),
        bias=False,
        dtype=torch.float32,
        device='cuda'
    )
    with torch.no_grad():
        conv_layer.weight.copy_(kernel)

    out_torch = conv_layer(x)
    out_triton = Conv(x,kernel)

    print(out_torch[0][0][0])
    print(out_triton[0][0][0])

    if torch.allclose(out_torch, out_triton):
        print('Data matches')
    else:
        print("Data no matchs")
