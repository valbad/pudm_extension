from torch import nn
from torch.autograd import Function
import torch
import importlib
import os
chamfer_found = importlib.util.find_spec("chamfer_3D") is not None
if not chamfer_found:
    print("Jitting Chamfer 3D")

    from torch.utils.cpp_extension import load
    _dir = os.path.dirname(os.path.abspath(__file__))
    chamfer_3D = load(name="chamfer_3D",
          sources=[
              os.path.join(_dir, "chamfer_cuda.cpp"),
              os.path.join(_dir, "chamfer3D.cu"),
          ])
    #print("Loaded JIT 3D CUDA chamfer distance")

else:
    import chamfer_3D
    #print("Loaded compiled 3D CUDA chamfer distance")


# Chamfer's distance module @thibaultgroueix
# GPU tensors only
class chamfer_3DFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        device = xyz1.device

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.to(device)
        dist2 = dist2.to(device)
        idx1 = idx1.to(device)
        idx2 = idx2.to(device)
        torch.cuda.set_device(device)

        chamfer_3D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        device = graddist1.device

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.to(device)
        gradxyz2 = gradxyz2.to(device)
        chamfer_3D.backward(
            xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
        )
        return gradxyz1, gradxyz2


class chamfer_3DDist(nn.Module):
    def __init__(self):
        super(chamfer_3DDist, self).__init__()

    def forward(self, input1, input2):
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        return chamfer_3DFunction.apply(input1, input2)

def hausdorff_distance(X,Y):
    '''
     the HD is from MPU
    Parameters
    ----------
    X
    Y

    Returns
    -------

    '''
    B,N,C = X.shape
    dist1, dist2 ,_ ,_ = chamfer_3DFunction.apply(X, Y)
    # dist1/dist2 are squared L2 distances; take sqrt for actual L2
    h1 = torch.amax(torch.sqrt(dist1), dim=1)  # (B,)
    h2 = torch.amax(torch.sqrt(dist2), dim=1)  # (B,)
    hd_loss = torch.maximum(h1, h2)  # true Hausdorff: max of both directions
    return hd_loss



if __name__ == '__main__':

    x = torch.zeros(size=(2,1024,3)).cuda()
    y = torch.zeros(size=(2,1024,3)).cuda()

    cd = chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cd(x,y)

    print(dist1.shape)
    print(dist2.shape)
