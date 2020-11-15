from hcat.r_unet import StackedDilation, RDCNet
import torch



sd = RDCNet(in_channels=3, out_channels=15).cuda().eval()

x = torch.rand((1,3,512,512,25)).cuda()
out = sd(x)


for x in [510,512]:
    for y in [510, 512]:
        for z in [20,22,24]:
            with torch.no_grad():
                in_ = torch.rand((1,3,x,y,z)).cuda()
                out = sd(in_)
            print(x,y,z, out.shape)
            assert out.shape[2] == x
            assert out.shape[3] == y
            assert out.shape[4] == z


