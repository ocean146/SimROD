import torch
import torch.nn as nn
import torch.nn.functional as F

def tanh_range(l=0.5, r=2.0):
    def get_activation(left, right):
        def activation(x):
            return (torch.tanh(x) * 0.5 + 0.5) * (right - left) + left
        return activation
    return get_activation(l, r)

class SimpleRGGBGamma(nn.Module):
    def __init__(self, gamma_range=[1.,4.]) -> None:
        super().__init__()
        self.gamma_range = gamma_range
        self.bias = nn.Parameter(torch.tensor([0., 0., 0., 0.]).unsqueeze(0))

    def apply_gamma(self, img, IA_params):
        IA_params = tanh_range(self.gamma_range[0], self.gamma_range[1])(IA_params)[..., None, None]
        out_image = img ** (1.0 / IA_params)
        return out_image
    
    def forward(self,x):
        return self.apply_gamma(x,self.bias)
    
class SimpleRGGBGammaEnhancedv6_GreenSEv4(SimpleRGGBGamma):
    """
    直接+
    """
    def __init__(self, in_ch=3, nf=32, tm_pts_num=8, gamma_range=[7, 10.5]) -> None:
        super().__init__(gamma_range)
        self.rgb_enhanced_net = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1), 
            nn.BatchNorm2d(8),  
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), 
            nn.BatchNorm2d(16), 
            nn.LeakyReLU(),
            )
        self.fusion = nn.Conv2d(32, 3, kernel_size=1)
        self.green_enhanced_net = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1), 
            nn.BatchNorm2d(8),  
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), 
            nn.BatchNorm2d(16), 
            nn.LeakyReLU(),
            )
        # self.seblock = SEBlock_no_scale(16,1)
        # self.seblock = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Sigmoid())

    def forward(self,img):
        bz,c,h,w = img.shape                  
        if h==1280:
            img = nn.functional.interpolate(img,(h//2,w//2),mode='bilinear', align_corners=True)            
        gamma_img = super().forward(img)* 255.
        r,g1,g2,b = gamma_img[:,0:1,:,:], gamma_img[:,1:2,:,:], gamma_img[:,2:3,:,:], gamma_img[:,3:4,:,:]
        atten = self.green_enhanced_net(torch.cat([g1,g2],dim=1))
        rgb_enhanced_img = self.rgb_enhanced_net(gamma_img)
        rgb_atten =rgb_enhanced_img + atten
        out = self.fusion(torch.cat([rgb_enhanced_img,rgb_atten],dim=1))
        if h==1280:
            out = nn.functional.interpolate(out,(h,w),mode='bilinear', align_corners=True)          
        return out 

class SimpeRGGBGammaEnhancedv6_GreenSEv4_duplicatedG(SimpleRGGBGammaEnhancedv6_GreenSEv4):
    def __init__(self, in_ch=3, nf=32, tm_pts_num=8, gamma_range=[7, 10.5]) -> None:
        super().__init__(in_ch, nf, tm_pts_num, gamma_range)

    def forward(self,x):
        r,g,b = x[:,0:1,:,:], x[:,1:2,:,:], x[:,2:3,:,:]
        rggb = torch.cat([r,g,g,b],dim=1)
        return super().forward(rggb)

if __name__ == '__main__':
    model = SimpeRGGBGammaEnhancedv6_GreenSEv4_duplicatedG()
    x = torch.ones(4,3,256,256)
    y = model(x)
    # loss = y.sum()
    # loss.backward()
    # check_grad(model)
    print(y.shape)