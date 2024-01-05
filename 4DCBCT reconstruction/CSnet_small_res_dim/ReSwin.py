# --------------------------------------------------------

import os
#只能使用指定显卡,0卡只能用于临床，数据训练只能用1，2，3卡
gpu_list = '0,1,2,3'
cuda = os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list   #需要设置在import torch 之前和import 使用了torch的库的前面  #此行代码把原来的1，2，3卡变成编号为0，1，2卡
device_num = [0,1,2,3]

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

def ConvUnembed(InChannel_1, OutChannel_1, Kernal_1, InChannel_2, OutChannel_2, Kernal_2 ):  #Block/pBlock  to Max pooling and C
        DownBlock = nn.Sequential(
                nn.Conv2d( InChannel_1, OutChannel_1, Kernal_1 ),
                nn.PReLU(),
                nn.Conv2d( InChannel_2, OutChannel_2, Kernal_2 ),
                nn.PReLU()
                ) 
        return DownBlock

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    #print('window_partition',x.shape)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  
        coords_flatten = torch.flatten(coords, 1)  
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  
        relative_coords[:, :, 0] += self.window_size[0] - 1  
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
       
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PatchMerging(nn.Module):#[B,H*W,C] ->[B,H/2*W/2,2*C]
    
    def __init__(self, dim, norm_layer=nn.LayerNorm):   #input_resolution：输入图像的分辨率，dim：输入通道数
        super().__init__()
        self.dim = dim
        #self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.reduction = nn.Linear(4 * dim, dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x,x_size):
        H,W = x_size    
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.   
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))   #从最后3维padding，C通道不变，W右侧padding，H底侧padding

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]   #0：：2表示从0开始，每隔2个取一个
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]  !修改为[B, H/2*W/2, C]
        x_size = H//2,W//2
        return x,x_size

class PatchEmbed(nn.Module):   #这个类是将输入的图片分成一个个patch，然后将每个patch展开成一个向量，然后将这些向量拼接起来，作为输出

    def __init__(self, embed_dim, norm_layer=None,**kwargs):
        super().__init__()
        self.embed_dim = embed_dim  #获取输入维度也就是通道数，使用时需要手动输入embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(self.embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C   #[B,C,H,W] -> [B,C,H*W] -> [B,H*W,C]
        if self.norm is not None:
            x = self.norm(x)

        return x

class PatchUnEmbed(nn.Module):  #这个类是将输入的patch向量，还原成原来的图片
    
    def __init__(self,embed_dim, norm_layer=None,**kwargs):
        super().__init__()
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(self.embed_dim)
        else:
            self.norm = None

    def forward(self, x, x_size):    #x_size需要在Patchembed中传入前获取，代码：x_size = (x.shape[2], x.shape[3])
        #print('patch_unembed的大小',x_size[0],x_size[1])
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C  #[B,H*W,C] -> [B,C,H*W] -> [B,C,H,W]
        # if self.norm is not None:
        #     x = self.norm(x)
            
        return x

class SwinTransformerBlock(nn.Module):   #sw transformer的基本模块，并不改变输入输出的维度，[B,H*W,C]->[B,H*W,C]
    #定义模型的基本参数
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,**kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(self.dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
       
        H, W = int(x_size[0]), int(x_size[1])
        #print('H,W',H,W)
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):  #输入实例化模型的参数
        H, W = x_size
        B, L, C = x.shape
        #print('Swin Transformer Block before x_shape',x.shape, 'x_size', x_size)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  

        
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        #print('Swin Transformer Block after x_shape',x.shape, 'x_size', x_size)
        return x

class SwinStage(nn.Module):
    "input->patch_embed->swin transformer block*2->norm->patch_unembed->output"

    def __init__(self, embed_dim, input_resolution,  num_heads, window_size, depth=1,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,use_checkpoint=False,**kwargs):

        super().__init__()
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        #定义一个两个swin transformer block的列表
        #blocks = SwinTransformerBlock*depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=self.embed_dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        #path embedding layer
        self.path_embed = PatchEmbed(embed_dim = self.embed_dim, norm_layer=norm_layer)
        #path unembeding layer
        self.path_unembed = PatchUnEmbed(embed_dim = self.embed_dim, norm_layer=norm_layer)
        #patch norm layer
        self.norm = norm_layer(self.embed_dim)
        
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=self.embed_dim, norm_layer=norm_layer)
            # self.norm = norm_layer(self.embed_dim*2)
            # self.path_unembed = PatchUnEmbed(embed_dim = self.embed_dim*2, norm_layer=norm_layer)
            #self.path_embed = PatchEmbed(embed_dim = int(self.embed_dim/2), norm_layer=norm_layer)
        else:
            self.downsample = None
        
    def forward(self, x):               #[B,H*W,embed_dim]->[B,H*W,embed_dim]
        self.x_size = (x.shape[2], x.shape[3])  #获取输入图像的H,W
        #print(x.shape)             #[1,96,512,512]
        
        x = self.path_embed(x)      #[B,H*W,embed_dim]
        
        if self.downsample is not None:
            x,self.x_size = self.downsample(x, self.x_size) #[B,H*W,embed_dim]->[B,H/2*W/2,embed_dim]  #[B,H*W,embed_dim]->[B,H/2*W/2, 2*embed_dim]
        
        x_before = self.path_unembed(x, self.x_size)                #[B,embed_dim,H,W]
        x_embed = x                 #[B,H*W,embed_dim]
        
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, self.x_size)
            else:
                x = blk(x, self.x_size)
       # print(x.shape),print(x_embed.shape)
        x = x + x_embed         #residual connection
        
        x = self.norm(x)
        
        x = self.path_unembed(x, self.x_size)  #[B,H*W,embed_dim] -> [B,embed_dim,H,W]
        #print(x.shape)
        x = x + x_before        #residual connection    

        return x

class Transformer(nn.Module):
    def __init__(self, img_size=512,embed_dim=96,num_layers=2, extract_layers=[1,2,3],**kwargs):  #[1，2，4，8，16，32]
        super().__init__()
        #path_merging layer
        self.embed_dim = embed_dim
        self.layer = nn.ModuleList()
        self.extract_layers = extract_layers
        img_size   = to_2tuple(img_size)

        for i_layer in range(num_layers):
            layer = SwinStage(embed_dim= self.embed_dim, 
                            input_resolution=(img_size[0]/(2**i_layer), img_size[0]/(2**i_layer)), 
                            num_heads=6, window_size=8, depth=2, downsample=PatchMerging if (i_layer >0) else None)
            self.layer.append(layer)

    def forward(self,x):
        extract_x = []
        
        for depth, layer_block in enumerate(self.layer):
            #print('初始输入：',x.shape)
            x = layer_block(x)
            #print('经过两个SW——blcok输入:',x.shape)
            #print(x.shape)
            if depth + 1 in self.extract_layers:
                extract_x.append(x)

        return extract_x

class PriorSwinNet(nn.Module):
    def __init__(self, img_size=512, patch_size=1, in_chans=1,
                 embed_dim=96, depths=1, num_heads=[6, 6, 6, 3],                 
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, img_range=1., resi_connection='1conv',
                 **kwargs):
        super(PriorSwinNet, self).__init__()
        num_in_ch           = in_chans        #输入通道数
        num_out_ch          = in_chans        #输出通道数
        self.img_range      = img_range       #？
        self.mean           = torch.zeros(1, 1, 1, 1)     #[1,1,1,1]的全0张量
        self.num_layers     = depths     #depths列表长度作为层数
        self.embed_dim      = embed_dim       #embed_dim作为特征通道数    
        self.ape            = ape             #是否使用绝对位置编码
        self.patch_norm     = patch_norm      #是否使用patch_norm
        self.num_features   = embed_dim       #embed_dim作为特征通道数
        self.mlp_ratio      = mlp_ratio       #mlp_ratio (float): Ratio of mlp hidden dim to embedding dim

        patch_size = to_2tuple(patch_size)    #patch_size ->(patch_size,patch_size)
        img_size   = to_2tuple(img_size)      #img_size ->(img_size,img_size)

        patches_resolution  = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]     #[img_size[0],img_size[1]]
        num_patches         = patches_resolution[0] * patches_resolution[1] 

        self.conv_embed     = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)                          #swintransoformer_block 前的第一层卷积  
        self.patch_embed    = PatchEmbed(embed_dim=embed_dim,norm_layer=norm_layer if self.patch_norm else None)   #patch_embed层，作用是将输入的图片分割成patch，然后将patch展平，再经过一个全连接层，得到一个向量
        self.patch_unembed  = PatchUnEmbed(embed_dim=embed_dim,norm_layer=norm_layer if self.patch_norm else None)   #patch_unembed层，作用是将patch展平的向量，经过一个全连接层，得到一个patch
                 
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)    #权重正态分布初始化

        self.pos_drop = nn.Dropout(p=drop_rate)   #p为丢弃概率，一般设置较小的值，如0.1

        #dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  #分段线性函数，返回一个列表，包含在区间0到drop_path_rate上均匀间隔的sum(depths)个点
        #transformer_encoder
        self.transformer = Transformer(img_size=512,embed_dim=self.embed_dim,num_layers=6, extract_layers=[1,2,3,4,5,6],
                                        num_heads=6, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,)

        self.norm = norm_layer(self.num_features)   #按照num_features通道数进行归一化
        #STC层后的残差连接做卷积
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)    #残差连接方式为1conv时，STC后的卷积层为3*3卷积核，步进为1，填充为1
        elif resi_connection == '3conv':                                        #残差连接方式为3conv时,先卷积做降维，在1*1卷积，最后卷积恢复通道数
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),    #STC后的卷积层为3*3卷积核，步进为1，填充为1，输出通道数为embed_dim // 4
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),   #LeakyReLU激活函数
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),#1*1卷积核，步进为1，填充为0，输出通道数为embed_dim // 4
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))     #3*3卷积核，步进为1，填充为1，输出通道数为embed_dim

        self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)   #最后的卷积层为3*3卷积核，步进为1，填充为1，输出通道数为num_out_ch
        #创建一个输入为[B,96,512,512],输出为[B,4,504,504]的卷积层
       
        self.conv_unembed0 = ConvUnembed(96,4,5,4,4,5)
        self.conv_unembed1 = ConvUnembed(96, 8, 5, 8,8, 5)  
        self.conv_unembed2 = ConvUnembed(96, 16, 5, 16, 16,5)
        self.conv_unembed3 = ConvUnembed(96, 32, 5, 32, 32,5)
        self.conv_unembed4 = ConvUnembed(96, 64, 5, 64, 64,5)
        self.conv_unembed5 = ConvUnembed(96, 128, 5, 128, 128,5)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):   
        #在forward中有调用[B,embed_dim,H,W],[B,embed_dim*2,H/2,W/2],[B,embed_dim*4,H/4,W/4],[B,embed_dim*8,H/8,W/8],[B,embed_dim*16,H/16,W/16],[B,embed_dim*32,H/32,W/32]

        encoder_features = self.transformer(x)
        
        return encoder_features

    def forward(self, x):
        self.mean = self.mean.type_as(x)    #之前self.mean定义为[1,1,1,1]维的全0的tensor
        x = (x - self.mean) * self.img_range  #这一步实际并没有改变什么？
        x_first = self.conv_embed(x)        #[B,C,H,W]->[B,embed_dim,H,W]
        z0,z1,z2,z3,z4,z5 = self.forward_features(x_first)   
        z0 = self.conv_unembed0(z0)
        z1 = self.conv_unembed1(z1)
        z2 = self.conv_unembed2(z2)
        z3 = self.conv_unembed3(z3)
        z4 = self.conv_unembed4(z4)
        z5 = self.conv_unembed5(z5)
        
        #res = self.conv_after_body(X_STC) + x_first
        #x = x + self.conv_last(res)
        #x = x / self.img_range + self.mean
        #x = x + self.conv_last(X_STC)
        return z0,z1,z2,z3,z4,z5 
      
if __name__ == '__main__':

    a = torch.Tensor(1,1,512,512)  
    
    sw_Block = PriorSwinNet(img_size=512,embed_dim=96,num_layers=6, extract_layers=[1,2,3,4,5,6])
    pytorch_total_params = sum(p.numel() for p in sw_Block.parameters() if p.requires_grad)  #可以通过每个参数组的求和得出参数量（计算可训练的参数if p.requires_grad）
    print("Total_params: {}".format(pytorch_total_params))    
    if torch.cuda.is_available():
        a=a.cuda() ; #sw_Block = sw_Block.cuda()
        #分布式训练   
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs")
        sw_Block = nn.DataParallel( sw_Block.cuda() )
        print('参数大小为：',sw_Block.module.parameters())
    else:
        sw_Block = sw_Block.cpu()
    
    b = sw_Block(a)   #[1,1,256,256]->[1,1,256,256],[1,1,512,512]
    print(b[0].shape,b[1].shape,b[2].shape,b[3].shape,b[4].shape,b[5].shape)      #不改变size,[1,1,256,256]->[1,1,256,256]
    
# if __name__ == '__main__':
#     a = torch.Tensor(1,4,512,512)  
#     sw_Block = PriorSwinNet(in_chans=4)
#     #print('模型的浮点运算量为：',sw_Block.flops())

#     if torch.cuda.is_available():
#         a=a.cuda() ; #sw_Block = sw_Block.cuda()
#         #分布式训练   
#     if torch.cuda.device_count() > 1:
#         print("Let's use", torch.cuda.device_count(), "GPUs")
#         sw_Block = nn.DataParallel( sw_Block.cuda() )
#         print('参数大小为：',sw_Block.module.parameters())
#     else:
#         sw_Block = sw_Block.cpu()
    
#     b = sw_Block(a)   #[1,1,256,256]->[1,1,256,256],[1,1,512,512]
#     print(b.shape)      #不改变size,[1,1,256,256]->[1,1,256,256]