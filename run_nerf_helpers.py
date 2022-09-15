import jittor as jt
from jittor import nn
import numpy as np

# Misc
img2mse = lambda x, y : jt.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * jt.log(x) / jt.log(jt.array(np.array([10.])))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        min_freq = self.kwargs['min_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**jt.linspace(min_freq, max_freq, steps=N_freqs)
        else:
            freq_bands = jt.linspace(2.**min_freq, 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return jt.concat([fn(inputs) for fn in self.embed_fns], -1)


class MipEmbedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x[:,:d])
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        min_freq = self.kwargs['min_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands_y = 2.**jt.linspace(min_freq, max_freq, steps=N_freqs)
            freq_bands_w = 4.**jt.linspace(min_freq, max_freq, steps=N_freqs)
        else:
            freq_bands_y = jt.linspace(2.**min_freq, 2.**max_freq, steps=N_freqs)
            freq_bands_w = jt.linspace(4.**min_freq, 4.**max_freq, steps=N_freqs)

        for ctr in range(len(freq_bands_y)):
            for p_fn in self.kwargs['periodic_fns']: 
                embed_fns.append(lambda inputs, p_fn=p_fn, freq_y=freq_bands_y[ctr], freq_w=freq_bands_w[ctr] : p_fn(inputs[:,:d] * freq_y) * jt.exp((-0.5) * freq_w * inputs[:,d:]))
                out_dim += d
                
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return jt.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, min_multires=0, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'min_freq_log2': min_multires,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [jt.sin, jt.cos]
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

def get_mip_embedder(multires, min_multires=0, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'min_freq_log2': min_multires,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [jt.sin, jt.cos],
    }
    
    embedder_obj = MipEmbedder(**embed_kwargs)
    embed = lambda inputs, eo=embedder_obj : eo.embed(inputs)
    return embed, embedder_obj.out_dim

class Bungee_NeRF_baseblock(nn.Module):
    def __init__(self, net_width=256, input_ch=3, input_ch_views=3):
        super(Bungee_NeRF_baseblock, self).__init__()
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, net_width)] + [nn.Linear(net_width, net_width) for _ in range(3)])
        self.views_linear = nn.Linear(input_ch_views + net_width, net_width//2)
        self.feature_linear = nn.Linear(net_width, net_width)
        self.alpha_linear = nn.Linear(net_width, 1)
        self.rgb_linear = nn.Linear(net_width//2, 3)

    def execute(self, input_pts, input_views):
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = jt.nn.relu(h)
        alpha = self.alpha_linear(h)
        feature0 = self.feature_linear(h)
        h0 = jt.concat([feature0, input_views], -1)
        h0 = self.views_linear(h0)
        h0 = jt.nn.relu(h0)
        rgb = self.rgb_linear(h0)
        return rgb, alpha, h

class Bungee_NeRF_resblock(nn.Module):
    def __init__(self, net_width=256, input_ch=3, input_ch_views=3):
        super(Bungee_NeRF_resblock, self).__init__()
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch+net_width, net_width), nn.Linear(net_width, net_width)])
        self.views_linear = nn.Linear(input_ch_views + net_width, net_width//2)
        self.feature_linear = nn.Linear(net_width, net_width)
        self.alpha_linear = nn.Linear(net_width, 1)
        self.rgb_linear = nn.Linear(net_width//2, 3)
    
    def execute(self, input_pts, input_views, h):
        h = jt.concat([input_pts, h], -1)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = jt.nn.relu(h)
        alpha = self.alpha_linear(h)
        feature0 = self.feature_linear(h)
        h0 = jt.concat([feature0, input_views], -1)
        h0 = self.views_linear(h0)
        h0 = jt.nn.relu(h0)
        rgb = self.rgb_linear(h0)
        return rgb, alpha, h

class Bungee_NeRF_block(nn.Module):
    def __init__(self, num_resblocks=3, net_width=256, input_ch=3, input_ch_views=3):
        super(Bungee_NeRF_block, self).__init__()
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.num_resblocks = num_resblocks

        self.baseblock = Bungee_NeRF_baseblock(net_width=net_width, input_ch=input_ch, input_ch_views=input_ch_views)
        self.resblocks = nn.ModuleList([
            Bungee_NeRF_resblock(net_width=net_width, input_ch=input_ch, input_ch_views=input_ch_views) for _ in range(num_resblocks)
        ])

    def execute(self, x):
        input_pts, input_views = jt.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        alphas = []
        rgbs = []
        base_rgb, base_alpha, h = self.baseblock(input_pts, input_views)
        alphas.append(base_alpha)
        rgbs.append(base_rgb)
        for i in range(self.num_resblocks):
            res_rgb, res_alpha, h = self.resblocks[i](input_pts, input_views, h)
            alphas.append(res_alpha)
            rgbs.append(res_rgb)

        output = jt.concat([jt.stack(rgbs,1), jt.stack(alphas,1)],-1)
        return output


def get_rays(H, W, focal, c2w):
    i, j = jt.meshgrid(jt.linspace(0, W-1, W), jt.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    dirs = jt.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -jt.ones_like(i)], -1)
    dirs = dirs/jt.norm(dirs, p=2, dim=-1)[...,None]
    rays_d = jt.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def get_radii_for_test(H, W, focal, c2w):
    i, j = jt.meshgrid(jt.linspace(0, W-1, W), jt.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    dirs = jt.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -jt.ones_like(i)], -1)
    rays_d = jt.sum(dirs[np.newaxis, ..., np.newaxis, :] * c2w[:, np.newaxis, np.newaxis, :3,:3], -1) 
    dx = jt.sqrt(
        jt.sum((rays_d[:, :-1, :, :] - rays_d[:, 1:, :, :])**2, -1))
    dx = jt.concat([dx, dx[:, -2:-1, :]], 1)
    radii = dx[..., None] * 2 / np.sqrt(12)
    return radii

def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / jt.sum(weights, -1, keepdims=True)
    cdf = jt.cumsum(pdf, -1)
    cdf = jt.concat([jt.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = jt.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = jt.random(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = jt.searchsorted(cdf, u, right=True)
    below = jt.maximum(jt.zeros_like(inds-1), inds-1)
    above = jt.minimum((cdf.shape[-1]-1) * jt.ones_like(inds), inds)
    inds_g = jt.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = jt.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = jt.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom[denom<1e-5] = 1.0
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


