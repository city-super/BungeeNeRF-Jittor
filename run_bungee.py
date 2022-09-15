import os, sys
import numpy as np
import imageio
import json
import random
import time
import jittor as jt
from jittor import nn
from tqdm import tqdm, trange
import datetime
import matplotlib.pyplot as plt

from run_nerf_helpers import *
from load_multiscale import load_multiscale_data

from tensorboardX import SummaryWriter


jt.flags.use_cuda = 1
DEBUG = False


def batchify(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs):
        arr = []
        for i in range(0, inputs.shape[0], chunk):
            arr.append(fn(inputs[i:i+chunk]))
        return jt.concat(arr, 0)
    return ret


def run_network(means, cov_diags, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    means_flat = jt.reshape(means, [-1, means.shape[-1]])
    cov_diags_flat = jt.reshape(cov_diags, [-1, cov_diags.shape[-1]])
    inputs_flat = jt.concat((means_flat, cov_diags_flat), -1)
    embedded = embed_fn(inputs_flat)

    input_dirs = viewdirs[:,None].expand(([*means.shape[:-1], viewdirs.shape[-1]]))
    input_dirs_flat = jt.reshape(input_dirs, [-1, input_dirs.shape[-1]])
    embedded_dirs = embeddirs_fn(input_dirs_flat)
    embedded = jt.concat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)   
    outputs = jt.reshape(outputs_flat, list(means.shape[:-1]) + list(outputs_flat.shape[1:]))
    return outputs


def batchify_rays(rays_flat, stage, radii, chunk=1024*32, **kwargs):
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], stage, radii[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k : jt.concat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, radii, chunk=1024*32, rays=None, stage=None, c2w=None, **kwargs):
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        rays_o, rays_d = rays

    rays_d = rays_d / jt.norm(rays_d, p=2, dim=-1, keepdim=True)

    sh = rays_d.shape # [..., 3]

    rays_o = jt.reshape(rays_o, [-1,3]).float()
    rays_d = jt.reshape(rays_d, [-1,3]).float()
    radii = jt.reshape(radii, [-1,1]).float()

    rays = jt.concat([rays_o, rays_d], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, stage, radii, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = jt.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]



def render_path(render_poses, hwf, chunk, render_kwargs, stage=0, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    radii = get_radii_for_test(H, W, focal, render_poses)

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, focal, radii[i], chunk=chunk, stage=stage, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            imageio.imwrite(os.path.join(savedir, '{:03d}.png'.format(i)), rgb8)

        del rgb
        del disp
        del acc
        del _


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    input_dims = 3
    embed_fn, input_ch = get_mip_embedder(args.multires, args.min_multires, args.i_embed, input_dims)
    embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.min_multires, args.i_embed)

    model = Bungee_NeRF_block(num_resblocks=args.cur_stage, net_width=args.netwidth, input_ch=input_ch, input_ch_views=input_ch_views)
    grad_vars = list(model.parameters())

    network_query_fn = lambda means, cov_diags, viewdirs, network_fn : run_network(means, cov_diags, viewdirs, network_fn,
                                                                            embed_fn=embed_fn,
                                                                            embeddirs_fn=embeddirs_fn,
                                                                            netchunk=args.netchunk)

    # Create optimizer
    optimizer = jt.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    total_iter = 0
    basedir = args.basedir
    expname = args.expname


    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = jt.load(ckpt_path)

        start = ckpt['global_step']
        total_iter = ckpt['total_iter']

        # load_state_dict, strict=False
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for name, param in model.named_parameters():
            if name in ckpt['network_fn_state_dict'].keys():
                new_state_dict[name] = ckpt['network_fn_state_dict'][name]
            else:
                new_state_dict[name] = param
        model.load_state_dict(new_state_dict)

        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except:
            print("Start a new training stage, reset optimizer.")
            start = 0
        
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'raw_noise_std' : args.raw_noise_std,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, total_iter, grad_vars, optimizer


def integrator(raw, z_vals, rays_d, stage, raw_noise_std=0, white_bkgd=False):
    raw2alpha = lambda raw, dists, act_fn=jt.nn.softplus: 1.-jt.exp(-act_fn(raw)*dists)
    z_vals = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = jt.concat([dists, jt.array(np.array([1e10]).astype(np.float32)).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * jt.norm(rays_d.unsqueeze(-2), p=2, dim=-1)

    acc_rgb = jt.sum(raw[...,:stage+1,:3], dim=2)
    rgb = (1+2*0.001)/(1+jt.exp(-acc_rgb))-0.001

    acc_alpha = jt.sum(raw[...,:stage+1,3], dim=2)
    noise = 0.
    if raw_noise_std > 0.:
        noise = jt.init.gauss(acc_alpha.shape, raw.dtype) * raw_noise_std
    alpha = raw2alpha(acc_alpha + noise, dists)  # [N_rays, N_samples]
    weights = alpha * jt.cumprod(jt.concat([jt.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = jt.sum(weights.unsqueeze(-1) * rgb, -2)  # [N_rays, 3]

    depth_map = jt.sum(weights * z_vals, -1)
    disp_map = 1./jt.maximum(1e-10 * jt.ones_like(depth_map), depth_map / jt.sum(weights, -1))
    acc_map = jt.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map.unsqueeze(-1))

    return rgb_map, disp_map, acc_map, weights, depth_map

def cast(origin, direction, radius, t): 
    t0, t1 = t[..., :-1], t[..., 1:]
    c, d = (t0 + t1)/2, (t1 - t0)/2
    t_mean = c + (2*c*d**2) / (3*c**2 + d**2)
    t_var = (d**2)/3 - (4/15) * ((d**4 * (12*c**2 - d**2)) / (3*c**2 + d**2)**2)
    r_var = radius**2 * ((c**2)/4 + (5/12) * d**2 - (4/15) * (d**4) / (3*c**2 + d**2))
    mean = origin[...,None,:] + direction[..., None, :] * t_mean[..., None]
    null_outer_diag = 1 - (direction**2) / jt.sum(direction**2, -1, keepdims=True)
    cov_diag = (t_var[..., None] * (direction**2)[..., None, :] + r_var[..., None] * null_outer_diag[..., None, :])
    
    return mean, cov_diag
    
def render_rays(ray_batch,
                        stage,
                        radii,
                        network_fn,
                        network_query_fn,
                        N_samples,
                        perturb=0.,
                        N_importance=0,
                        raw_noise_std=0.,
                        ray_nearfar=None,
                        scene_origin=None,
                        scene_scaling_factor=None):

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,:3], ray_batch[:,-3:]

    if ray_nearfar == 'sphere': ## treats earth as a sphere and computes the intersection of a ray and a sphere
        globe_center = jt.array(np.array(scene_origin) * scene_scaling_factor).float()
       
        # 6371011 is earth radius, 250 is the assumed height limitation of buildings in the scene
        earth_radius = 6371011 * scene_scaling_factor
        earth_radius_plus_bldg = (6371011+250) * scene_scaling_factor
        
        ## intersect with building upper limit sphere
        delta = (2*jt.sum((rays_o-globe_center) * rays_d, dim=-1))**2 - 4*jt.norm(rays_d, p=2, dim=-1)**2 * (jt.norm((rays_o-globe_center), p=2, dim=-1)**2 - (earth_radius_plus_bldg)**2)
        d_near = (-2*jt.sum((rays_o-globe_center) * rays_d, dim=-1) - delta**0.5) / (2*jt.norm(rays_d, p=2, dim=-1)**2)
        rays_start = rays_o + (d_near[...,None]*rays_d)
        
        ## intersect with earth
        delta = (2*jt.sum((rays_o-globe_center) * rays_d, dim=-1))**2 - 4*jt.norm(rays_d, p=2, dim=-1)**2 * (jt.norm((rays_o-globe_center), dim=-1)**2 - (earth_radius)**2)
        d_far = (-2*jt.sum((rays_o-globe_center) * rays_d, dim=-1) - delta**0.5) / (2*jt.norm(rays_d, p=2, dim=-1)**2)
        rays_end = rays_o + (d_far[...,None]*rays_d)

        ## compute near and far for each ray
        new_near = jt.norm(rays_o - rays_start, p=2, dim=-1, keepdim=True)
        near = new_near * 0.9
        
        new_far = jt.norm(rays_o - rays_end, p=2, dim=-1, keepdim=True)
        far = new_far * 1.1
        
        # disparity sampling for the first half and linear sampling for the rest
        t_vals_lindisp = jt.linspace(0., 1., steps=N_samples) 
        z_vals_lindisp = 1./(1./near * (1.-t_vals_lindisp) + 1./far * (t_vals_lindisp))
        z_vals_lindisp_half = z_vals_lindisp[:,:int(N_samples*2/3)]

        linear_start = z_vals_lindisp_half[:,-1:]
        t_vals_linear = jt.linspace(0., 1., steps=N_samples-int(N_samples*2/3)+1)
        z_vals_linear_half = linear_start * (1-t_vals_linear) + far * t_vals_linear
        
        z_vals = jt.concat((z_vals_lindisp_half, z_vals_linear_half[:,1:]), -1)
        _, z_vals = jt.argsort(z_vals, -1)
        z_vals = z_vals.expand([N_rays, N_samples])

    elif ray_nearfar == 'flat': ## treats earth as a flat surface and computes the intersection of a ray and a plane
        normal = jt.array([0, 0, 1]) * scene_scaling_factor
        p0_far = jt.array([0, 0, 0]) * scene_scaling_factor
        p0_near = jt.array([0, 0, 250]) * scene_scaling_factor

        near = (p0_near - rays_o * normal).sum(-1) / (rays_d * normal).sum(-1)
        far = (p0_far - rays_o * normal).sum(-1) / (rays_d * normal).sum(-1)
        near = near.clamp(min=1e-6)
        near, far = near.unsqueeze(-1), far.unsqueeze(-1)

        # disparity sampling for the first half and linear sampling for the rest
        t_vals_lindisp = jt.linspace(0., 1., steps=N_samples) 
        z_vals_lindisp = 1./(1./near * (1.-t_vals_lindisp) + 1./far * (t_vals_lindisp))
        z_vals_lindisp_half = z_vals_lindisp[:,:int(N_samples*2/3)]

        linear_start = z_vals_lindisp_half[:,-1:]
        t_vals_linear = jt.linspace(0., 1., steps=N_samples-int(N_samples*2/3)+1)
        z_vals_linear_half = linear_start * (1-t_vals_linear) + far * t_vals_linear
        
        z_vals = jt.concat((z_vals_lindisp_half, z_vals_linear_half[:,1:]), -1)
        z_vals, _ = jt.sort(z_vals, -1)
        z_vals = z_vals.expand([N_rays, N_samples])

    else:
        pass

    if perturb > 0.:
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = jt.concat([mids, z_vals[...,-1:]], -1)
        lower = jt.concat([z_vals[...,:1], mids], -1)
        t_rand = jt.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand


    means, cov_diags = cast(rays_o, rays_d, radii, z_vals) 
    raw = network_query_fn(means, cov_diags, rays_d, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = integrator(raw, z_vals, rays_d, stage, raw_noise_std)

    rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
    
    if  N_importance > 0:
        weights_pad = jt.concat([
            weights[..., :1],
            weights,
            weights[..., -1:],
        ], -1)
        weights_max = jt.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])
        weights_prime = weights_blur + 0.01

        z_samples = sample_pdf(z_vals, weights_prime, N_importance, det=(perturb==0.))
        z_samples = z_samples.detach()
        _, z_vals = jt.argsort(z_samples, -1)

        means, cov_diags = cast(rays_o, rays_d, radii, z_vals)
        raw = network_query_fn(means, cov_diags, rays_d, network_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = integrator(raw, z_vals, rays_d, stage, raw_noise_std)
        
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, 
                        help='input data directory')

    # training options
    parser.add_argument("--N_iters", type=int, default=200000, 
                        help='number of iters to run at current stage')
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--cur_stage", type=int, default=0,
                        help='current training stage: smaller value means further scale')
    parser.add_argument("--use_batching", action='store_true',
                        help='recommand set to False at later training stage for speed up')
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 
    parser.add_argument("--ray_nearfar", type=str, default='sphere', help='options: sphere/flat')


    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--min_multires", type=int, default=0, 
                        help='log2 of min freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for blender)')
    parser.add_argument("--factor", type=int, default=None, 
                        help='downsample factor for images')
    parser.add_argument("--holdout", type=int, default=8, 
                        help='will take every 1/N images as test set')


    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
 
    return parser

def train():
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    images, poses, scene_scaling_factor, scene_origin, scale_split = load_multiscale_data(args.datadir, args.factor)

    n_images = len(images)
    images = images[scale_split[args.cur_stage]:]
    poses = poses[scale_split[args.cur_stage]:]

    if args.holdout > 0:
        print('Auto holdout,', args.holdout)
        i_test = np.arange(images.shape[0])[::args.holdout]

    i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                    (i not in i_test)])

    hwf = poses[0,:3,-1]
    poses = poses[:,:3,:4]
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start_iter, total_iter, grad_vars, optimizer = create_nerf(args)
    scene_stat = {
        'ray_nearfar' : args.ray_nearfar,
        'scene_origin' : scene_origin,
        'scene_scaling_factor' : scene_scaling_factor,
    }
    render_kwargs_train.update(scene_stat)
    render_kwargs_test.update(scene_stat)

    global_step = start_iter
    
    # Move testing data to GPU
    if args.render_test:
        render_poses = np.array(poses[i_test])
        render_poses = jt.array(render_poses)

    # Short circuit if only rendering out from trained model
    if args.render_test:
        print('RENDER TEST')
        with jt.no_grad():
            testsavedir = os.path.join(basedir, expname, 'render_{:06d}'.format(start_iter))
            os.makedirs(testsavedir, exist_ok=True)
            # By default it uses the deepest output head to render result (i.e. cur_stage). 
            # Sepecify 'stage' to shallower output head for lower level of detail rendering.
            rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test, stage=args.cur_stage, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering, saved in ', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            return

    scale_codes = []
    prev_spl = n_images
    cur_scale = 0
    for spl in scale_split[:args.cur_stage+1]:
        scale_codes.append(np.tile(np.ones(((prev_spl-spl),1,1,1))*cur_scale, (1,H,W,1)))
        prev_spl = spl
        cur_scale += 1
    scale_codes = np.concatenate(scale_codes, 0)
    scale_codes = scale_codes.astype(np.int64)

    if args.use_batching:
        rays = np.stack([get_rays_np(H, W, focal, p) for p in poses], 0)
        directions = rays[:,1,:,:,:]
        dx = np.sqrt(
            np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :])**2, -1))
        dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)
        radii = dx[..., None] * 2 / np.sqrt(12)

        rays_rgb = np.concatenate([rays, images[:,None]], 1)
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) 
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) 
        radii = np.stack([radii[i] for i in i_train], 0)
        scale_codes = np.stack([scale_codes[i] for i in i_train], 0)

        rays_rgb = np.reshape(rays_rgb, [-1,3,3])
        radii = np.reshape(radii, [-1, 1])
        scale_codes = np.reshape(scale_codes, [-1, 1])       
        
        print('shuffle rays')
        rand_idx = jt.randperm(rays_rgb.shape[0])
        rays_rgb = rays_rgb[rand_idx]
        radii = radii[rand_idx]
        scale_codes = scale_codes[rand_idx]
        print('done')
        i_batch = 0

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    
    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    if not jt.mpi or jt.mpi.local_rank()==0:
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "")\
                                        .replace(":", "")\
                                        .replace(" ", "_")
        gpu_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        log_dir = os.path.join("./logs", "summaries", "log_" + date +"_gpu" + gpu_idx)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

    for i in trange(start_iter+1, args.N_iters+1):
        time0 = time.time()
        if args.use_batching:
            batch = jt.array(rays_rgb[i_batch : i_batch+args.N_rand])
            batch_radii = jt.array(radii[i_batch : i_batch+args.N_rand])
            batch_scale_codes = jt.array(scale_codes[i_batch : i_batch+args.N_rand])
            batch = jt.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]
            i_batch += args.N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = jt.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                radii = radii[rand_idx]
                scale_codes = scale_codes[rand_idx]
                i_batch = 0
        else:
            img_i = np.random.choice(i_train)
            target = jt.array(images[img_i])
            scale_code = jt.array(scale_codes[img_i])
            pose = poses[img_i]
            rays_o, rays_d = get_rays(H, W, focal, jt.array(pose))
            dx = jt.sqrt(jt.sum((rays_d[:-1, :, :] - rays_d[1:, :, :])**2, -1))
            dx = jt.concat([dx, dx[-2:-1, :]], 0)
            radii = dx[..., None] * 2 / np.sqrt(12)

            if i < args.precrop_iters:
                dH = int(H//2 * args.precrop_frac)
                dW = int(W//2 * args.precrop_frac)
                coords = jt.stack(
                    jt.meshgrid(
                        jt.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                        jt.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    ), -1)
            else:
                coords = jt.stack(jt.meshgrid(jt.linspace(0, H-1, H), jt.linspace(0, W-1, W)), -1)

            coords = jt.reshape(coords, [-1,2])
            select_inds = np.random.choice(coords.shape[0], size=[args.N_rand], replace=False)
            select_coords = coords[select_inds].long()
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]] 
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]] 
            batch_rays = jt.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]
            batch_radii = radii[select_coords[:, 0], select_coords[:, 1]]
            batch_scale_codes = scale_code[select_coords[:, 0], select_coords[:, 1]]

        for stage in range(max(batch_scale_codes).item()+1):
            rgb, _, _, extras = render(H, W, focal, batch_radii, chunk=args.chunk, rays=batch_rays, stage=stage, **render_kwargs_train)
            img_loss = img2mse(rgb*(batch_scale_codes<=stage), target_s*(batch_scale_codes<=stage))
            psnr = mse2psnr(img_loss)
            loss = img_loss
            if 'rgb0' in extras:
                loss += img2mse(extras['rgb0']*(batch_scale_codes<=stage), target_s*(batch_scale_codes<=stage))

            optimizer.backward(loss)
        optimizer.step()
        
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0

        # Rest is logging
        if (i+1)%args.i_weights==0 and (not jt.mpi or jt.mpi.local_rank()==0):
            print(i)
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            jt.save({
                'global_step': global_step,
                'total_iter': total_iter,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)
                 
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            writer.add_scalar('Train/loss', loss.item(), total_iter)
            writer.add_scalar('Train/psnr', psnr.item(), total_iter)
                
        global_step += 1
        total_iter += 1

if __name__=='__main__':
    train()
