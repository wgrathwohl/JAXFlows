"""
Written by big willy copyright big willy incorporated
"""

import jax.numpy as np
from jax import random
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, Conv
from jax.nn import log_sigmoid
from utils import squeeze2d, unsqueeze2d


"""
Probability utils
"""
def sample_n01(rng, shape):
    return random.normal(rng, shape)


def log_prob_n01(x):
    logpx = -np.square(x) / 2 - np.log(np.sqrt(2 * np.pi))
    logpx = logpx.reshape(logpx.shape[0], -1)
    return np.sum(logpx, axis=-1)


"""
Overview of flow interface

flow interface: init_flow(rng, *params):
    returns params, forward_fn, reverse_fn

forward_fn(params, prev_sample, prev_logp):
    return this_sample, this_logp

reverse_fn(params, next_sample, next_logp):
    return this_sample, this_logp
"""


"""
Utils for chaining together flows
"""


def init_flow_chain(rng, init_fns, init_params, init_batch=None):
    assert len(init_fns) == len(init_params)
    p_chain, f_chain, r_chain = [], [], []
    for init_fn, init_param in zip(init_fns, init_params):
        rng, srng = random.split(rng)
        p, f, r = init_fn(srng, *init_param, init_batch=init_batch)
        p_chain.append(p)
        f_chain.append(f)
        r_chain.append(r)
        if init_batch is not None:
            init_batch, _ = f(p, init_batch, 0.)

    def chain_forward(params, prev_sample, prev_logp=0.):
        x, logp = prev_sample, prev_logp
        for p, f in zip(params, f_chain):
            x, logp = f(p, x, logp)
        return x, logp

    def chain_reverse(params, next_sample, next_logp=0.):
        x, logp = next_sample, next_logp
        for p, r in reversed(list(zip(params, r_chain))):
            x, logp = r(p, x, logp)
        return x, logp

    return p_chain, chain_forward, chain_reverse


def init_factor_out(flow_params, flow_forward, flow_reverse, split_fn, rejoin_fn):
    """
    takes an existing flow and turns it into a flow which factors out dimensions
    """
    def factor_forward(params, x, prev_logp=0.):
        y, delta_logp = flow_forward(params, x, prev_logp=prev_logp)
        y_keep, y_factorout = split_fn(y)
        return y_keep, delta_logp, y_factorout

    def factor_reverse(params, y_next, prev_logp=0., y_factorout=None):
        y = rejoin_fn(y_next, y_factorout)
        return flow_reverse(params, y, prev_logp=prev_logp)

    return flow_params, factor_forward, factor_reverse


def init_factor_out_chain(rng, init_fns, init_params, split_fn, rejoin_fn, init_batch=None):
    assert len(init_fns) == len(init_params)
    p_chain, f_chain, r_chain = [], [], []
    for init_fn, init_param in zip(init_fns, init_params):
        rng, srng = random.split(rng)
        p, f, r = init_fn(srng, *init_param, init_batch=init_batch)
        p, f, r = init_factor_out(p, f, r, split_fn, rejoin_fn)
        p_chain.append(p)
        f_chain.append(f)
        r_chain.append(r)
        if init_batch is not None:
            init_batch, _, __ = f(p, init_batch, 0.)

    def chain_forward(params, prev_sample, prev_logp=0.):
        x_next, logp = prev_sample, prev_logp
        zs = []
        for p, f in zip(params, f_chain):
            x_next, logp, x_factorout = f(p, x_next, logp)
            zs.append(x_factorout)
        zs.append(x_next)
        return zs, logp

    def chain_reverse(params, zs, next_logp=0.):
        zs, z_next = zs[:-1], zs[-1]  # split off the final z
        logp = next_logp
        for p, r, z_factorout in reversed(list(zip(params, r_chain, zs))):
            z_next, logp = r(p, z_next, logp, z_factorout)
        return z_next, logp

    return p_chain, chain_forward, chain_reverse


"""
Some basic split/rejoin fns
"""


def split_dims(x):
    d = x.shape[1] // 2
    return x[:, :d], x[:, d:]


def rejoin_dims(x1, x2):
    return np.concatenate([x1, x2], 1)


def split_channels(x):
    c = x.shape[3] // 2
    return x[:, :, :, :c], x[:, :, :, c:]


def rejoin_channels(x1, x2):
    return np.concatenate([x1, x2], 3)


"""
Linear Real-NVP
"""


def init_nvp(rng, dim, flip, init_batch=None):
    net_init, net_apply = stax.serial(Dense(512), Relu, Dense(512), Relu, Dense(dim))
    in_shape = (-1, dim // 2)
    _, net_params = net_init(rng, in_shape)

    def shift_and_log_scale_fn(net_params, x1):
        s = net_apply(net_params, x1)
        return np.split(s, 2, axis=1)

    def nvp_forward(net_params, prev_sample, prev_logp=0.):
        d = dim // 2
        x1, x2 = prev_sample[:, :d], prev_sample[:, d:]
        if flip:
            x2, x1 = x1, x2
        shift, log_scale = shift_and_log_scale_fn(net_params, x1)
        y2 = x2 * np.exp(log_scale) + shift
        if flip:
            x1, y2 = y2, x1
        y = np.concatenate([x1, y2], axis=-1)
        return y, prev_logp + np.sum(log_scale, axis=-1)

    def nvp_reverse(net_params, next_sample, next_logp=0.):
        d = dim // 2
        y1, y2 = next_sample[:, :d], next_sample[:, d:]
        if flip:
            y1, y2 = y2, y1
        shift, log_scale = shift_and_log_scale_fn(net_params, y1)
        x2 = (y2 - shift) * np.exp(-log_scale)
        if flip:
            y1, x2 = x2, y1
        x = np.concatenate([y1, x2], axis=-1)
        return x, next_logp - np.sum(log_scale, axis=-1)
    
    return net_params, nvp_forward, nvp_reverse


def init_nvp_chain(rng, dim, n=2, init_batch=None, actnorm=False):
    """Helper for making Real-NVP chains"""
    flip = False
    params = []
    chain = []
    for _ in range(n):
        if actnorm:
            params.append(())
            chain.append(init_actnorm)

        params.append((dim, flip))
        chain.append(init_nvp)

        flip = not flip
    return init_flow_chain(rng, chain, params, init_batch=init_batch)


"""
Linear Actnorm
"""


def init_actnorm(rng, init_batch=None):
    assert init_batch is not None, "Actnorm requires data-dependent init"
    mu, sig = np.mean(init_batch, axis=0), np.std(init_batch, axis=0)
    log_scale = np.log(sig)
    params = (mu, log_scale)

    def actnorm_forward(params, prev_sample, prev_logp=0.):
        mu, log_scale = params
        y = (prev_sample - mu[None]) * np.exp(-log_scale)[None]
        return y, prev_logp - np.sum(log_scale)

    def actnorm_reverse(params, next_sample, next_logp=0.):
        mu, log_scale = params
        x = next_sample * np.exp(log_scale)[None] + mu[None]
        return x, next_logp + np.sum(log_scale)

    return params, actnorm_forward, actnorm_reverse


"""
Convolutional Actnorm
"""


def init_conv_actnorm(rng, init_batch=None):
    assert init_batch is not None, "Actnorm requires data-dependent init"
    mu, sig = np.mean(init_batch, axis=(0, 1, 2)), np.std(init_batch, axis=(0, 1, 2))
    log_scale = np.log(sig)
    params = (mu, log_scale)

    def actnorm_forward(params, prev_sample, prev_logp=0.):
        mu, log_scale = params
        y = (prev_sample - mu[None, None, None, :]) * np.exp(-log_scale)[None, None, None, :]
        b, h, w, c = prev_sample.shape
        return y, prev_logp - np.sum(log_scale) * w * h

    def actnorm_reverse(params, next_sample, next_logp=0.):
        mu, log_scale = params
        x = next_sample * np.exp(log_scale)[None, None, None, :] + mu[None, None, None, :]
        b, h, w, c = next_sample.shape
        return x, next_logp + np.sum(log_scale) * w * h

    return params, actnorm_forward, actnorm_reverse


"""
Squeeze Layers
"""


def init_squeeze(rng, init_batch=None):
    params = ()

    def squeeze_forward(params, prev_sample, prev_logp=0.):
        return squeeze2d(prev_sample), prev_logp

    def squeeze_reverse(params, next_sample, next_logp=0.):
        return unsqueeze2d(next_sample), next_logp

    return params, squeeze_forward, squeeze_reverse


"""
Convolutional Coupling Layers
"""


def init_conv_affine_coupling(rng, in_shape, n_channels, flip, sigmoid=True, init_batch=None):
    """
    in_shape: tuple of (h, w, c)
    """
    h, w, c = in_shape
    assert c % 2 == 0, "channels must be even doooooooog!"
    half_c = c // 2
    net_init, net_apply = stax.serial(Conv(n_channels, (3, 3), padding="SAME"), Relu,
                                      Conv(n_channels, (3, 3), padding="SAME"), Relu,
                                      Conv(c, (3, 3), padding="SAME"))
    _, net_params = net_init(rng, (-1, h, w, half_c))

    def shift_and_log_scale_fn(net_params, x1):
        s = net_apply(net_params, x1)
        return np.split(s, 2, axis=3)

    def conv_coupling_forward(net_params, prev_sample, prev_logp=0.):
        x1, x2 = prev_sample[:, :, :, :half_c], prev_sample[:, :, :, half_c:]
        if flip:
            x2, x1 = x1, x2
        shift, log_scale = shift_and_log_scale_fn(net_params, x1)
        if sigmoid:
            log_scale = log_sigmoid(log_scale + 2.)
        y2 = x2 * np.exp(log_scale) + shift
        if flip:
            x1, y2 = y2, x1
        y = np.concatenate([x1, y2], axis=-1)
        return y, prev_logp + np.sum(log_scale, axis=(1, 2, 3))

    def conv_coupling_reverse(net_params, next_sample, next_logp=0.):
        y1, y2 = next_sample[:, :, :, :half_c], next_sample[:, :, :, half_c:]
        if flip:
            y1, y2 = y2, y1
        shift, log_scale = shift_and_log_scale_fn(net_params, y1)
        if sigmoid:
            log_scale = log_sigmoid(log_scale + 2.)
        x2 = (y2 - shift) * np.exp(-log_scale)
        if flip:
            y1, x2 = x2, y1
        x = np.concatenate([y1, x2], axis=-1)
        return x, next_logp - np.sum(log_scale, axis=(1, 2, 3))

    return net_params, conv_coupling_forward, conv_coupling_reverse


"""
High level convolutional flows
"""


def init_conv_flow_step(rng, in_shape, n_channels, flip, init_batch=None):
    """ One step of flow actnorm --> affine coupling"""
    return init_flow_chain(rng,
                           [init_conv_actnorm, init_conv_affine_coupling],
                           [(), (in_shape, n_channels, flip)],
                           init_batch=init_batch)


def init_conv_flow_block(rng, in_shape, n_steps, n_channels, init_batch=None):
    """ Flow block: squeeze --> n_steps * flow_step """
    flip = False
    init_fns = [init_squeeze]
    init_params = [()]
    h, w, c = in_shape
    squeeze_shape = h // 2, w // 2, c * 4
    for _ in range(n_steps):
        init_fns.append(init_conv_flow_step)
        init_params.append((squeeze_shape, n_channels, flip))
        flip = not flip
    return init_flow_chain(rng, init_fns, init_params, init_batch=init_batch)


def init_multiscale_conv_flow(rng, in_shape, n_channels, n_blocks, n_steps, init_batch=None):
    """ Creates a multi-scale convolutional normalizing flow like Glow but currently no 1x1 convolutions """
    params = []
    chain = []
    cur_shape = in_shape
    for _ in range(n_blocks):
        chain.append(init_conv_flow_block)
        params.append(
            (cur_shape, n_steps, n_channels)
        )
        h, w, c = cur_shape
        cur_shape = h // 2, w // 2, c * 2

    return init_factor_out_chain(rng, chain, params, split_channels, rejoin_channels, init_batch=init_batch)


"""
Utilities to build functions for training and eval
"""


def make_log_prob_fn(forward_fn, base_dist_log_prob):
    def log_prob(p, x):
        z, logp = forward_fn(p, x)
        return base_dist_log_prob(z) + logp
    return log_prob


def make_sample_fn(reverse_fn, base_dist_sample):
    def sample(rng, p, n):
        z = base_dist_sample(rng, n)
        return reverse_fn(p, z, 0.)[0]
    return sample


if __name__ == "__main__":
    rng = random.PRNGKey(0)

    rng, srng = random.split(rng)
    init_batch = random.normal(srng, (13, 32, 32, 4))

    #ps, forward, reverse = init_conv_actnorm(rng, init_batch=init_batch)
    #1/0

    #ps, forward, reverse = init_conv_affine_coupling(rng, (32, 32, 4), 64, True, init_batch=init_batch)
    #1/0

    #ps, forward, reverse = init_conv_flow_step(rng, (32, 32, 4), 64, True, init_batch=init_batch)
    #1/0

    init_batch = random.normal(srng, (13, 32, 32, 3))
    #ps, forward, reverse = init_conv_flow_block(rng, (32, 32, 3), 4, 64, init_batch=init_batch)
    #z, logp = forward(ps, init_batch)
    #1/0

    rng, srng = random.split(rng)
    ps, forward, reverse = init_multiscale_conv_flow(srng, (32, 32, 3), 64, 3, 20, init_batch=init_batch)
    z, logp = forward(ps, init_batch)
    for _z in z:
        print(_z.shape)
    1/0


    #
    #
    # from sklearn import cluster, datasets, mixture
    # from sklearn.preprocessing import StandardScaler
    # import matplotlib.pyplot as plt
    # from jax.experimental import optimizers
    # from jax import jit, grad
    # import numpy as onp
    #
    # n_samples = 2000
    # noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    # X, y = noisy_moons
    # X = StandardScaler().fit_transform(X)
    #
    # iters = int(1e5)
    # data_generator = (X[onp.random.choice(X.shape[0], 100)] for _ in range(iters))
    #
    #
    #
    # rng, srng = random.split(rng)
    # init_batch = next(data_generator)
    # ps, forward_fn, reverse_fn = init_nvp_chain(srng, 2, n=4, init_batch=init_batch, actnorm=True)
    #
    # log_prob_fn = make_log_prob_fn(forward_fn, log_prob_n01)
    # sample_fn = make_sample_fn(reverse_fn, lambda rng, n: sample_n01(rng, (n, 2)))
    #
    # def loss(params, batch):
    #     return -np.mean(log_prob_fn(params, batch))
    #
    #
    # opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)
    #
    # @jit
    # def step(i, opt_state, batch):
    #     params = get_params(opt_state)
    #     g = grad(loss)(params, batch)
    #     return opt_update(i, g, opt_state)
    #
    #
    #
    # opt_state = opt_init(ps)
    # for i in range(iters):
    #     x = next(data_generator)
    #     opt_state = step(i, opt_state, x)
    #     if i % 4000 == 0:
    #         ps = get_params(opt_state)
    #         print(i, loss(ps, x))
    #         x_samp = sample_fn(rng, ps, n_samples)
    #         plt.clf()
    #         plt.scatter(x_samp[:, 0], x_samp[:, 1])
    #         plt.savefig("fig_{}.png".format(i))
