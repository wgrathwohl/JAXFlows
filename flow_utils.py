import jax.numpy as np
from jax import random
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu


# probability distributions
def sample_n01(rng, shape):
    return random.normal(rng, shape)


def log_prob_n01(x):
    logpx = -np.square(x) / 2 - np.log(np.sqrt(2 * np.pi))
    logpx = logpx.reshape(logpx.shape[0], -1)
    return np.sum(logpx, axis=-1)


# flow interface: init_flow(rng, *paras):
#     returns params, forward_fn, reverse_fn
#
# forward_fn(params, prev_sample, prev_logp):
#     return this_sample, this_logp
#
# reverse_fn(params, next_sample, next_logp):
#     return this_sample, this_logp

def init_nvp(rng, dim, flip, init_batch=None):
    net_init, net_apply = stax.serial(Dense(512), Relu, Dense(512), Relu, Dense(dim))
    in_shape = (-1, dim // 2)
    _, net_params = net_init(rng, in_shape)

    def shift_and_log_scale_fn(net_params, x1):
        s = net_apply(net_params, x1)
        return np.split(s, 2, axis=1)

    def nvp_forward(net_params, prev_sample, prev_logp):
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


def init_flow_chain(rng, init_fns, init_params, init_batch=None):
    assert len(init_fns) == len(init_params)
    p_chain, f_chain, r_chain = [], [], []
    for init_fn, init_param in zip(init_fns, init_params):
        rng, srng = random.split(rng)
        p, f, r = init_fn(srng, *init_param)
        p_chain.append(p)
        f_chain.append(f)
        r_chain.append(r)
        if init_batch is not None:
            init_batch, _ = f(p, init_batch, 0.)

    def chain_forward(params, prev_sample, prev_logp):
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


def init_nvp_chain(rng, dim, n=2):
    flip = False
    params = []
    for _ in range(n):
        params.append((dim, flip))
        flip = not flip
    return init_flow_chain(rng, [init_nvp for _ in range(n)], params)


def make_log_prob_fn(reverse_fn, base_dist_log_prob):
    def log_prob(p, x):
        z, logp = reverse_fn(p, x)
        return base_dist_log_prob(z) + logp
    return log_prob


def make_sample_fn(forward_fn, base_dist_sample):
    def sample(rng, p, n):
        z = base_dist_sample(rng, n)
        return forward_fn(p, z, 0.0)[0]
    return sample







    



if __name__ == "__main__":
    from sklearn import cluster, datasets, mixture
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    from jax.experimental import optimizers
    from jax import jit, grad
    import numpy as onp

    n_samples = 2000
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    X, y = noisy_moons
    X = StandardScaler().fit_transform(X)

    rng = random.PRNGKey(0)

    rng, srng = random.split(rng)
    ps, forward_fn, reverse_fn = init_nvp_chain(srng, 2, n=4)

    log_prob_fn = make_log_prob_fn(reverse_fn, log_prob_n01)
    sample_fn = make_sample_fn(forward_fn, lambda rng, n: sample_n01(rng, (n, 2)))

    def loss(params, batch):
        return -np.mean(log_prob_fn(params, batch))


    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)

    @jit
    def step(i, opt_state, batch):
        params = get_params(opt_state)
        g = grad(loss)(params, batch)
        return opt_update(i, g, opt_state)


    iters = int(1e5)
    data_generator = (X[onp.random.choice(X.shape[0], 100)] for _ in range(iters))
    opt_state = opt_init(ps)
    for i in range(iters):
        x = next(data_generator)
        opt_state = step(i, opt_state, x)
        if i % 4000 == 0:
            ps = get_params(opt_state)
            print(i, loss(ps, x))
            x_samp = sample_fn(rng, ps, n_samples)
            plt.clf()
            plt.scatter(x_samp[:, 0], x_samp[:, 1])
            plt.savefig("fig_{}.png".format(i))
    1/0
    # ps = get_params(opt_state)
    #
    # from matplotlib import animation
    #
    # rng, srng = random.split(rng)
    # x = sample_n01(srng, 1000, 2)
    # values = [x]
    # for p, config in zip(ps, cs):
    #     shift_log_scale_fn, flip = config
    #     x = nvp_forward(p, shift_log_scale_fn, x, flip=flip)
    #     values.append(x)
    #
    # # First set up the figure, the axis, and the plot element we want to animate
    # fig, ax = plt.subplots()
    # #ax.set_xlim(xlim)
    # #ax.set_ylim(ylim)
    #
    # y = values[0]
    # paths = ax.scatter(y[:, 0], y[:, 1], s=10, color='red')
    #
    #
    # def animate(i):
    #     l = i // 48
    #     t = (float(i % 48)) / 48
    #     y = (1 - t) * values[l] + t * values[l + 1]
    #     paths.set_offsets(y)
    #     return (paths,)
    #
    #
    # anim = animation.FuncAnimation(fig, animate, frames=48 * len(cs), interval=1, blit=False)
    # anim.save('anim.gif', writer='imagemagick', fps=100)
