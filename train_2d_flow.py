from flow_utils import *



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

    iters = int(1e5)
    data_generator = (X[onp.random.choice(X.shape[0], 100)] for _ in range(iters))

    rng = random.PRNGKey(0)

    rng, srng = random.split(rng)
    init_batch = next(data_generator)
    ps, forward_fn, reverse_fn = init_nvp_chain(srng, 2, n=4, init_batch=init_batch, actnorm=True)

    log_prob_fn = make_log_prob_fn(forward_fn, log_prob_n01)
    sample_fn = make_sample_fn(reverse_fn, lambda rng, n: sample_n01(rng, (n, 2)))

    def loss(params, batch):
        return -np.mean(log_prob_fn(params, batch))


    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)

    @jit
    def step(i, opt_state, batch):
        params = get_params(opt_state)
        g = grad(loss)(params, batch)
        return opt_update(i, g, opt_state)


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
