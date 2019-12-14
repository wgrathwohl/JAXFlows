import jax.numpy as np


def squeeze2d(input, factor=2):
    if factor == 1:
        return input

    B, H, W, C = input.shape

    assert H % factor == 0 and W % factor == 0, "H or W modulo factor is not 0"
    x = np.transpose(input, (0, 3, 1, 2))
    x = np.reshape(x, (B, C, H // factor, factor, W // factor, factor))
    x = np.transpose(x, (0, 1, 3, 5, 2, 4))
    x = np.reshape(x, (B, C * factor * factor, H // factor, W // factor))
    x = np.transpose(x, (0, 2, 3, 1))
    return x


def unsqueeze2d(input, factor=2):
    if factor == 1:
        return input

    factor2 = factor ** 2

    B, H, W, C = input.shape

    assert C % factor2 == 0, "C modulo factor squared is not 0"
    x = np.transpose(input, (0, 3, 1, 2))
    x = np.reshape(x, (B, C // factor2, factor, factor, H, W))
    x = np.transpose(x, (0, 1, 4, 2, 5, 3))
    x = np.reshape(x, (B, C // factor2, H * factor, W * factor))
    x = np.transpose(x, (0, 2, 3, 1))
    return x


if __name__ == "__main__":
    from jax import random
    rng = random.PRNGKey(0)
    x = random.normal(rng, (1, 4, 4, 3))

    print(x.shape)

    xs = squeeze2d(x)

    print(xs.shape)

    xr = unsqueeze2d(xs)

    print(xr.shape)

    print(x[0, :, :, 0])
    print(xs[0, :, :, 0])
    print(xr[0, :, :, 0])