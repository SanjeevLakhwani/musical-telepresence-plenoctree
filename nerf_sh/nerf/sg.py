import jax
import jax.numpy as jnp


def spher2cart(r, theta, phi):
    """Convert spherical coordinates into Cartesian coordinates."""
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    return jnp.stack([x, y, z], axis=-1)  


def eval_sg(sg_lambda, sg_mu, sg_coeffs, dirs):
    """
    Evaluate spherical gaussians at unit directions
    using learnable SG basis.
    Works with jnp.
    ... Can be 0 or more batch dimensions.
    N is the number of SG basis we use.

    Output = \sigma_{i}{coeffs_i * \exp ^ {lambda_i * (\dot(mu_i, dirs) - 1)}}

    Args:
        sg_lambda: The sharpness of the SG lobes. [N] or [..., N]
        sg_mu: The directions of the SG lobes. [N, 3 or 2] or [..., N, 3 or 2]
        sg_coeffs: The coefficients of the SG (lob amplitude). [..., C, N]
        dirs: unit directions [..., 3]

    Returns:
        [..., C]
    """
    sg_lambda = jax.nn.softplus(sg_lambda)  # force lambda > 0
    # spherical coordinates -> Cartesian coordinates
    if sg_mu.shape[-1] == 2:
        theta, phi = sg_mu[..., 0], sg_mu[..., 1]
        sg_mu = spher2cart(1.0, theta, phi) # [..., N, 3]
    product = jnp.einsum(
        "...ij,...j->...i", sg_mu, dirs)  # [..., N]
    basis = jnp.exp(jnp.einsum(
        "...i,...i->...i", sg_lambda, product - 1))  # [..., N]
    output = jnp.einsum(
        "...ki,...i->...k", sg_coeffs, basis)  # [..., C]
    output /= sg_lambda.shape[-1]
    return output


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
        
    Args:
        angle: rotation angle along 3 axis (in radians). [..., 3]
    Returns:
        Rotation matrix corresponding to the euler angles. [..., 3, 3]
    """
    x, y, z = angle[..., 0], angle[..., 1], angle[..., 2]
    cosz = jnp.cos(z)
    sinz = jnp.sin(z)
    cosy = jnp.cos(y)
    siny = jnp.sin(y)
    cosx = jnp.cos(x)
    sinx = jnp.sin(x)
    zeros = jnp.zeros_like(z)
    ones = jnp.ones_like(z)
    zmat = jnp.stack([jnp.stack([cosz,  -sinz, zeros], axis=-1),
                      jnp.stack([sinz,   cosz, zeros], axis=-1),
                      jnp.stack([zeros, zeros,  ones], axis=-1)], axis=-1)
    ymat = jnp.stack([jnp.stack([ cosy, zeros,  siny], axis=-1),
                      jnp.stack([zeros,  ones, zeros], axis=-1),
                      jnp.stack([-siny, zeros,  cosy], axis=-1)], axis=-1)
    xmat = jnp.stack([jnp.stack([ ones, zeros, zeros], axis=-1),
                      jnp.stack([zeros,  cosx, -sinx], axis=-1),
                      jnp.stack([zeros,  sinx,  cosx], axis=-1)], axis=-1)
    rotMat = jnp.einsum("...ij,...jk,...kq->...iq", xmat, ymat, zmat)
    return rotMat
