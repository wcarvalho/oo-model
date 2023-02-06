"""
Monte Carlo tree search.
"""

from typing import Callable
from functools import partial

import chex
import jax
import jax.numpy as jnp
import mctx

from utils import batched_policy, env_step

from muzero import types as muzero_types
from acme.jax import networks as networks_lib


def recurrent_fn(params, rng_key: chex.Array, action: chex.Array, embedding):
    """One simulation step in MCTS."""
    del rng_key
    agent = params
    env = embedding
    env, reward = jax.vmap(env_step)(env, action)
    state = jax.vmap(lambda e: e.canonical_observation())(env)
    prior_logits, value = jax.vmap(
        lambda a, s: a(s), in_axes=(None, 0))(agent, state)
    discount = -1.0 * jnp.ones_like(reward)
    terminated = env.is_terminated()
    assert value.shape == terminated.shape
    value = jnp.where(terminated, 0.0, value)
    assert discount.shape == terminated.shape
    discount = jnp.where(terminated, 0.0, discount)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=prior_logits,
        value=value,
    )
    return recurrent_fn_output, env


def improve_policy_with_mcts(
    params: muzero_types.Params,
    rng_key: networks_lib.PRNGKey,
    prior_logits: chex.Array,
    value: chex.Array,
    state: chex.Array,
    rec_fn: mctx.RecurrentFn,
    num_simulations: int,
    maxvisit_init: int = 50,
    gumbel_scale: float = 1.0,
):
    """Improve agent policy using MCTS.

    Returns:
        An improved policy.
    """

    root = mctx.RootFnOutput(prior_logits=prior_logits,
                             value=value,
                             embedding=state)
    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=rec_fn,
        num_simulations=num_simulations,
        # invalid_actions=jax.vmap(lambda e: e.invalid_actions())(env),
        qtransform=partial(
            mctx.qtransform_completed_by_mix_value,
            value_scale=0.1,
            maxvisit_init=maxvisit_init,
            rescale_values=True,
        ),
        gumbel_scale=gumbel_scale,
    )
    return policy_output
