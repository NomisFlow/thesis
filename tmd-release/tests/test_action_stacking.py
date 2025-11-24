import jax
import jax.numpy as jnp
from impls.utils.networks import GCActor

def test_gc_actor_action_stacking():
    # Define dimensions
    hidden_dims = [256, 256]
    action_dim = 8
    observation_dim = 32
    batch_size = 4

    # Initialize GCActor with standard configuration
    actor = GCActor(
        hidden_dims=hidden_dims,
        action_dim=action_dim,
        state_dependent_std=False,
        const_std=True
    )

    # Create dummy data
    key = jax.random.PRNGKey(0)
    observations = jax.random.normal(key, (batch_size, observation_dim))
    goals = jax.random.normal(key, (batch_size, observation_dim))
    prev_actions = jax.random.normal(key, (batch_size, action_dim))
    
    # Initialize parameters WITH prev_actions
    variables = actor.init(key, observations, goals, prev_actions=prev_actions)
    
    # Forward pass with stacked observations -> Should Succeed
    dist = actor.apply(variables, observations, goals, prev_actions=prev_actions)
    actions = dist.mode()
    assert actions.shape == (batch_size, action_dim)
    
    # Forward pass WITHOUT prev_actions using the SAME variables -> Should Fail
    # This confirms that the network structure (specifically the first layer weights) 
    # depends on the input size including prev_actions.
    try:
        actor.apply(variables, observations, goals)
        print("FAILURE: Network accepted input without prev_actions despite being initialized with them.")
        exit(1)
    except Exception as e: # Expecting size mismatch error from JAX/Flax
        print("SUCCESS: Network correctly rejected input missing prev_actions.")

    print("GCActor successfully handles prev_actions.")

if __name__ == "__main__":
    test_gc_actor_action_stacking()