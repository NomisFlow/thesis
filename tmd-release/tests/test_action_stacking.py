import jax
import jax.numpy as jnp
from impls.utils.networks import GCActor

class RaisesContext:
    def __init__(self, expected_exception):
        self.expected_exception = expected_exception

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            raise AssertionError(f"Expected {self.expected_exception}, but no exception was raised.")
        return True # Suppress exception

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
    with RaisesContext(Exception):
        actor.apply(variables, observations, goals)

    print("GCActor successfully handles prev_actions.")

if __name__ == "__main__":
    test_gc_actor_action_stacking()