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
    with RaisesContext(Exception):
        actor.apply(variables, observations, goals)

    # --- New Test Case: Image Observations (Broadcasting check) ---
    image_shape = (64, 64, 3)
    observations_img = jax.random.normal(key, (batch_size, *image_shape))
    goals_img = jax.random.normal(key, (batch_size, *image_shape))
    
    # Init with images + prev_actions (without encoder, should broadcast)
    variables_img = actor.init(key, observations_img, goals_img, prev_actions=prev_actions)
    dist_img = actor.apply(variables_img, observations_img, goals_img, prev_actions=prev_actions)
    actions_img = dist_img.mode()
    
    # Output of MLP on (B, H, W, C) is (B, H, W, A)
    # But GCActor applies Dense to the last dim.
    # Then mean_net applies Dense to last dim.
    # distrax.MultivariateNormalDiag expects loc to be (B, A)? 
    # No, if loc is (B, H, W, A), it creates a batch of distributions.
    # GCActor returns `distribution`.
    # If we just want actions, typically we might expect (B, A).
    # BUT if inputs are spatial, outputs will be spatial.
    # This test just confirms it doesn't CRASH.
    
    assert actions_img.shape == (batch_size, 64, 64, action_dim)

    print("GCActor successfully handles prev_actions with vectors and images.")

if __name__ == "__main__":
    test_gc_actor_action_stacking()
