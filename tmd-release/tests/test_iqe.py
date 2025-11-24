
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

# Add the project root to the path so we can import the modules
# We need 'impls' to be in the path because the code uses 'from utils...' and 'from agents...'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../impls')))

from agents.tmd import TMDAgent

# Mock classes to simulate the agent's network dependencies
class MockNetwork:
    def __init__(self, alpha_val=0.5):
        # Inverse sigmoid to get raw value
        self.alpha_raw_val = jnp.log(alpha_val / (1.0 - alpha_val))

    def select(self, name):
        if name == 'alpha_raw':
            return lambda: self.alpha_raw_val
        raise ValueError(f"Unexpected select: {name}")

@struct.dataclass
class MockConfig:
    config_dict: dict

    def __getitem__(self, key):
        return self.config_dict[key]
    
    def get(self, key, default=None):
        return self.config_dict.get(key, default)

class TestIQE:
    @pytest.fixture
    def agent(self):
        # Create a minimal agent with IQE enabled
        config = MockConfig({
            'components': 4,
            'use_iqe': True,
            'lr': 3e-4, # Dummy
        })
        # Initialize with alpha=0.5
        network = MockNetwork(alpha_val=0.5)
        
        # TMDAgent is a flax PyTreeNode (frozen), so we instantiate it directly 
        # passing the fields. config is a non-pytree field.
        agent = TMDAgent(
            rng=jax.random.PRNGKey(0),
            network=network,
            config=config
        )
        
        return agent

    def test_self_distance_zero(self, agent):
        """d(x, x) should be 0."""
        # Shape: (batch, dim)
        # dim must be divisible by components (4)
        x = jax.random.normal(jax.random.PRNGKey(1), (10, 16))
        d = agent.iqe_distance(x, x)
        
        # Using a small tolerance for float precision
        assert jnp.allclose(d, 0.0, atol=1e-5)

    def test_positivity(self, agent):
        """d(x, y) >= 0."""
        k1, k2 = jax.random.split(jax.random.PRNGKey(1))
        x = jax.random.normal(k1, (10, 16))
        y = jax.random.normal(k2, (10, 16))
        d = agent.iqe_distance(x, y)
        
        assert jnp.all(d >= -1e-6) # Allow tiny float errors

    def test_asymmetry(self, agent):
        """d(x, y) != d(y, x) generally."""
        key = jax.random.PRNGKey(3)
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (1, 16))
        y = jax.random.normal(k2, (1, 16))
        
        d_xy = agent.iqe_distance(x, y)
        d_yx = agent.iqe_distance(y, x)
        
        # They should differ substantially for random vectors
        assert not jnp.allclose(d_xy, d_yx, atol=1e-4)

    def test_triangle_inequality_approx(self, agent):
        """d(x, z) <= d(x, y) + d(y, z)."""
        # IQE is a quasimetric, so triangle inequality should hold.
        key = jax.random.PRNGKey(4)
        k1, k2, k3 = jax.random.split(key, 3)
        x = jax.random.normal(k1, (100, 16))
        y = jax.random.normal(k2, (100, 16))
        z = jax.random.normal(k3, (100, 16))
        
        d_xy = agent.iqe_distance(x, y)
        d_yz = agent.iqe_distance(y, z)
        d_xz = agent.iqe_distance(x, z)
        
        # Check violation count
        violations = d_xz > (d_xy + d_yz) + 1e-5
        num_violations = jnp.sum(violations)
        
        # Ideally 0, but allow a few due to numerical issues or alpha blending properties?
        # Strict quasimetrics must satisfy this.
        assert num_violations == 0, f"Found {num_violations} triangle inequality violations"

    def test_batched_computation(self, agent):
        """Test correctness of broadcasting."""
        k1, k2 = jax.random.split(jax.random.PRNGKey(5))
        x = jax.random.normal(k1, (5, 16))
        y = jax.random.normal(k2, (5, 16))
        
        # Pairwise vs Elementwise
        # iqe_distance in TMD implementation seems to expect (..., dim) and broadcast
        # Let's check if it handles broadcasting correctly manually
        
        # Elementwise
        d_elem = agent.iqe_distance(x, y)
        assert d_elem.shape == (5,)
        
        # Broadcasting: dist between x[0] and all y
        d_broad = agent.iqe_distance(x[0:1], y)
        assert d_broad.shape == (5,)
        
        assert jnp.allclose(d_broad[0], d_elem[0])

    def test_gradients(self, agent):
        """Ensure gradients propagate through the distance function."""
        k1, k2 = jax.random.split(jax.random.PRNGKey(7))
        x = jax.random.normal(k1, (16,))
        y = jax.random.normal(k2, (16,))
        
        def loss_fn(x_in, y_in):
            return agent.iqe_distance(x_in, y_in)
        
        grad_fn = jax.grad(loss_fn, argnums=(0, 1))
        gx, gy = grad_fn(x, y)
        
        assert not jnp.allclose(gx, 0.0)
        assert not jnp.allclose(gy, 0.0)
        assert jnp.all(jnp.isfinite(gx))
        assert jnp.all(jnp.isfinite(gy))

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
