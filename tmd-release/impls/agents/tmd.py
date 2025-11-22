from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    DiscreteStateActionRepresentation,
    GCActor,
    GCDiscreteActor,
    Param,
    SequenceEncoder,
    StateRepresentation,
)


class TMDAgent(flax.struct.PyTreeNode):
    """Temporal Metric Distillation (TMD) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @jax.jit
    def mrn_distance(self, x, y):
        K = self.config['components']
        assert x.shape[-1] % K == 0

        @jax.jit
        def mrn_distance_component(x, y):
            eps = 1e-6
            d = x.shape[-1]
            mask = jnp.arange(d) < d // 2
            max_component = jnp.max(jax.nn.relu((x - y) * mask), axis=-1)
            l2_component = jnp.sqrt(jnp.square((x - y) * (1 - mask)).sum(axis=-1) + eps)
            assert max_component.shape == l2_component.shape
            return max_component + l2_component

        x_split = jnp.stack(jnp.split(x, K, axis=-1), axis=-1)
        y_split = jnp.stack(jnp.split(y, K, axis=-1), axis=-1)
        dists = jax.vmap(mrn_distance_component, in_axes=(-1, -1), out_axes=-1)(x_split, y_split)

        return dists.mean(axis=-1)

    def iqe_distance(self, x, y):
        k = self.config['components']
        alpha_raw = self.network.select('alpha_raw')()
        alpha = jax.nn.sigmoid(alpha_raw)
        reshape = (x.shape[-1] // k, k)
        x = jnp.reshape(x, (*x.shape[:-1], *reshape))
        y = jnp.reshape(y, (*y.shape[:-1], *reshape))
        valid = x < y
        D = x.shape[-1]
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        neg_inc_copies = jnp.take_along_axis(valid, ixy % D, axis=-1) * jnp.where(ixy < D, -1, 1)
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = (neg_inp_copies < 0) * (-1.0)
        neg_incf = jnp.concatenate([neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1)
        components = (sxy * neg_incf).sum(-1)
        result = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)
        return result

    @jax.jit
    def distance(self, x, y):
        x, y = jnp.broadcast_arrays(x, y)
        if self.config['use_iqe']:
            return self.iqe_distance(x, y)
        else:
            return self.mrn_distance(x, y)

    @jax.jit
    def critic_loss(self, batch, grad_params):
        batch_size = batch['observations'].shape[0]

        phi = self.network.select('phi')(batch['observations'], batch['actions'], params=grad_params)
        psi_s = self.network.select('psi')(batch['observations'], params=grad_params)
        psi_next = self.network.select('psi')(batch['next_observations'], params=grad_params)
        psi_g = self.network.select('psi')(batch['value_goals'], params=grad_params)

        if len(phi.shape) == 2:  # Non-ensemble
            phi = phi[None, ...]
            psi_s = psi_s[None, ...]
            psi_next = psi_next[None, ...]
            psi_g = psi_g[None, ...]

        dist = self.distance(phi[:, :, None], psi_g[:, None, :])
        logits = -dist / jnp.sqrt(phi.shape[-1])
        # logits.shape is (e, B, B) with one term for positive pair and (B - 1) terms for negative pairs in each row.

        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.softmax_cross_entropy(logits=_logits.T, labels=I),
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)
        if self.config['stopgrad_phi_invariance']:
            action_dist = self.distance(psi_s, jax.lax.stop_gradient(phi))
        else:
            action_dist = self.distance(psi_s, phi)

        action_invariance_loss = jnp.mean(action_dist)

        dist_next = self.distance(psi_next[:, :, None], psi_g[:, None, :])

        t = self.config['t']
        gamma = self.config['discount']
        if self.config['stopgrad_psi_backup']:
            dist = self.distance(phi[:, :, None], jax.lax.stop_gradient(psi_g[:, None, :]))
        dist_next = jax.lax.stop_gradient(dist_next)

        delta = dist - dist_next
        mask = delta > t
        delta_clipped = jnp.where(mask, t, delta)
        divergence = jnp.where(mask, delta, gamma * jnp.exp(delta_clipped) - dist)

        dw = self.config['diag_backup']
        divergence = divergence * (1 - dw) + jnp.diagonal(divergence, axis1=1, axis2=2)[..., None] * dw
        backup_loss = jnp.mean(divergence)

        critic_loss = contrastive_loss + action_invariance_loss + self.config['zeta'] * backup_loss

        logits = jnp.mean(logits, axis=0)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return (
            (contrastive_loss, backup_loss, action_invariance_loss),
            critic_loss,
            {
                'contrastive_loss': contrastive_loss,
                'action_invariance_loss': action_invariance_loss,
                'backup_loss': backup_loss,
                'critic_loss': critic_loss,
                'binary_accuracy': jnp.mean((logits > 0) == I),
                'categorical_accuracy': jnp.mean(correct),
                'logits_pos': logits_pos,
                'logits_neg': logits_neg,
                'logits': logits.mean(),
                'dist': dist.mean(),
                'biggest_diff_in_dist': jnp.max(dist - dist_next),
            },
        )

    @jax.jit
    def actor_loss(self, batch, grad_params, rng=None):
        # Maximize log Q if actor_log_q is True (which is default).
        if self.config['use_latent']:
            psi_s, psi_g = (
                self.network.select('psi')(batch['observations'], params=grad_params),
                self.network.select('psi')(batch['actor_goals'], params=grad_params),
            )
            if len(psi_s.shape) == 3:
                psi_s = jnp.mean(psi_s, axis=0)
                psi_g = jnp.mean(psi_g, axis=0)
            if self.config['freeze_enc_for_actor_grad']:
                psi_s, psi_g = jax.lax.stop_gradient(psi_s), jax.lax.stop_gradient(psi_g)
            dist = self.network.select('actor')(psi_s, psi_g, params=grad_params)
        else:
            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

        phi = self.network.select('phi')(batch['observations'], q_actions)
        psi = self.network.select('psi')(batch['actor_goals'])
        q1, q2 = -self.distance(phi, psi)
        q = jnp.minimum(q1, q2)

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        log_prob = dist.log_prob(batch['actions'])

        bc_loss = -(self.config['alpha'] * log_prob).mean()

        actor_loss = q_loss + bc_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def aux_loss(self, batch, grad_params, rng):
        batch_size = batch['actions'].shape[0]  # Retrieves batch size

        # Flatten the sequence and batch dimensions
        obs_seq = batch['observations_seq']
        flat_obs_seq = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size * self.config['sequence_length'], *x.shape[2:])),
            obs_seq
        )

        # Encode the flattened sequence
        flat_psi_seq = self.network.select('psi')(flat_obs_seq, params=grad_params)
        
        # Reshape back to (ensemble, batch, sequence, features)
        ensemble_size = flat_psi_seq.shape[0]
        latent_dim = flat_psi_seq.shape[-1]
        psi_seq = flat_psi_seq.reshape((ensemble_size, batch_size, self.config['sequence_length'], latent_dim))

        # Get an RNG for the sequence encoder's carry initialization.
        rng, sequence_encoder_rng = jax.random.split(rng) # 'rng' is from total_loss method

        # Vmap the sequence encoder over the ensemble dimension
        sequence_encoder_vmapped = jax.vmap(
            lambda seq: self.network.select('sequence_encoder')(seq, params=grad_params, rngs={'carry_init': sequence_encoder_rng})
        )
        context_vector = sequence_encoder_vmapped(psi_seq)
        
        # Target latent state psi(s_{i+k})
        psi_target = self.network.select('psi')(batch['trajectory_goals'], params=grad_params)

        # Ensure context_vector and psi_target have compatible shapes for contrastive loss
        # Typically, contrastive loss expects (num_ensembles, batch_size, embedding_dim)
        # If there's no ensemble (i.e. shape is (batch_size, latent_dim)), add a dimension.
        # Expected shapes:
        # context_vector: (ensemble_size, batch_size, latent_dim) or (batch_size, latent_dim)
        # psi_target: (ensemble_size, batch_size, latent_dim) or (batch_size, latent_dim)
        if len(context_vector.shape) == 2:
            context_vector = context_vector[None, ...]
        if len(psi_target.shape) == 2:
            psi_target = psi_target[None, ...]

        # Compute contrastive loss
        # Use a similar approach to the critic_loss for contrastive objective
        dist = self.distance(context_vector[:, :, None], psi_target[:, None, :])
        logits = -dist / jnp.sqrt(context_vector.shape[-1])

        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.softmax_cross_entropy(logits=_logits.T, labels=I),
        )(logits)
        aux_loss = jnp.mean(contrastive_loss)
        
        # Logging information
        logits = jnp.mean(logits, axis=0)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return aux_loss, {
            'aux_contrastive_loss': aux_loss,
            'aux_logits_pos': logits_pos,
            'aux_logits_neg': logits_neg,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, contrastive_only=False):
        info = {}
        rng = rng if rng is not None else self.rng

        (contrastive_loss, backup_loss, action_invariance_loss), critic_loss, critic_info = self.critic_loss(
            batch, grad_params
        )
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        
        # Pass rng to aux_loss
        aux_loss, aux_info = self.aux_loss(batch, grad_params, rng)
        for k, v in aux_info.items():
            info[f'aux/{k}'] = v

        loss = critic_loss + actor_loss + self.config['beta'] * aux_loss
        return loss, info

    @jax.jit
    def update(self, batch, contrastive_only=False):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, contrastive_only=contrastive_only)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        if self.config['use_latent']:
            psi_s, psi_g = self.network.select('psi')(observations), self.network.select('psi')(goals)
            if len(psi_s.shape) == 2:  # in inference, we don't have batch dimension
                psi_s = jnp.mean(psi_s, axis=0)
                psi_g = jnp.mean(psi_g, axis=0)
            dist = self.network.select('actor')(psi_s, psi_g, temperature=temperature)
        else:
            dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, carry_init_rng = jax.random.split(rng, 3)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            encoders['state'] = encoder_module()
        if config['discrete']:
            phi_def = DiscreteStateActionRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state'),
                action_dim=action_dim,
            )
            psi_def = DiscreteStateActionRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state'),
                action_dim=action_dim,
            )
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            phi_def = StateRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state'),
            )
            psi_def = StateRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state'),
            )
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )
        if config['use_iqe']:
            network_info = dict(
                actor=(actor_def, (ex_observations, ex_goals)),
                phi=(phi_def, (ex_observations, ex_actions)),
                psi=(psi_def, (ex_goals,)),
                alpha_raw=(Param(), ()),
            )
        else:
            if config['use_latent']:
                embed = jnp.zeros((1, config['latent_dim']))
                network_info = dict(
                    actor=(actor_def, (embed, embed)),
                    phi=(phi_def, (ex_observations, ex_actions)),
                    psi=(psi_def, (ex_goals,)),
                )
            else:
                network_info = dict(
                    actor=(actor_def, (ex_observations, ex_goals)),
                    phi=(phi_def, (ex_observations, ex_actions)),
                    psi=(psi_def, (ex_goals,)),
                )
        sequence_encoder_def = SequenceEncoder(
            hidden_dims=config['sequence_encoder_hidden_dims'][0],
            output_dim=config['latent_dim'],
        )

        networks = {k: v[0] for k, v in network_info.items()}
        networks['sequence_encoder'] = sequence_encoder_def
        network_args = {k: v[1] for k, v in network_info.items()}
        # Placeholder for sequence input (batch_size, sequence_length, latent_dim)
        network_args['sequence_encoder'] = (jnp.zeros((1, 1, config['latent_dim'])),)

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        
        # Force initialization on CPU to avoid QR decomposition issues on Metal
        with jax.default_device(jax.devices('cpu')[0]):
            network_params = network_def.init({'params': init_rng, 'carry_init': carry_init_rng}, **network_args)['params']
        
        # Move params back to default device
        network_params = jax.device_put(network_params)
        
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='tmd',  # Agent name.
            lr=3e-4,
            components=8,  # Number of components to average in the MRN/IQE distance ensemble.
            batch_size=512,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            sequence_encoder_hidden_dims=(256,),  # Hidden dimensions for the sequence encoder.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            zeta=0.05,  # Weight for TMD backup and invariance losses.
            beta=0.1,  # Weight for the auxiliary loss.
            t=3.0,  # Clipping threshold for the backup LINEX loss.
            diag_backup=0.5,  # Weighting of backups on diagonal (i.e., for s,g ~ p(s,g)) vs. off-diagonal (i.e., for s,g ~ p(s)p(g)).
            stopgrad_psi_backup=False,  # Whether to stop gradient for psi in the backup loss.
            stopgrad_phi_invariance=False,  # Whether to stop gradient for phi in the invariance loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            sequence_length=5,  # Length of the observation sequence for auxiliary loss.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            use_iqe=False,  # Whether to use IQE distance or MRN distance
            use_latent=False,  # Whether to use latent for policy action sampling
            freeze_enc_for_actor_grad=False,  # Whether to stop grad for actor when using encoder
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
