import torch
from torch import nn

from layers.Embed import PatchEmbedding
from layers.RevIN import RevIN
from .adapter import FeatureAffineAdapter
from .moe_layer import TemporalDynamicMoELayer

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        padding = self.stride
        self.patch_embedding = PatchEmbedding(
            configs.d_model, 
            self.patch_len, 
            self.stride, 
            padding, 
            configs.dropout
        )

        self.num_patches = int((configs.seq_len - self.patch_len) / self.stride + 2)

        # Optional private recalibration between the shared encoder and the MoE
        # stack.  ``none`` reproduces the historical arm bit for bit; ``private``
        # adds 2*d_model per-client parameters initialised to the identity.
        self.adapter_mode = str(getattr(configs, 'fedtdo_adapter', 'none') or 'none')
        if self.adapter_mode not in ('none', 'private'):
            raise ValueError(f'unknown fedtdo_adapter {self.adapter_mode!r}')
        self.adapter = (FeatureAffineAdapter(configs.d_model)
                        if self.adapter_mode == 'private' else None)

        # Temporal MoE
        self.num_temporal_moe_layers = getattr(configs, 'num_temporal_moe_layers', 2)
        num_experts = getattr(configs, 'num_experts', 4)
        num_drift_experts = getattr(configs, 'num_drift_experts', 3)
        num_rnn_layers = getattr(configs, 'num_rnn_layers', 2)
        top_k = getattr(configs, 'top_k', 2)

        self.drift_detection_enabled = getattr(configs, 'enable_drift_detection', True)
        self.drift_window_size = getattr(configs, 'drift_window_size', 576)
        self.drift_history_size = getattr(configs, 'drift_history_size', 20)
        self.drift_k_sigma = getattr(configs, 'drift_k_sigma', 2.0)

        self.train_epochs = getattr(configs, 'train_epochs', 10)
        self.learning_rate = getattr(configs, 'learning_rate', 0.0001)
        self.finetune_patience = getattr(configs, 'finetune_patience', 5)

        self.cycle_length = configs.cycle_length

        self.channel_independence = getattr(configs, 'channel_independence', 1)
        self.use_relation_layer = getattr(configs, 'use_relation_layer', 1)
        
        self.temporal_moe_layers = nn.ModuleList([
            TemporalDynamicMoELayer(
                d_model=configs.d_model,
                num_patches=self.num_patches,
                n_vars=configs.enc_in,
                cycle_length=self.cycle_length,
                num_experts=num_experts,
                num_drift_experts=num_drift_experts,
                dropout=configs.dropout,
                num_rnn_layers=num_rnn_layers,
                drift_window_size=self.drift_window_size,
                use_relation_layer=self.use_relation_layer,
                enable_drift_detection=bool(self.drift_detection_enabled),
                top_k=top_k
            ) for _ in range(self.num_temporal_moe_layers)
        ])

        revin_affine = getattr(configs, 'revin_affine', True)
        self.revin_layer = RevIN(configs.enc_in, eps=1e-5, affine=revin_affine)

        self.register_buffer('global_patch_counter', torch.tensor(0, dtype=torch.long))

        self.head_input_dim = configs.d_model * int((configs.seq_len - self.patch_len) / self.stride + 2)
        if self.channel_independence:
            self.projection_layers = nn.ModuleList([
                nn.Linear(self.head_input_dim, configs.pred_len)
                for _ in range(configs.enc_in)
            ])
        else:
            self.projection_layer = nn.Linear(self.head_input_dim, configs.pred_len)

    def _encode_patches(self, x_enc, x_mark_enc):
        """RevIN-normalise then patch-embed.

        Returns:
            enc_out [B*N, P, D], RevIN stats, cycle indices, B, N
        """
        B, S, N = x_enc.shape
        x_norm, means, stdev = self.revin_layer.normalize(x_enc)  # [B, S, N]
        x_patches = x_norm.permute(0, 2, 1)  # [B, N, S]
        enc_out, _ = self.patch_embedding(x_patches)  # [B*N, P, D]

        # The adapter sits *inside* the encoding step so every consumer of the
        # representation - MoE experts, drift detector, regime signature - sees
        # the same features.  It is deliberately excluded from
        # ``get_encoder_state_dict``: the trunk is shared, this is not.
        if self.adapter is not None:
            enc_out = self.adapter(enc_out)

        if x_mark_enc is not None:
            first_timestamp = x_mark_enc[:, 0, :]  # [B, mark_dim]
            patch_cycle_indices = (first_timestamp[:, 0] % self.cycle_length).long()
        else:
            patch_cycle_indices = torch.zeros(B, dtype=torch.long, device=x_enc.device)
        return enc_out, means, stdev, patch_cycle_indices, B, N

    def _decode_head(self, enc_out, B, N, means, stdev):
        P = enc_out.shape[1]
        D = enc_out.shape[2]
        enc_out = enc_out.view(B, N, P, D)  # [B, N, P, D]
        enc_out_flat = enc_out.reshape(B, N, P * D)  # [B, N, D*P]

        if self.channel_independence:
            predictions = torch.stack([
                self.projection_layers[i](enc_out_flat[:, i, :])
                for i in range(N)
            ], dim=2)  # [B, pred_len, N]
        else:
            predictions = self.projection_layer(enc_out_flat)
            predictions = predictions.permute(0, 2, 1)  # [B, pred_len, N]

        return self.revin_layer.denormalize(predictions, means, stdev)

    def forecast(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None,
                 external_expert=None, transfer_layer=None, mix_alpha=0.0,
                 observe_online_drift=False, shadow_mode=False):
        """
        Args:
            x_enc: [B, S, N]
            x_mark_enc: [B, S, mark_dim]
        Returns:
            predictions: [B, pred_len, N]
        """
        enc_out, means, stdev, patch_cycle_indices, B, N = self._encode_patches(
            x_enc, x_mark_enc
        )

        temporal_hidden_states = [None] * self.num_temporal_moe_layers

        final_temporal_hidden_states = []

        batch_start_patch_idx = self.global_patch_counter.item()
        BN = enc_out.shape[0]
        P = enc_out.shape[1]
        
        for layer_idx, moe_layer in enumerate(self.temporal_moe_layers):
            enc_out, new_hidden = moe_layer(
                enc_out,
                hidden_state=temporal_hidden_states[layer_idx],
                batch_start_patch_idx=batch_start_patch_idx,
                patch_cycle_indices=patch_cycle_indices,
                external_expert=(external_expert if layer_idx == transfer_layer else None),
                external_alpha=mix_alpha,
                observe_online_drift=(observe_online_drift and layer_idx == 0),
                shadow_mode=shadow_mode,
            )
            final_temporal_hidden_states.append(new_hidden)
        
        if self.training:
            self.global_patch_counter += BN * P

        self._last_temporal_hidden_states = final_temporal_hidden_states
        self._last_enc_out = enc_out.detach()

        # The FedTDO state signature comes from the shared encoder's hidden
        # representation plus the causal drift signal, not from raw inputs
        # (plan §6.1).  It is cached so the client can reuse it for regime
        # memory without paying a second encoder pass.
        if observe_online_drift:
            drift_signal = self.online_drift_signal()
            self.last_regime_signature = self._compute_regime_signature(
                enc_out, float(drift_signal.get('score', 0.0))
            )

        return self._decode_head(enc_out, B, N, means, stdev)

    @torch.no_grad()
    def forecast_with_external_expert(self, x_enc, x_mark_enc, expert_capsule,
                                     mix_alpha=0.5, observe_online_drift=False):
        """Produce a non-mutating counterfactual prediction with one capsule."""
        external_expert = expert_capsule.materialize(x_enc.device)
        return self.forecast(
            x_enc, x_mark_enc,
            external_expert=external_expert,
            transfer_layer=expert_capsule.layer_idx,
            mix_alpha=mix_alpha,
            observe_online_drift=observe_online_drift,
            shadow_mode=True,
        )

    def get_encoder_state_dict(self):
        """Shared PatchEmbedding weights for federated encoder versioning."""
        return {k: v.detach().cpu().clone() for k, v in self.patch_embedding.state_dict().items()}

    def load_encoder_state_dict(self, state_dict, strict=True):
        self.patch_embedding.load_state_dict(state_dict, strict=strict)

    def adapter_state_dict(self):
        """Private adapter weights - never uploaded, never averaged."""
        if self.adapter is None:
            return {}
        return {k: v.detach().cpu().clone() for k, v in self.adapter.state_dict().items()}

    def adapter_parameters(self):
        return [] if self.adapter is None else list(self.adapter.parameters())

    def freeze_encoder(self):
        # Only the trunk is frozen.  The adapter stays trainable on purpose:
        # it is the client's entire budget for undoing the shared encoder's
        # mismatch (2*d_model parameters instead of a whole private network).
        for parameter in self.patch_embedding.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self):
        for parameter in self.patch_embedding.parameters():
            parameter.requires_grad = True

    def observe_drift_online(self, x_enc, x_mark_enc=None, slide_on_drift=True):
        """Causal online drift observation used by the FedTDO state signature.

        Only already-revealed inputs are touched (no labels), it runs under
        ``no_grad`` and never advances ``global_patch_counter``, so a shadow
        observation cannot perturb the deployed model's bookkeeping.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            enc_out, _, _, _, B, N = self._encode_patches(x_enc, x_mark_enc)
            BN, P, D = enc_out.shape
            batch_start_patch_idx = self.global_patch_counter.item()

            layer_results = []
            max_score = 0.0
            any_drift = False
            for moe_layer in self.temporal_moe_layers:
                if moe_layer.drift_detector is None:
                    continue
                sample_size = min(max(1, BN // 10), 50)
                step = max(1, BN // sample_size)
                indices = torch.arange(0, BN, step, device=enc_out.device)[:sample_size]
                start_patch_indices = indices * P + batch_start_patch_idx
                result = moe_layer.drift_detector.observe_online(
                    enc_out[indices].detach(),
                    start_patch_indices,
                    slide_on_drift=slide_on_drift,
                )
                layer_results.append(result)
                max_score = max(max_score, float(result.get('score', 0.0)))
                if result.get('drift'):
                    any_drift = True

            signature = self._compute_regime_signature(enc_out, max_score)

        if was_training:
            self.train()

        return {
            'drift_detected': any_drift if layer_results else None,
            'drift_score_01': max_score,
            'layer_results': layer_results,
            'hidden_signature': signature,
        }

    def _compute_regime_signature(self, enc_out, drift_score_01):
        """Low-dim signature from patch embeddings for regime-memory matching."""
        mean_feat = enc_out.mean(dim=(0, 1))  # [D]
        std_feat = enc_out.std(dim=(0, 1), unbiased=False)  # [D]
        k = min(16, mean_feat.numel())
        signature = torch.cat([
            mean_feat[:k],
            std_feat[:k],
            torch.tensor([float(drift_score_01)], device=enc_out.device, dtype=mean_feat.dtype),
        ])
        return signature.detach().cpu()

    def online_drift_signal(self):
        """Return the most recently observed causal drift signal."""
        if not self.temporal_moe_layers:
            return {'ready': False, 'drift': False, 'mmd': 0.0, 'score': 0.0}
        return getattr(
            self.temporal_moe_layers[0], 'last_online_drift',
            {'ready': False, 'drift': False, 'mmd': 0.0, 'score': 0.0},
        )
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            raise NotImplementedError(f"Task {self.task_name} not supported")
    
    def enable_drift_detection(self, window_size=576, history_size=50, k_sigma=3.0):
        for layer_idx, moe_layer in enumerate(self.temporal_moe_layers):
            moe_layer.enable_drift_detection(
                window_size=window_size,
                history_size=history_size,
                k_sigma=k_sigma
            )
    
    def check_and_handle_drift(self, train_mode=True):
        for moe_layer in self.temporal_moe_layers:
            moe_layer.check_and_handle_drift(
                finetune_epochs=self.train_epochs,
                finetune_lr=self.learning_rate,
                finetune_patience=self.finetune_patience,
                train_mode=train_mode
            )
