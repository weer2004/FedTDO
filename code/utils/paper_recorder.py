"""Paper-grade recording of a single prequential run.

Design constraints (see the task specification):

* nothing here changes the algorithm -- every method is called with the same
  arguments, in the same order, as before this module existed;
* every recorded row is an event that *actually happened*; nothing is inferred
  from the configuration;
* a disabled recorder (``enabled=False``) makes every call a no-op, so the
  harness can run without paying for I/O;
* the derived numbers (cumulative MSE, rolling MSE) are recomputable from the
  raw rows that are written here.

Layout::

    <root>/raw/<dataset>/<method>/seed_<seed>/run_<run_id>/
        config.json
        run_summary.json
        per_window_metrics.csv
        online_updates.csv
        candidate_evaluations.csv
        feedback_rewards.csv
        transfer_events.csv
        communication_events.csv
        cost_events.csv
        regime_events.csv
        predictions/client_*.npz

A separate ``run_<run_id>`` directory per invocation is what keeps two
concurrent sweeps from overwriting each other's CSVs.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Anything that is not safe inside a file name.  Column names are used verbatim as
# client ids, and several Weather columns contain '/'.
_FILENAME_UNSAFE = re.compile(r'[^0-9A-Za-z._-]+')

SCHEMAS: Dict[str, List[str]] = {
    'per_window_metrics': [
        'run_id', 'method', 'dataset', 'seed', 'client_id', 'window_idx',
        'sse', 'sae', 'num_targets', 'window_mse', 'window_mae',
        'cumulative_mse', 'cumulative_mae',
        'update_count', 'gradient_steps', 'label_reveal_idx',
        'target_start_idx', 'target_end_idx', 'regime_id',
    ],
    'online_updates': [
        'run_id', 'client_id', 'window_idx', 'update_type', 'optimizer',
        'learning_rate', 'batch_size', 'gradient_steps', 'total_gradient_steps',
        'update_loss', 'update_duration_ms', 'target_start_idx',
        'target_end_idx', 'label_available_idx', 'update_time_idx',
    ],
    'candidate_evaluations': [
        'eval_id', 'run_id', 'client_id', 'window_idx', 'candidate_expert_id',
        'source_client_id', 'candidate_version', 'predicted_gain',
        'estimated_uncertainty', 'shadow_mse', 'evaluation_cost',
        'evaluation_duration_ms', 'selected', 'transferred',
    ],
    'feedback_rewards': [
        'eval_id', 'run_id', 'client_id', 'candidate_expert_id',
        'evaluation_window', 'feedback_release_window', 'predicted_gain',
        'realized_gain', 'reward', 'reward_basis',
    ],
    'transfer_events': [
        'event_id', 'run_id', 'client_id', 'window_idx', 'event_type',
        'source_client_id', 'expert_id', 'old_expert_id', 'new_expert_id',
        'estimated_gain', 'realized_gain', 'transfer_bytes', 'accepted',
        'decision_reason',
    ],
    'communication_events': [
        'event_id', 'run_id', 'client_id', 'window_idx', 'round_idx',
        'direction', 'payload_type', 'expert_id', 'source_client_id',
        'bytes_sent', 'bytes_received', 'duration_ms',
    ],
    'cost_events': [
        'run_id', 'client_id', 'window_idx', 'operation', 'duration_ms',
        'estimated_flops', 'peak_memory_mb',
    ],
    'regime_events': [
        'run_id', 'client_id', 'window_idx', 'event_type',
        'detected_regime_id', 'previous_regime_id', 'detection_score',
        'detection_threshold', 'related_expert_id',
    ],
}


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ''
    except (OSError, subprocess.SubprocessError):
        return ''


class PaperRecorder:
    """Append-only event recorder for one (method, dataset, seed) run."""

    def __init__(self, *, root: str, method: str, dataset: str, seed: int,
                 run_kind: str = 'fedtdo', enabled: bool = True,
                 extra_config: Optional[dict] = None):
        self.enabled = bool(enabled)
        self.method = str(method)
        self.dataset = str(dataset)
        self.seed = int(seed)
        self.run_kind = run_kind
        self.run_id = (f'{self.method}_{self.dataset}_s{self.seed}_'
                       f'{time.strftime("%Y%m%d-%H%M%S")}_{uuid.uuid4().hex[:6]}')
        self.started = time.time()
        self._files = {}
        self._writers = {}
        self._preds: Dict[str, Dict[str, list]] = {}
        self._cum: Dict[str, List[float]] = {}
        self._update_count: Dict[str, int] = {}
        self._grad_steps: Dict[str, int] = {}
        self._last_eval_id: Dict[str, List[dict]] = {}
        self._expert_source: Dict[str, str] = {}
        self._eval_counter = 0
        self._event_counter = 0
        self._comm_counter = 0
        self._totals = {
            'evaluated_windows': 0, 'online_update_events': 0,
            'gradient_steps': 0, 'candidate_evaluations': 0,
            'transfer_events': 0, 'upload_bytes': 0, 'download_bytes': 0,
            'forecasting_time': 0.0, 'shadow_evaluation_time': 0.0,
            'adaptation_time': 0.0, 'transfer_time': 0.0,
            'encoder_sync_time': 0.0,
        }
        if not self.enabled:
            self.dir = Path(root) / 'raw' / self.dataset / self.method / 'disabled'
            return
        self.dir = (Path(root) / 'raw' / self.dataset / self.method
                    / f'seed_{self.seed}' / f'run_{self.run_id}')
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / 'predictions').mkdir(exist_ok=True)
        self._config = {
            'run_id': self.run_id,
            'method': self.method,
            'dataset': self.dataset,
            'seed': self.seed,
            'run_kind': self.run_kind,
            'git_commit': _git_commit(),
            'python': sys_version(),
            'host': platform.node(),
            'start_time': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(self.started)),
            'end_time': None,
            'experiment_status': 'running',
        }
        if extra_config:
            self._config.update(extra_config)
        self._write_json('config.json', self._config)

    # ------------------------------------------------------------- plumbing
    def _path(self, name: str) -> Path:
        return self.dir / f'{name}.csv'

    def _writer(self, name: str):
        if name not in self._writers:
            handle = open(self._path(name), 'w', newline='', encoding='utf-8')
            self._files[name] = handle
            writer = csv.DictWriter(handle, fieldnames=SCHEMAS[name],
                                    extrasaction='ignore')
            writer.writeheader()
            self._writers[name] = writer
        return self._writers[name]

    def _append(self, name: str, row: dict):
        if not self.enabled:
            return
        self._writer(name).writerow(row)
        self._files[name].flush()

    def _write_json(self, name: str, payload: dict):
        with open(self.dir / name, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, default=_json_default)

    def set_config(self, **kwargs):
        """Late-bound configuration values (resolved after partition setup)."""
        if not self.enabled:
            return
        self._config.update(kwargs)
        self._write_json('config.json', self._config)

    # -------------------------------------------------------------- windows
    def register_client(self, client_id: str):
        if not self.enabled:
            return
        self._preds.setdefault(client_id, {'x_history': [], 'y_true': [],
                                           'y_pred': [], 'window_idx': []})
        self._cum.setdefault(client_id, [0.0, 0.0, 0.0])   # sse, sae, n
        self._update_count.setdefault(client_id, 0)
        self._grad_steps.setdefault(client_id, 0)

    def window(self, *, client_id: str, window_idx: int, y_true, y_pred,
               x_history=None, update_count: Optional[int] = None,
               gradient_steps: Optional[int] = None,
               label_reveal_idx: int = -1, target_start_idx: int = -1,
               target_end_idx: int = -1, regime_id: str = ''):
        """One evaluated window: raw SSE/SAE plus the stored prediction.

        ``update_count``/``gradient_steps`` default to the recorder's own running
        totals, which at call time are exactly the counts accumulated *before*
        this window's prediction.
        """
        if not self.enabled:
            return
        self.register_client(client_id)
        if update_count is None:
            update_count = self._update_count[client_id]
        if gradient_steps is None:
            gradient_steps = self._grad_steps[client_id]
        yt = np.asarray(y_true, dtype=np.float64)
        yp = np.asarray(y_pred, dtype=np.float64)
        n = float(yt.size)
        sse = float(np.sum((yt - yp) ** 2))
        sae = float(np.sum(np.abs(yt - yp)))
        cum = self._cum[client_id]
        cum[0] += sse
        cum[1] += sae
        cum[2] += n
        self._append('per_window_metrics', {
            'run_id': self.run_id, 'method': self.method, 'dataset': self.dataset,
            'seed': self.seed, 'client_id': client_id, 'window_idx': int(window_idx),
            'sse': sse, 'sae': sae, 'num_targets': int(n),
            'window_mse': sse / n if n else float('nan'),
            'window_mae': sae / n if n else float('nan'),
            'cumulative_mse': cum[0] / cum[2] if cum[2] else float('nan'),
            'cumulative_mae': cum[1] / cum[2] if cum[2] else float('nan'),
            'update_count': int(update_count),
            'gradient_steps': int(gradient_steps),

            'label_reveal_idx': int(label_reveal_idx),
            'target_start_idx': int(target_start_idx),
            'target_end_idx': int(target_end_idx),
            'regime_id': regime_id,
        })
        store = self._preds[client_id]
        store['y_true'].append(yt)
        store['y_pred'].append(yp)
        store['window_idx'].append(int(window_idx))
        if x_history is not None:
            store['x_history'].append(np.asarray(x_history, dtype=np.float64))
        self._totals['evaluated_windows'] += 1

    # --------------------------------------------------------------- events
    def online_update(self, *, client_id: str, window_idx: int, gradient_steps: int,
                      learning_rate: float, batch_size: int, update_loss=None,
                      duration_ms=None, optimizer: str = 'SGD',
                      update_type: str = 'window_level',
                      target_start_idx: int = -1, target_end_idx: int = -1,
                      label_available_idx: int = -1, update_time_idx: int = -1):
        if not self.enabled:
            return
        self.register_client(client_id)
        self._update_count[client_id] += 1
        self._grad_steps[client_id] += int(gradient_steps)
        self._append('online_updates', {
            'run_id': self.run_id, 'client_id': client_id,
            'window_idx': int(window_idx), 'update_type': update_type,
            'optimizer': optimizer, 'learning_rate': float(learning_rate),
            'batch_size': int(batch_size), 'gradient_steps': int(gradient_steps),
            'total_gradient_steps': self._grad_steps[client_id],
            'update_loss': '' if update_loss is None else float(update_loss),
            'update_duration_ms': '' if duration_ms is None else float(duration_ms),
            'target_start_idx': int(target_start_idx),
            'target_end_idx': int(target_end_idx),
            'label_available_idx': int(label_available_idx),
            'update_time_idx': int(update_time_idx),
        })
        self._totals['online_update_events'] += 1
        self._totals['gradient_steps'] += int(gradient_steps)
        if duration_ms:
            self._totals['adaptation_time'] += float(duration_ms) / 1000.0

    def candidate_evaluation(self, *, client_id: str, window_idx: int,
                             candidate_expert_id: str, source_client_id: str = '',
                             candidate_version: str = '', predicted_gain=None,
                             estimated_uncertainty=None, shadow_mse=None,
                             evaluation_cost=None, duration_ms=None,
                             selected: bool = False, transferred: bool = False):
        if not self.enabled:
            return None
        self._eval_counter += 1
        eval_id = f'{self.run_id}-e{self._eval_counter:07d}'
        self._append('candidate_evaluations', {
            'eval_id': eval_id, 'run_id': self.run_id, 'client_id': client_id,
            'window_idx': int(window_idx),
            'candidate_expert_id': candidate_expert_id,
            'source_client_id': source_client_id,
            'candidate_version': candidate_version,
            'predicted_gain': '' if predicted_gain is None else float(predicted_gain),
            'estimated_uncertainty': '' if estimated_uncertainty is None else float(estimated_uncertainty),
            'shadow_mse': '' if shadow_mse is None else float(shadow_mse),
            'evaluation_cost': '' if evaluation_cost is None else float(evaluation_cost),
            'evaluation_duration_ms': '' if duration_ms is None else float(duration_ms),
            'selected': int(bool(selected)), 'transferred': int(bool(transferred)),
        })
        self._totals['candidate_evaluations'] += 1
        if duration_ms:
            self._totals['shadow_evaluation_time'] += float(duration_ms) / 1000.0
        self._last_eval_id.setdefault(client_id, []).append({
            'eval_id': eval_id, 'expert_id': candidate_expert_id,
            'window_idx': int(window_idx),
            'predicted_gain': predicted_gain,
        })
        return eval_id

    def feedback(self, *, client_id: str, expert_id: str, realized_gain: float,
                 reward=None, evaluation_window: int = -1,
                 feedback_release_window: int = -1, reward_basis: str = ''):
        """Attach the realised gain to the evaluation that produced it."""
        if not self.enabled:
            return
        queued = self._last_eval_id.get(client_id) or []
        match = None
        for item in queued:
            if item['expert_id'] == expert_id:
                match = item
                break
        eval_id = match['eval_id'] if match else f'{self.run_id}-unmatched'
        predicted = match['predicted_gain'] if match else None
        if match is not None and 'window_idx' in match:
            # The window the evaluation was made in.  With a label delay this is
            # not the window whose label just arrived, so it must come from the
            # queued evaluation, never from the caller.
            evaluation_window = int(match['window_idx'])
        self._append('feedback_rewards', {
            'eval_id': eval_id, 'run_id': self.run_id, 'client_id': client_id,
            'candidate_expert_id': expert_id,
            'evaluation_window': int(evaluation_window),
            'feedback_release_window': int(feedback_release_window),
            'predicted_gain': '' if predicted is None else float(predicted),
            'realized_gain': float(realized_gain),
            'reward': '' if reward is None else float(reward),
            'reward_basis': reward_basis,
        })
        if match is not None:
            queued.remove(match)

    def transfer(self, *, client_id: str, window_idx: int, event_type: str,
                 expert_id: str = '', old_expert_id: str = '',
                 new_expert_id: str = '', source_client_id: str = '',
                 estimated_gain=None, realized_gain=None, transfer_bytes=None,
                 accepted: bool = True, decision_reason: str = ''):
        if not self.enabled:
            return
        if not source_client_id and expert_id:
            source_client_id = self._expert_source.get(expert_id, '')
        self._event_counter += 1
        self._append('transfer_events', {
            'event_id': f'{self.run_id}-t{self._event_counter:07d}',
            'run_id': self.run_id, 'client_id': client_id,
            'window_idx': int(window_idx), 'event_type': event_type,
            'source_client_id': source_client_id, 'expert_id': expert_id,
            'old_expert_id': old_expert_id, 'new_expert_id': new_expert_id,
            'estimated_gain': '' if estimated_gain is None else float(estimated_gain),
            'realized_gain': '' if realized_gain is None else float(realized_gain),
            'transfer_bytes': '' if transfer_bytes is None else int(transfer_bytes),
            'accepted': int(bool(accepted)), 'decision_reason': decision_reason,
        })
        if event_type in ('switch', 'install', 'evict'):
            self._totals['transfer_events'] += 1

    def communication(self, *, client_id: str, window_idx: int, direction: str,
                       payload_type: str, n_bytes: int, round_idx: int = 0,
                       expert_id: str = '', source_client_id: str = '',
                       duration_ms=None):
        if not self.enabled:
            return
        self._comm_counter += 1
        sent = int(n_bytes) if direction == 'upload' else 0
        received = int(n_bytes) if direction == 'download' else 0
        if expert_id and source_client_id:
            # Index for later joins: a transfer row can name the expert but not
            # always the client it came from, and this map is the recorded link.
            self._expert_source[expert_id] = source_client_id
        self._append('communication_events', {
            'event_id': f'{self.run_id}-c{self._comm_counter:07d}',
            'run_id': self.run_id, 'client_id': client_id,
            'window_idx': int(window_idx), 'round_idx': int(round_idx),
            'direction': direction, 'payload_type': payload_type,
            'expert_id': expert_id, 'source_client_id': source_client_id,
            'bytes_sent': sent,
            'bytes_received': received,
            'duration_ms': '' if duration_ms is None else float(duration_ms),
        })
        self._totals['upload_bytes'] += sent
        self._totals['download_bytes'] += received

    def cost(self, *, client_id: str, window_idx: int, operation: str,
             duration_ms: float, estimated_flops=None, peak_memory_mb=None):
        if not self.enabled:
            return
        self._append('cost_events', {
            'run_id': self.run_id, 'client_id': client_id,
            'window_idx': int(window_idx), 'operation': operation,
            'duration_ms': float(duration_ms),
            'estimated_flops': '' if estimated_flops is None else int(estimated_flops),
            'peak_memory_mb': '' if peak_memory_mb is None else float(peak_memory_mb),
        })
        if operation == 'forecasting':
            self._totals['forecasting_time'] += float(duration_ms) / 1000.0
        elif operation == 'shadow_eval':
            self._totals['shadow_evaluation_time'] += float(duration_ms) / 1000.0
        elif operation == 'adaptation':
            self._totals['adaptation_time'] += float(duration_ms) / 1000.0
        elif operation == 'transfer':
            self._totals['transfer_time'] += float(duration_ms) / 1000.0
        elif operation == 'encoder_sync':
            self._totals['encoder_sync_time'] += float(duration_ms) / 1000.0

    def regime(self, *, client_id: str, window_idx: int, event_type: str,
               detected_regime_id=None, previous_regime_id=None,
               detection_score=None, detection_threshold=None,
               related_expert_id: str = ''):
        if not self.enabled:
            return
        self._append('regime_events', {
            'run_id': self.run_id, 'client_id': client_id,
            'window_idx': int(window_idx), 'event_type': event_type,
            'detected_regime_id': '' if detected_regime_id is None else str(detected_regime_id),
            'previous_regime_id': '' if previous_regime_id is None else str(previous_regime_id),
            'detection_score': '' if detection_score is None else float(detection_score),
            'detection_threshold': '' if detection_threshold is None else float(detection_threshold),
            'related_expert_id': related_expert_id,
        })

    # ---------------------------------------------------------------- finish
    def finish(self, summary: Optional[dict] = None, status: str = 'completed'):
        if not self.enabled:
            return {}
        for client_id, store in self._preds.items():
            if not store['y_true']:
                continue
            payload = {
                'y_true': np.stack(store['y_true']),
                'y_pred': np.stack(store['y_pred']),
                'window_idx': np.asarray(store['window_idx'], dtype=np.int64),
            }
            if store['x_history']:
                payload['x_history'] = np.stack(store['x_history'])
            # ``client_id`` is a raw column name, and Weather's columns contain a
            # solidus ("sh (g/kg)"), which opened a *directory* inside
            # ``predictions/`` and killed the run with FileNotFoundError after the
            # whole training + online evaluation had already been paid for.
            safe_id = _FILENAME_UNSAFE.sub('_', str(client_id)).strip('_') or 'client'
            np.savez_compressed(self.dir / 'predictions' / f'client_{safe_id}.npz',
                                **payload)
        record = dict(self._totals)
        record['run_id'] = self.run_id
        record['dataset'] = self.dataset
        record['method'] = self.method
        record['seed'] = self.seed
        record['total_communication_bytes'] = (record['upload_bytes']
                                               + record['download_bytes'])
        record['total_runtime'] = time.time() - self.started
        record['experiment_status'] = status
        if summary:
            record.update(summary)
        self._config['end_time'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        self._config['experiment_status'] = status
        self._config['totals'] = record
        self._write_json('config.json', self._config)
        self._write_json('run_summary.json', record)
        # Every run ships the same file set.  A method without a transfer
        # mechanism produces header-only tables rather than missing ones, so a
        # downstream join never has to special-case the schema.
        for name in SCHEMAS:
            if name not in self._writers and not self._path(name).exists():
                self._writer(name)
        for handle in self._files.values():
            try:
                handle.close()
            except OSError:
                pass
        self._files.clear()
        self._writers.clear()
        return record


def sys_version() -> str:
    import sys
    return sys.version.split()[0]


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
