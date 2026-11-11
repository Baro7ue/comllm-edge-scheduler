"""New matched-initialization diagnostic; not a bitwise replay of the archive.

Reuses frozen production step code. Outputs diagnostics only; never saves adapters,
changes test files, or writes an accepted training-run folder.
"""
from pathlib import Path
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import math
import random
import time

SOURCE_SHA = '31a7a05b849a623931b315ed75d41c822da7638d79c948d41c7603024be57225'
ACCEPT_SHA = '71f4ee5603d63ad6a377f5dd568c1f6f355c8f20510d7d2b735b9dea5b4ced94'
DRIVE = Path('/content/drive/MyDrive/COMLLM')
WORK = Path('/content/comllm_s7_fixed')
ROOT = DRIVE / 's7_collapse_pair_v5_20261108'
OLD = DRIVE / 's7_fixed_runs_20261108'
MAX_UPDATES = 20  # Earlier failed2+2, plus20 here =24 for the investigation.
MAX_SECONDS = 480


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def build_prefixes(train, batches):
    """Replace only12 offload-optimal rows in original28; keep other slots."""
    original = [row for batch in batches[:7] for row in batch]
    assert len(original) == 28 and sum(r['best'] == 0 for r in original) == 2
    rng = random.Random(20261108)
    seen = {r['_idx'] for r in original}
    replacements = [r for r in train if r['best'] == 0 and r['_idx'] not in seen]
    rng.shuffle(replacements)
    balanced = list(original)
    replacement_idx = 0
    for start in range(0, 28, 4):
        positions = [i for i in range(start, start + 4) if original[i]['best'] != 0]
        rng.shuffle(positions)
        needed = 2 - sum(original[i]['best'] == 0 for i in range(start, start + 4))
        for i in positions[:needed]:
            balanced[i] = replacements[replacement_idx]
            replacement_idx += 1
    assert replacement_idx == 12
    assert sum(r['best'] == 0 for r in balanced) == 14
    excluded = {r['_idx'] for r in original + balanced}
    probe = []
    for local in (True, False):
        pool = [r for r in train if (r['best'] == 0) == local and r['_idx'] not in excluded]
        rng.shuffle(pool)
        probe.extend(pool[:8])
    assert len(probe) == 16 and not excluded & {r['_idx'] for r in probe}
    return original, balanced, probe


def token_probe(engine, rows, deadline):
    """Canonical first-token mass only, not all parseable-local answer probability."""
    from transformers.generation.logits_process import TopPLogitsWarper
    torch = engine.torch
    cpu = torch.get_rng_state().clone()
    cuda = [x.clone() for x in torch.cuda.get_rng_state_all()]
    was_training = engine.model.training
    local_ids = engine.tok('本地执行', add_special_tokens=False)['input_ids']
    edge_ids = engine.tok('卸载到边缘服务器1', add_special_tokens=False)['input_ids']
    assert local_ids[0] != edge_ids[0]
    result = []
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            engine.model.eval()
            for start in range(0, len(rows), 4):
                assert time.monotonic() < deadline, 'probe deadline reached'
                batch = rows[start:start + 4]
                p, pm = engine.padded(batch)
                pos = (pm.cumsum(1) - 1).clamp(min=0)
                logits = engine.model(input_ids=p, attention_mask=pm, position_ids=pos,
                                      logits_to_keep=1, use_cache=False).logits[:, -1].float()
                scaled = logits / engine.cfg.temperature
                probs = scaled.softmax(-1)
                warped = TopPLogitsWarper(engine.cfg.top_p)(p, scaled.clone())
                sampled = warped.softmax(-1)
                for i, row in enumerate(batch):
                    top = torch.topk(probs[i], 5)
                    result.append({'idx': row['_idx'], 'best': row['best'],
                        'local_token_id': local_ids[0], 'edge_token_id': edge_ids[0],
                        'raw_local_mass': float(probs[i, local_ids[0]]),
                        'nucleus_local_mass': float(sampled[i, local_ids[0]]),
                        'local_cut': bool(torch.isneginf(warped[i, local_ids[0]])),
                        'raw_edge_mass': float(probs[i, edge_ids[0]]),
                        'entropy': float(-(probs[i] * probs[i].clamp_min(1e-30).log()).sum()),
                        'top5_ids': top.indices.tolist(), 'top5_mass': top.values.tolist()})
    finally:
        engine.model.train(was_training)
    assert torch.equal(cpu, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all()))
    return {'rows': result, 'rng_restored': True,
            'canonical_local_tokens': local_ids, 'canonical_edge_tokens': edge_ids,
            'scope': 'first token of canonical 本地执行; not total parseable-local probability'}


def main():
    started = time.monotonic()
    from train import s7_train_fixed as S
    assert sha(S.__file__) == SOURCE_SHA
    accept_path = OLD / 'fixed_acceptance_20261108T134519.138956Z.json'
    assert sha(accept_path) == ACCEPT_SHA
    accepted = json.loads(accept_path.read_text())
    assert accepted['status'] == 'passed'
    assert sha(OLD / 'fixed_preflight.json') == accepted['source_files']['fixed_preflight.json']['sha256']
    from train import s7_reward, grpo_reward
    actual_sources = {Path(module.__file__).name: sha(module.__file__)
                      for module in (S, s7_reward, grpo_reward)}
    assert actual_sources == accepted['identity']['implementation_sha256']
    cfg = S.TrainConfig(base='/content/s7_sft7b', s3_data=str(WORK / 'data/s3'),
                       s4_data=str(WORK / 'data/s4'), out=str(ROOT), group='nolacs', mode='full')
    cfg.validate()
    data = S.load_data(cfg)
    assert data['audit'] == accepted['identity']['data']
    identity = S.model_identity(cfg.base)
    assert identity['files'] == accepted['identity']['base_model']['files']
    assert identity['config'] == accepted['identity']['base_model']['config']
    batches = S.training_batches(data['train'], cfg)
    assert len(batches) == 208
    original, balanced, panel = build_prefixes(data['train'], batches)
    full_log = [json.loads(x) for x in (OLD / 'nolacs/train_log.jsonl').read_text().splitlines()]
    assert sha(OLD / 'nolacs/train_log.jsonl') == accepted['source_files']['nolacs/train_log.jsonl']['sha256']
    ROOT.mkdir(exist_ok=False)
    save(ROOT / 'preflight.json', {'source_sha256': SOURCE_SHA, 'script_sha256': sha(__file__),
        'actual_implementation_sha256': actual_sources,
        'accepted_report_sha256': ACCEPT_SHA, 'data_audit': data['audit'],
        'max_total_updates': MAX_UPDATES, 'max_seconds': MAX_SECONDS,
        'previous_failed_updates': 4, 'whole_investigation_update_ceiling': 24,
        'design': 'new matched-initialization comparison; archive exact samples are descriptive only',
        'arm_updates': {'baseline': 7, 'full_support': 6, 'balanced_prefix': 7},
        'fair_common_comparison_update': 6,
        'original_prefix': [r['_idx'] for r in original],
        'balanced_prefix': [r['_idx'] for r in balanced], 'probe_ids': [r['_idx'] for r in panel],
        'probe_training_overlap': 0, 'original_prefix_local_best': 2,
        'balanced_prefix_local_best': 14,
        'original_test_files_unchanged': True,
        'interventions': {'baseline': 'none', 'full_support': 'top_p .95 to 1 only',
                         'balanced_prefix': 'replace12 offload-optimal rows with local-optimal rows; slots otherwise retained'},
        'measurement_timing': 'all panel forwards after6 updates; baseline/balanced also after7; SFT reference measured adapter-disabled after6'})
    total_updates = 0
    summaries = []
    for arm in ('baseline', 'full_support', 'balanced_prefix'):
        assert time.monotonic() - started < MAX_SECONDS
        arm_cfg = dataclasses.replace(cfg)
        if arm == 'full_support':
            arm_cfg.top_p = 1.0  # Explicit diagnostic override; frozen runner itself is untouched.
        prefix = balanced if arm == 'balanced_prefix' else original
        arm_updates = 6 if arm == 'full_support' else 7
        folder = ROOT / arm
        folder.mkdir()
        save(folder / 'config.json', dataclasses.asdict(arm_cfg))
        random.seed(arm_cfg.seed)
        import numpy as np
        np.random.seed(arm_cfg.seed)
        engine = S.TorchEngine(arm_cfg, data['instruction'])
        original_generate = None
        try:
            runtime = {'versions': {k: importlib.metadata.version(k) for k in ('torch','transformers','peft','accelerate','numpy')},
                       'gpu': engine.versions['gpu']}
            expected_env = json.loads((OLD / 'fixed_preflight.json').read_text())['environment']
            assert runtime == {key: expected_env[key] for key in ('versions', 'gpu')}
            rows, dropped = engine.prepare(data['train'])
            assert not dropped and len(rows) == 830
            val, val_dropped = engine.prepare(data['val'])
            assert not val_dropped and len(val) == 100
            h = hashlib.sha256()
            for name, param in sorted(engine.model.named_parameters()):
                if 'lora_A' in name:
                    h.update(name.encode())
                    h.update(param.detach().contiguous().cpu().view(engine.torch.uint8).numpy().tobytes())
            fingerprint = {'lora_A_sha256': h.hexdigest(),
                'torch_num_threads': engine.torch.get_num_threads(),
                'torch_num_interop_threads': engine.torch.get_num_interop_threads(),
                'default_dtype': str(engine.torch.get_default_dtype()),
                'attention_implementation': engine.model.config._attn_implementation}
            save(folder / 'initial_fingerprint.json', fingerprint)
            if summaries:
                assert fingerprint == summaries[0]['initial_fingerprint'], 'fresh adapter initialization differs between arms'
            before = engine.policy_consistency_probe(rows[:2], require_nonzero_b=False)
            assert before['passed'] and before['initial_lora_B_all_zero']
            h = hashlib.sha256(engine.torch.get_rng_state().numpy().tobytes())
            for state in engine.torch.cuda.get_rng_state_all():
                h.update(state.numpy().tobytes())
            rng_fingerprint = h.hexdigest()
            if summaries:
                assert rng_fingerprint == summaries[0]['initial_rng_sha256'], 'initial RNG differs between arms'
            save(folder / 'initial_probe.json', before)
            generated = []
            original_generate = engine.model.generate

            def observed_generate(*args, **kwargs):
                output = original_generate(*args, **kwargs)
                width = kwargs['input_ids'].shape[1]
                generated.clear()
                for tokens in output[:, width:].detach().cpu().tolist():
                    n = next((i + 1 for i, t in enumerate(tokens) if t in engine.stops), len(tokens))
                    generated.append(tokens[:n])
                return output

            engine.model.generate = observed_generate
            with (folder / 'probes.jsonl').open('x') as probe_stream, (folder / 'steps.jsonl').open('x') as step_stream:
                exact = True
                for update in range(arm_updates):
                    assert total_updates < MAX_UPDATES and time.monotonic() - started < MAX_SECONDS
                    batch = prefix[update * 4:(update + 1) * 4]
                    info = engine.step(batch, 0.0, update, 208)
                    total_updates += 1
                    assert len(generated) == 16
                    info.update({'update': update + 1, 'generated_token_ids': list(generated),
                                 'completion_lengths_with_eos': [len(t) for t in generated],
                                 'GT_local_count': sum(r['best'] == 0 for r in batch)})
                    if arm == 'baseline':
                        old = full_log[update]
                        same = (info['prompt_indices'] == old['prompt_indices'] and
                                info['sampled_texts'] == old['sampled_texts'] and
                                info['sampled_rewards'] == old['sampled_rewards'])
                        exact = exact and same
                        info['original_samples_rewards_exact_match'] = same
                    step_stream.write(json.dumps(info, ensure_ascii=False, allow_nan=False) + '\n'); step_stream.flush()
                    print(json.dumps({'arm': arm, 'update': update + 1,
                        'local_samples': info['sampled_local_count'], 'GT_local': info['GT_local_count'],
                        'archived_exact_so_far': exact if arm == 'baseline' else None}), flush=True)
                    if update + 1 >= 6:
                        if update + 1 == 6:
                            with engine.model.disable_adapter():
                                probe = token_probe(engine, panel, started + MAX_SECONDS)
                                probe_stream.write(json.dumps({'after_update': 0, 'measurement': 'frozen_SFT_adapter_disabled_after6', **probe}, ensure_ascii=False) + '\n'); probe_stream.flush()
                        probe = token_probe(engine, panel, started + MAX_SECONDS)
                        probe_stream.write(json.dumps({'after_update': update + 1, 'measurement': 'trained_policy', **probe}, ensure_ascii=False) + '\n'); probe_stream.flush()
                        greedy = []
                        for start in range(0, 16, 4):
                            assert time.monotonic() - started < MAX_SECONDS, 'greedy deadline reached'
                            texts = engine.answers(panel[start:start + 4])
                            for row, text in zip(panel[start:start + 4], texts):
                                greedy.append({'idx': row['_idx'], 'best': row['best'], 'text': text,
                                               'pred': S.parse_prediction(text, row['servers'])})
                        save(folder / ('panel_greedy_after' + str(update + 1) + '.json'), greedy)
            summary = {'arm': arm, 'updates': arm_updates, 'status': 'complete', 'runtime': runtime,
                       'initial_fingerprint': fingerprint,
                       'initial_rng_sha256': rng_fingerprint,
                       'archived_candidate_reward_exact': exact if arm == 'baseline' else None,
                       'archive_exactness_required': False,
                       'first_token_probe_scope_only': True, 'new_full_test_run': False}
            save(folder / 'summary.json', summary)
            summaries.append(summary)
        finally:
            if original_generate is not None:
                engine.model.generate = original_generate
                del original_generate
            engine.close()
            del engine
            gc.collect()
    assert total_updates == 20
    save(ROOT / 'summary.json', {'status': 'complete', 'total_updates': total_updates,
        'invocation_seconds': time.monotonic() - started, 'arms': summaries,
        'limitations': ['new matched-initialization comparison, not exact archived replay',
                        'fair 6-update comparison across3 arms; baseline/balanced also measured after7',
                        'one seed only; no claim of full-run generalization',
                        'balanced-prefix intervention changes12 training states, not a reward-code change',
                        'probability probe covers canonical local first token only']})
    print('COLLAPSE_PROBE_COMPLETE', str(ROOT), flush=True)


if __name__ == '__main__':
    main()
