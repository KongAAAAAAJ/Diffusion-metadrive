#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

TARGETS = [
    Path('models/bev_planner/joint_grpo.py'),
    Path('train/bev_joint_grpo_online/config.py'),
    Path('train/bev_joint_grpo_online/train.py'),
    Path('train/bev_joint_grpo_online/rollout.py'),
    Path('train/bev_joint_grpo_online/runner.py'),
    Path('configs/train/bev_joint_grpo.yaml'),
]

class PatchError(RuntimeError):
    pass


def _insert_after_field(text: str, field: str, new_line: str, *, guard: str) -> str:
    if guard in text:
        return text
    pattern = re.compile(rf'^(?P<indent>\s*){re.escape(field)}\s*:[^\n]*$', re.M)
    m = pattern.search(text)
    if not m:
        raise PatchError(f'cannot find field: {field}')
    indent = m.group('indent')
    return text[:m.end()] + '\n' + indent + new_line + text[m.end():]


def _insert_after_assignment(text: str, needle_regex: str, insertion: str, *, guard: str, count_expected_at_least: int = 1) -> str:
    if guard in text:
        return text
    pattern = re.compile(needle_regex, re.M)
    matches = list(pattern.finditer(text))
    if len(matches) < count_expected_at_least:
        raise PatchError(f'cannot find assignment anchor: {needle_regex}')
    # only first occurrence by design
    m = matches[0]
    return text[:m.end()] + insertion + text[m.end():]


def _replace_block(text: str, start: str, end: str, transform):
    s = text.find(start)
    if s < 0:
        raise PatchError(f'cannot find block start: {start}')
    e = text.find(end, s)
    if e < 0:
        raise PatchError(f'cannot find block end: {end}')
    block = text[s:e]
    new_block = transform(block)
    return text[:s] + new_block + text[e:]


def edit_joint_grpo(text: str) -> str:
    # 1) Config field.
    text = _insert_after_field(
        text,
        'unsafe_advantage_value',
        'skip_all_unsafe_group: bool = False',
        guard='skip_all_unsafe_group: bool =',
    )

    # 2) Config validation.
    if 'skip_all_unsafe_group must be a bool' not in text:
        anchor = '        if not isinstance(self.unsafe_advantage_override_enabled, bool):\n            raise JointGRPOError("unsafe_advantage_override_enabled must be a bool")'
        if anchor not in text:
            raise PatchError('cannot find JointGRPOConfig unsafe override validation')
        replacement = anchor + '\n        if not isinstance(self.skip_all_unsafe_group, bool):\n            raise JointGRPOError("skip_all_unsafe_group must be a bool")'
        text = text.replace(anchor, replacement, 1)

    def transform_adv(block: str) -> str:
        # Signature.
        if 'skip_all_unsafe_group: bool = False' not in block:
            block, n = re.subn(
                r'(\n\s*unsafe_advantage_value:\s*float\s*=\s*-1\.0,)',
                r'\1\n    skip_all_unsafe_group: bool = False,',
                block,
                count=1,
            )
            if n != 1:
                raise PatchError('cannot add skip_all_unsafe_group to standard_grpo_advantages signature')

        # Validation.
        if 'skip_all_unsafe_group must be a bool' not in block:
            anchor = '    if not isinstance(unsafe_override_enabled, bool):\n        raise JointGRPOError("unsafe_override_enabled must be a bool")'
            if anchor not in block:
                raise PatchError('cannot find standard_grpo_advantages unsafe validation')
            block = block.replace(
                anchor,
                anchor + '\n    if not isinstance(skip_all_unsafe_group, bool):\n        raise JointGRPOError("skip_all_unsafe_group must be a bool")',
                1,
            )

        # Mask validation must run if either override or all-unsafe skipping needs masks.
        # Limit replacement to the first mask-validation if inside this function.
        if 'if unsafe_override_enabled or skip_all_unsafe_group:' not in block:
            old = '    if unsafe_override_enabled:\n        for name, mask in ('
            if old not in block:
                raise PatchError('cannot find unsafe mask validation branch in standard_grpo_advantages')
            block = block.replace(old, '    if unsafe_override_enabled or skip_all_unsafe_group:\n        for name, mask in (', 1)

        # Build unsafe_mask once after validation, before reward centering.
        if 'unsafe_mask = collision_mask | out_of_drivable_mask\n    centered =' not in block:
            anchor = '    centered = current_rewards - current_rewards.mean(dim=-1, keepdim=True)'
            if anchor not in block:
                raise PatchError('cannot find centered reward anchor in standard_grpo_advantages')
            insertion = (
                '    unsafe_mask = None\n'
                '    if unsafe_override_enabled or skip_all_unsafe_group:\n'
                '        unsafe_mask = collision_mask | out_of_drivable_mask\n'
            )
            block = block.replace(anchor, insertion + anchor, 1)

        # Avoid reassigning unsafe_mask in override branch.
        block = block.replace(
            '    if unsafe_override_enabled:\n        unsafe_mask = collision_mask | out_of_drivable_mask\n        advantages = torch.where(',
            '    if unsafe_override_enabled:\n        advantages = torch.where(',
            1,
        )

        # Zero all-unsafe groups before valid-mode masking/signal calculation.
        if 'all_unsafe_group = unsafe_mask.all(dim=-1)' not in block:
            anchor = '    advantages = advantages * valid_executable_mode_mask.unsqueeze(-1).to('
            if anchor not in block:
                raise PatchError('cannot find advantage valid-mask anchor')
            insertion = (
                '    if skip_all_unsafe_group:\n'
                '        all_unsafe_group = unsafe_mask.all(dim=-1)\n'
                '        advantages = torch.where(\n'
                '            all_unsafe_group.unsqueeze(-1),\n'
                '            torch.zeros_like(advantages),\n'
                '            advantages,\n'
                '        )\n'
            )
            block = block.replace(anchor, insertion + anchor, 1)
        return block

    text = _replace_block(text, 'def standard_grpo_advantages(', 'def _active_tensor_mean(', transform_adv)

    # 3) Thread config into both trainer calls.
    call_anchor = '            unsafe_advantage_value=self.config.unsafe_advantage_value,\n            trajectories_per_mode=self.config.trajectories_per_mode,'
    call_repl = '            unsafe_advantage_value=self.config.unsafe_advantage_value,\n            skip_all_unsafe_group=self.config.skip_all_unsafe_group,\n            trajectories_per_mode=self.config.trajectories_per_mode,'
    if 'skip_all_unsafe_group=self.config.skip_all_unsafe_group' not in text:
        n = text.count(call_anchor)
        if n < 2:
            raise PatchError(f'expected >=2 trainer advantage call anchors, found {n}')
        text = text.replace(call_anchor, call_repl)

    # 4) Contract text/version is diagnostic only; update if matching the current contract.
    text = text.replace(
        'stage2_joint_grpo_optimizer_v17_optional_unsafe_advantage',
        'stage2_joint_grpo_optimizer_v18_skip_all_unsafe_group',
    )
    if 'all-unsafe groups can be masked from trajectory PG' not in text:
        text = text.replace(
            'no frozen-reward gate; optional fixed negative unsafe override is configurable"',
            'no frozen-reward gate; optional fixed negative unsafe override is configurable; "\n            "all-unsafe groups can be masked from trajectory PG"',
            1,
        )
    return text


def edit_online_config(text: str) -> str:
    # Dataclass field.
    if 'skip_all_unsafe_group: bool = False' not in text:
        anchor = '    unsafe_advantage_value: float = -1.0'
        if anchor not in text:
            raise PatchError('cannot find JointGRPOAdvantageConfig unsafe_advantage_value')
        text = text.replace(anchor, anchor + '\n    skip_all_unsafe_group: bool = False', 1)
    # Validation.
    if "grpo.advantage.skip_all_unsafe_group must be a bool" not in text:
        anchor = "        if not isinstance(self.unsafe_override_enabled, bool):\n            raise OnlineGRPOError('grpo.advantage.unsafe_override_enabled must be a bool')"
        if anchor not in text:
            raise PatchError('cannot find JointGRPOAdvantageConfig bool validation')
        text = text.replace(
            anchor,
            anchor + "\n        if not isinstance(self.skip_all_unsafe_group, bool):\n            raise OnlineGRPOError('grpo.advantage.skip_all_unsafe_group must be a bool')",
            1,
        )
    return text


def edit_train(text: str) -> str:
    # Parse the new YAML boolean.
    if "skip_all_unsafe_group = advantage_mapping.get('skip_all_unsafe_group'" not in text:
        anchor = "    if not isinstance(unsafe_override_enabled, bool):\n        raise OnlineGRPOError('grpo.advantage.unsafe_override_enabled must be a bool')"
        if anchor not in text:
            raise PatchError('cannot find YAML unsafe_override parsing validation')
        insertion = (
            anchor
            + "\n    skip_all_unsafe_group = advantage_mapping.get('skip_all_unsafe_group', False)"
            + "\n    if not isinstance(skip_all_unsafe_group, bool):"
            + "\n        raise OnlineGRPOError('grpo.advantage.skip_all_unsafe_group must be a bool')"
        )
        text = text.replace(anchor, insertion, 1)
    # Pass into dataclass.
    if 'skip_all_unsafe_group=skip_all_unsafe_group' not in text:
        anchor = "        unsafe_advantage_value=float(advantage_mapping.get('unsafe_advantage_value', -1.0)),"
        if anchor not in text:
            raise PatchError('cannot find JointGRPOAdvantageConfig construction')
        text = text.replace(anchor, anchor + '\n        skip_all_unsafe_group=skip_all_unsafe_group,', 1)
    return text


def edit_rollout(text: str) -> str:
    def transform(block: str) -> str:
        if 'skip_all_unsafe_group: bool = False' not in block:
            block, n = re.subn(
                r'(\n\s*unsafe_advantage_value:\s*float\s*=\s*-1\.0,)',
                r'\1\n    skip_all_unsafe_group: bool = False,',
                block,
                count=1,
            )
            if n != 1:
                raise PatchError('cannot add skip_all_unsafe_group to numpy reward signal signature')
        if 'skip_all_unsafe_group must be a bool' not in block:
            anchor = "    if not isinstance(unsafe_override_enabled, bool):\n        raise OnlineGRPOError('unsafe_override_enabled must be a bool')"
            if anchor not in block:
                raise PatchError('cannot find numpy unsafe override validation')
            block = block.replace(
                anchor,
                anchor + "\n    if not isinstance(skip_all_unsafe_group, bool):\n        raise OnlineGRPOError('skip_all_unsafe_group must be a bool')",
                1,
            )
        if 'if unsafe_override_enabled or skip_all_unsafe_group:' not in block:
            old = '    if unsafe_override_enabled:\n        collision = np.asarray(collision_mask)'
            if old not in block:
                raise PatchError('cannot find numpy unsafe mask branch')
            block = block.replace(
                old,
                '    if unsafe_override_enabled or skip_all_unsafe_group:\n        collision = np.asarray(collision_mask)',
                1,
            )
        # Build unsafe once and use it for both override and group skip.
        if 'unsafe = collision | out\n    centered =' not in block:
            anchor = '    centered = values - values.mean(axis=-1, keepdims=True)'
            if anchor not in block:
                raise PatchError('cannot find numpy centered reward anchor')
            block = block.replace(
                anchor,
                '    unsafe = None\n    if unsafe_override_enabled or skip_all_unsafe_group:\n        unsafe = collision | out\n' + anchor,
                1,
            )
        block = block.replace(
            '    if unsafe_override_enabled:\n        unsafe = collision | out\n        advantages = np.where(',
            '    if unsafe_override_enabled:\n        advantages = np.where(',
            1,
        )
        if 'all_unsafe_group = np.all(unsafe, axis=-1)' not in block:
            anchor = '    advantages *= valid[..., None]'
            if anchor not in block:
                raise PatchError('cannot find numpy valid advantage mask anchor')
            insertion = (
                '    if skip_all_unsafe_group:\n'
                '        all_unsafe_group = np.all(unsafe, axis=-1)\n'
                '        advantages = np.where(\n'
                '            all_unsafe_group[..., None],\n'
                '            np.float32(0.0),\n'
                '            advantages,\n'
                '        ).astype(np.float32, copy=False)\n'
            )
            block = block.replace(anchor, insertion + anchor, 1)
        return block
    return _replace_block(text, 'def _standard_grpo_reward_signals(', 'def _diffusion_attempt_generators(', transform)


def edit_runner(text: str) -> str:
    # The value is threaded into two places: JointGRPOConfig and the NumPy
    # rollout-gating helper. Insert it after every matching unsafe value line
    # that does not already have the skip line immediately after it.
    pattern = re.compile(
        r'^(?P<indent>\s*)unsafe_advantage_value=advantage_config\.unsafe_advantage_value,\n'
        r'(?!\s*skip_all_unsafe_group=advantage_config\.skip_all_unsafe_group,)',
        re.M,
    )
    text, _ = pattern.subn(
        lambda m: (
            m.group(0)
            + m.group('indent')
            + 'skip_all_unsafe_group=advantage_config.skip_all_unsafe_group,\n'
        ),
        text,
    )
    count = text.count('skip_all_unsafe_group=advantage_config.skip_all_unsafe_group')
    if count < 2:
        raise PatchError(
            f'expected skip_all_unsafe_group to be threaded into >=2 runner calls; found {count}'
        )
    return text


def edit_yaml(text: str) -> str:
    if re.search(r'^\s*skip_all_unsafe_group\s*:', text, re.M):
        # Force desired value true without touching unsafe_override_enabled.
        text = re.sub(r'^(\s*skip_all_unsafe_group[ \t]*:)[ \t]*(?:true|false)[ \t]*$', r'\1 true', text, count=1, flags=re.M)
        return text
    pattern = re.compile(r'^(?P<indent>\s*)unsafe_advantage_value\s*:\s*[^\n]+$', re.M)
    m = pattern.search(text)
    if not m:
        raise PatchError('cannot find YAML grpo.advantage.unsafe_advantage_value')
    indent = m.group('indent')
    return text[:m.end()] + '\n' + indent + 'skip_all_unsafe_group: true' + text[m.end():]


EDITORS = {
    TARGETS[0]: edit_joint_grpo,
    TARGETS[1]: edit_online_config,
    TARGETS[2]: edit_train,
    TARGETS[3]: edit_rollout,
    TARGETS[4]: edit_runner,
    TARGETS[5]: edit_yaml,
}


def verify_python(path: Path, text: str) -> None:
    if path.suffix == '.py':
        try:
            compile(text, str(path), 'exec')
        except SyntaxError as exc:
            raise PatchError(f'generated Python syntax error in {path}: {exc}') from exc


def verify_semantics(outputs: dict[Path, str]) -> None:
    j = outputs[TARGETS[0]]
    c = outputs[TARGETS[1]]
    t = outputs[TARGETS[2]]
    ro = outputs[TARGETS[3]]
    ru = outputs[TARGETS[4]]
    y = outputs[TARGETS[5]]
    checks = [
        ('JointGRPOConfig field', 'skip_all_unsafe_group: bool = False' in j),
        ('Torch all-unsafe mask', 'all_unsafe_group = unsafe_mask.all(dim=-1)' in j),
        ('Torch zeroing', 'torch.zeros_like(advantages)' in j),
        ('Torch trainer threading', 'skip_all_unsafe_group=self.config.skip_all_unsafe_group' in j),
        ('Online advantage config', 'skip_all_unsafe_group: bool = False' in c),
        ('YAML parser', 'skip_all_unsafe_group=skip_all_unsafe_group' in t),
        ('NumPy all-unsafe mask', 'all_unsafe_group = np.all(unsafe, axis=-1)' in ro),
        ('Runner torch threading', 'skip_all_unsafe_group=advantage_config.skip_all_unsafe_group' in ru),
        ('Runner numpy threading', 'skip_all_unsafe_group=advantage_config.skip_all_unsafe_group' in ru),
        ('YAML enabled', bool(re.search(r'^\s*skip_all_unsafe_group\s*:\s*true\s*$', y, re.M))),
    ]
    failed = [name for name, ok in checks if not ok]
    if failed:
        raise PatchError('semantic verification failed: ' + ', '.join(failed))


def main() -> int:
    parser = argparse.ArgumentParser(description='Semantically add skip_all_unsafe_group to ChassisFusion GRPO.')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='simulate and validate, write nothing')
    mode.add_argument('--apply', action='store_true', help='apply changes with .bak_skip_all_unsafe backups')
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='repository root (default: cwd)')
    args = parser.parse_args()
    root = args.root.resolve()

    outputs: dict[Path, str] = {}
    originals: dict[Path, str] = {}
    print(f'[skip-all-unsafe] repo root: {root}')
    for rel in TARGETS:
        path = root / rel
        if not path.is_file():
            raise PatchError(f'missing required file: {rel}')
        original = path.read_text(encoding='utf-8')
        originals[rel] = original
        try:
            modified = EDITORS[rel](original)
        except PatchError as exc:
            raise PatchError(f'{rel}: {exc}') from exc
        verify_python(rel, modified)
        outputs[rel] = modified
        state = 'already-compatible' if modified == original else 'will-modify'
        print(f'  {state:18s} {rel}')

    verify_semantics(outputs)

    # Optional YAML parse if PyYAML is available in the environment.
    try:
        import yaml  # type: ignore
        payload = yaml.safe_load(outputs[TARGETS[5]])
        value = payload['grpo']['advantage']['skip_all_unsafe_group']
        if value is not True:
            raise PatchError('YAML skip_all_unsafe_group did not parse as true')
        print('  yaml-parse          OK')
    except ImportError:
        print('  yaml-parse          skipped (PyYAML unavailable)')

    if args.check:
        print('[skip-all-unsafe] CHECK PASSED; no files were written.')
        return 0

    changed = []
    for rel in TARGETS:
        if outputs[rel] == originals[rel]:
            continue
        path = root / rel
        backup = path.with_name(path.name + '.bak_skip_all_unsafe')
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(outputs[rel], encoding='utf-8')
        changed.append(str(rel))
    print(f'[skip-all-unsafe] APPLY PASSED; changed {len(changed)} file(s).')
    for rel in changed:
        print(f'  changed             {rel}')
    print('[skip-all-unsafe] Backups use suffix: .bak_skip_all_unsafe')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except PatchError as exc:
        print(f'[skip-all-unsafe] ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2)
