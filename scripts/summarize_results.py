#!/usr/bin/env python3
"""Summarize one or more evaluate.py outputs as a Markdown table: completions per category and tracking errors.

    python scripts/summarize_results.py ours=outputs/eval/test [other=path/to/other/eval]

Errors are means over clips that every listed policy completed, so columns stay comparable: world root
position error (raw world frame, includes drift) and aligned body position error (heading and XY aligned,
i.e. pose accuracy). Each clip is one nominal attempt from its first frame, stopped at the first failure.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def load(entry: str) -> tuple[str, dict]:
    label, separator, folder = entry.partition('=')
    if not separator:
        label, folder = Path(entry).name, entry
    results = json.loads((Path(folder) / 'results.json').read_text())
    return label, {row['clip_id']: row for row in results['rows']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('evaluations', nargs='+', metavar='LABEL=DIR')
    args = parser.parse_args()
    runs = dict(load(entry) for entry in args.evaluations)
    clips = set.intersection(*(set(rows) for rows in runs.values()))
    first = next(iter(runs.values()))
    categories = Counter(first[c]['category'] for c in clips)
    labels = list(runs)
    print('| Category | Clips | ' + ' | '.join(labels) + ' |')
    print('|---|---:|' + '---:|' * len(labels))
    for category in sorted(categories):
        done = [sum(rows[c]['complete'] for c in clips if rows[c]['category'] == category) for rows in runs.values()]
        print(f'| {category} | {categories[category]} | ' + ' | '.join(map(str, done)) + ' |')
    totals = [sum(rows[c]['complete'] for c in clips) for rows in runs.values()]
    print(f'| **Total** | **{len(clips)}** | ' + ' | '.join(f'**{t}**' for t in totals) + ' |')
    common = [c for c in clips if all(rows[c]['complete'] for rows in runs.values())]
    print(f'\nOn the {len(common)} clips every policy completed (mean over clips of the per-clip mean):\n')
    print('| Metric | ' + ' | '.join(labels) + ' |')
    print('|---|' + '---:|' * len(labels))
    for key, name in (('world_root_position_m', 'World root position error (cm)'),
                      ('aligned_body_position_m', 'Aligned body position error (cm)')):
        values = [100 * sum(rows[c]['metrics_on_observed_prefix'][key]['mean'] for c in common) / max(len(common), 1)
                  for rows in runs.values()]
        print(f'| {name} | ' + ' | '.join(f'{v:.1f}' for v in values) + ' |')


if __name__ == '__main__':
    main()
