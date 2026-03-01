import argparse
import os
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:
    raise SystemExit(
        'Missing tensorboard. Install with: pip install tensorboard\n' + str(e)
    )

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_scalars(event_path):
    acc = EventAccumulator(str(event_path))
    acc.Reload()
    tags = acc.Tags().get('scalars', [])
    scalars = {tag: acc.Scalars(tag) for tag in tags}
    return scalars


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--event', required=True, help='Path to events.out.tfevents.* file')
    p.add_argument('--tag', default='train/loss', help='Scalar tag to plot (default: train/loss)')
    p.add_argument('--out', required=True, help='Output PNG path')
    p.add_argument('--csv', required=False, help='Optional CSV output path')
    args = p.parse_args()

    event_path = Path(args.event)
    if not event_path.exists():
        raise SystemExit(f'Event file not found: {event_path}')

    scalars = load_scalars(event_path)
    if args.tag not in scalars:
        raise SystemExit('Tag not found. Available tags: ' + ', '.join(scalars.keys()))

    data = scalars[args.tag]
    steps = [d.step for d in data]
    values = [d.value for d in data]

    plt.figure(figsize=(7,4), dpi=200)
    plt.plot(steps, values, color='#4e79a7', linewidth=1.8)
    plt.title(f'Loss vs Steps ({args.tag})')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)

    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['step', 'value'])
            w.writerows(zip(steps, values))

    print('OK')


if __name__ == '__main__':
    main()
