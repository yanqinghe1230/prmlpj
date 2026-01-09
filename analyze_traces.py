import json
import numpy as np
import argparse
from pathlib import Path


def analyze(traces, D_thresh=0.02, M=10):
    N = len(traces)
    results = []

    for t in traces:
        d = np.array(t['distances'], dtype=np.float32)
        contacts = np.array(t.get('contacts', []), dtype=bool)
        success = bool(t.get('success', False))
        steps = int(t.get('steps', len(d)))
        final_distance = float(t.get('final_distance', d[-1] if len(d)>0 else np.nan))

        # find basin entry: first t0 where d[t0:t0+M].max() < D_thresh
        t0 = None
        for i in range(0, max(0, len(d)-M+1)):
            if np.nanmax(d[i:i+M]) < D_thresh:
                t0 = i
                break

        entered_basin = t0 is not None
        time_to_basin = t0 if entered_basin else None

        min_distance = float(np.nanmin(d)) if len(d)>0 else np.nan
        time_of_min = int(np.nanargmin(d)) if len(d)>0 else None

        contact_time = None
        if len(contacts) > 0:
            idxs = np.where(contacts)[0]
            if len(idxs) > 0:
                contact_time = int(idxs[0])

        results.append({
            'entered_basin': entered_basin,
            'time_to_basin': time_to_basin,
            'min_distance': min_distance,
            'time_of_min': time_of_min,
            'contact_time': contact_time,
            'success': success,
            'steps': steps,
            'final_distance': final_distance
        })

    # aggregate
    entered = [r for r in results if r['entered_basin']]
    not_entered = [r for r in results if not r['entered_basin']]

    summary = {}
    summary['N'] = N
    summary['basin_entry_fraction'] = len(entered)/N if N>0 else 0
    summary['entered_median_time'] = np.median([r['time_to_basin'] for r in entered]) if len(entered)>0 else None
    summary['entered_iqr_time'] = (np.percentile([r['time_to_basin'] for r in entered],25), np.percentile([r['time_to_basin'] for r in entered],75)) if len(entered)>0 else (None,None)
    summary['min_distance_median'] = np.median([r['min_distance'] for r in results])
    summary['min_distance_iqr'] = (np.percentile([r['min_distance'] for r in results],25), np.percentile([r['min_distance'] for r in results],75))

    # contact stats
    contact_times = [r['contact_time'] for r in results if r['contact_time'] is not None]
    summary['contact_fraction'] = len(contact_times)/N if N>0 else 0
    summary['median_contact_time'] = int(np.median(contact_times)) if len(contact_times)>0 else None

    # cross metrics
    succ_and_enter = len([r for r in results if r['success'] and r['entered_basin']])
    enter_but_not_succ = len([r for r in results if r['entered_basin'] and not r['success']])
    succ_but_not_enter = len([r for r in results if r['success'] and not r['entered_basin']])
    summary['succ_and_enter_frac'] = succ_and_enter / N if N>0 else 0
    summary['enter_but_not_succ_frac'] = enter_but_not_succ / N if N>0 else 0
    summary['succ_but_not_enter_frac'] = succ_but_not_enter / N if N>0 else 0

    return results, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traces', type=str, default='traces.json')
    parser.add_argument('--D', type=float, default=0.02, help='Distance threshold for basin')
    parser.add_argument('--M', type=int, default=10, help='Persistence length in steps')
    parser.add_argument('--out', type=str, default='trace_analysis.json')
    args = parser.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        print('traces file not found:', traces_path)
        raise SystemExit(1)

    traces = json.load(open(traces_path))
    results, summary = analyze(traces, D_thresh=args.D, M=args.M)

    print('\nSummary:')
    for k,v in summary.items():
        print(f"  {k}: {v}")

    with open(args.out, 'w') as f:
        json.dump({'summary': summary, 'per_episode': results}, f, indent=2)

    print('\nWrote analysis to', args.out)
