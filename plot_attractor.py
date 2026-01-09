import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import pybullet as p
from evaluate import BCPolicyWrapper, KukaButtonEnv


def run_and_collect(model_path, episodes=50, device='cpu', history_len=15, max_steps=1000):
    policy = BCPolicyWrapper(model_path, device=device, history_len=history_len)
    env = KukaButtonEnv(render=False, dt=1./60., max_steps=max_steps)

    traces = []  # list of dicts: {'distances': list, 'contacts': list, 'success': bool, 'steps': int, 'final_distance': float}

    for ep in range(episodes):
        observation = env.reset()
        if hasattr(policy, 'reset'):
            policy.reset()

        traj = []
        success = False
        final_dist = None

        for step in range(env.max_steps):
            action = policy.get_action(observation)
            observation, reward, done, info = env.step(action)

            # detect contact between end-effector and button
            contact_flag = False
            contacts = p.getContactPoints(bodyA=env.robot_id, bodyB=env.button_id)
            for c in contacts:
                if c[3] == env.end_effector_link_id:
                    contact_flag = True
                    break

            traj.append({'distance': info['distance_to_button'], 'contact': contact_flag})

            if done:
                success = info['success']
                final_dist = info['distance_to_button']
                steps = step + 1
                break

        if final_dist is None:
            final_dist = traj[-1]['distance'] if len(traj) > 0 else None
            steps = env.max_steps

        distances = [t['distance'] for t in traj]
        contacts = [t['contact'] for t in traj]
        traces.append({'distances': distances, 'contacts': contacts, 'success': success, 'steps': steps, 'final_distance': final_dist})
        print(f"Episode {ep+1}/{episodes}: success={success}, steps={steps}, final_dist={final_dist:.4f}")

    env.close()

    # save traces to JSON for later analysis
    with open('traces.json', 'w') as f:
        json.dump(traces, f, indent=2)

    return traces


def plot_traces(traces, out_path='attractor_plot.png'):
    # Align traces by time step, pad with nan
    max_len = max((len(t['distances']) for t in traces), default=0)
    n = len(traces)
    data = np.full((n, max_len), np.nan)
    success_idx = []
    fail_idx = []
    for i, t in enumerate(traces):
        L = len(t['distances'])
        if L > 0:
            data[i, :L] = t['distances']
        if t['success']:
            success_idx.append(i)
        else:
            fail_idx.append(i)

    plt.figure(figsize=(10, 6))

    # Plot failures in light gray
    for i in fail_idx:
        plt.plot(data[i], color='gray', alpha=0.35, linewidth=0.8)

    # Plot successes in green
    for i in success_idx:
        plt.plot(data[i], color='green', alpha=0.7, linewidth=1.2)

    # Median/quantile for successes
    if len(success_idx) > 0:
        succ_data = data[success_idx]
        median = np.nanmedian(succ_data, axis=0)
        p25 = np.nanpercentile(succ_data, 25, axis=0)
        p75 = np.nanpercentile(succ_data, 75, axis=0)
        plt.plot(median, color='blue', linewidth=2.2, label='Median (success)')
        plt.fill_between(np.arange(max_len), p25, p75, color='blue', alpha=0.15, label='25-75% (success)')

    # Median/quantile for failures (optional)
    if len(fail_idx) > 0:
        fail_data = data[fail_idx]
        f_median = np.nanmedian(fail_data, axis=0)
        plt.plot(f_median, color='black', linestyle='--', linewidth=1.6, label='Median (failure)')

    # Attractor threshold line (example)
    #plt.axhline(0.03, color='red', linestyle=':', linewidth=1.2, label='Attractor threshold (0.03 m)')

    plt.xlabel('Time step')
    plt.ylabel('Distance to button (m)')
    plt.title('Distance-to-Button over Time — Attractor Behavior')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot distance-over-time traces to diagnose attractor failure')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes to run')
    parser.add_argument('--out', type=str, default='attractor_plot.png', help='Output figure path')
    parser.add_argument('--device', type=str, default='cpu', help='Device for model')
    parser.add_argument('--history_len', type=int, default=15, help='History length for LSTM wrapper')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    args = parser.parse_args()

    traces = run_and_collect(args.model, episodes=args.episodes, device=args.device, history_len=args.history_len, max_steps=args.max_steps)
    plot_traces(traces, out_path=args.out)
