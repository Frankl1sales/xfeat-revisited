import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not os.path.exists(os.path.join(REPO_ROOT, 'modules')):
    raise RuntimeError(f"Could not find 'modules/' under repository root: {REPO_ROOT}")

if os.path.abspath(os.getcwd()) != REPO_ROOT:
    raise RuntimeError(f"Run this module from the repository root: {REPO_ROOT}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots for Mac and Windows.")
    parser.add_argument('--mac_file', type=str, default='benchmark_results_MAC.csv', help='Path to the MacBook benchmark CSV file')
    parser.add_argument('--win_file', type=str, default='benchmark_results_WIN.csv', help='Path to the Windows benchmark CSV file')
    parser.add_argument('--output', type=str, default='benchmark_plot_combined.png', help='Name of the output plot file')
    args = parser.parse_args()

    df_mac = pd.read_csv(args.mac_file)
    df_win = pd.read_csv(args.win_file)

    df_mac['Machine'] = 'MacBook Pro M1 Pro'
    df_win['Machine'] = 'Windows AMD Ryzen 7 5800H'
    df = pd.concat([df_mac, df_win], ignore_index=True)

    method_styles = {
        'XFeat': {'color': 'tab:orange', 'marker': 's', 'size': 100, 'mew': 1.5},
        'XFeat*': {'color': 'tab:orange', 'marker': 'o', 'size': 100, 'mew': 1.5},
        'ALIKE': {'color': 'tab:red', 'marker': 'P', 'size': 100, 'mew': 1.5},
        'DISK': {'color': 'tab:green', 'marker': 'D', 'size': 100, 'mew': 1.5},
        'SuperPoint': {'color': 'tab:blue', 'marker': '^', 'size': 100, 'mew': 1.5},
        'ORB': {'color': 'tab:purple', 'marker': '*', 'size': 150, 'mew': 1.5},
    }

    machine_styles = {
        'MacBook Pro M1 Pro': {'facecolor_modifier': lambda c: c, 'linestyle': '-'},
        'Windows AMD Ryzen 7 5800H': {'facecolor_modifier': lambda c: 'white', 'linestyle': '--'}
    }

    fps_col = 'FPS_VGA' if 'FPS_VGA' in df.columns else 'FPS'
    acc_col = 'PoseAccuracy@10deg' if 'PoseAccuracy@10deg' in df.columns else 'Accuracy@10°'
    std_col = 'FPS_VGA_STD' if 'FPS_VGA_STD' in df.columns else ('FPS_STD' if 'FPS_STD' in df.columns else None)

    plt.figure(figsize=(10, 6))

    for machine in df['Machine'].unique():
        df_m = df[df['Machine'] == machine]
        m_style = machine_styles[machine]

        for method in df_m['Method'].unique():
            row = df_m[df_m['Method'] == method].iloc[0]
            style = method_styles.get(method, {'color': 'gray', 'marker': 'o', 'size': 100, 'mew': 1.5})

            facecolor = m_style['facecolor_modifier'](style['color'])

            if std_col and std_col in df.columns:
                plt.errorbar(row[fps_col], row[acc_col], xerr=row[std_col],
                            fmt=style['marker'], markersize=8, capsize=4,
                            color=style['color'], markerfacecolor=facecolor,
                            markeredgecolor=style['color'], markeredgewidth=style['mew'],
                            ecolor=style['color'], alpha=0.8)
            else:
                plt.scatter(row[fps_col], row[acc_col],
                           marker=style['marker'], s=style['size'],
                           facecolors=facecolor, edgecolors=style['color'],
                           linewidths=style['mew'], zorder=3)

            offset_y = -1.5 if machine == 'MacBook Pro M1 Pro' else 1.5
            va = 'top' if machine == 'MacBook Pro M1 Pro' else 'bottom'

            plt.text(row[fps_col], row[acc_col] + offset_y,
                    method, fontsize=8, va=va, ha='center',
                    color=style['color'], alpha=0.9)

        pareto_methods = df_m[df_m['Method'].str.contains('XFeat|ORB')]
        if len(pareto_methods) > 0:
            pareto_points = pareto_methods[[fps_col, acc_col]].values
            pareto_points = pareto_points[pareto_points[:, 0].argsort()]

            plt.plot(pareto_points[:, 0], pareto_points[:, 1],
                    color='red', alpha=0.3, linestyle=m_style['linestyle'], linewidth=2)

    plt.xlabel("Frames per second @ VGA (640x480)")
    plt.ylabel("Pose accuracy @ 10° (%)")
    plt.ylim(30, 90)
    plt.grid(True, alpha=0.3)

    mac_patch = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                              markersize=8, markerfacecolor='gray', label='MacBook Pro M1 Pro')
    amd_patch = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                              markersize=8, markerfacecolor='white', markeredgecolor='gray', markeredgewidth=1.5, label='Windows AMD Ryzen 7 5800H')

    plt.legend(handles=[mac_patch, amd_patch], loc='lower right', title="Hardware Setup", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {args.output}")


if __name__ == '__main__':
    main()
