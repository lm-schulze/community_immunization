# library imports
import numpy as np   
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns


def avg_over_reps(
    df: pd.DataFrame,
    group_cols: list[str],
    drop_cols: list[str] = None,
) -> pd.DataFrame:
    """Average all numeric columns over SIR / immunization/ network repetitions.

    Parameters
    ----------
    df         : raw simulation results
    group_cols : columns that uniquely identify the simulation parameter combination
                (here, should be rewire_steps, coverage, algorithm, beta, gamma)
    drop_cols  : repetetition index columns to discard after averaging

    Returns
    -------
    DataFrame with one row per condition, containing per-metric averages.
    """
    return (
        df.groupby(group_cols, as_index=False, observed=True)
          .mean(numeric_only=True)
          .drop(columns=[c for c in drop_cols if c in df.columns])
    )


def compute_best_algorithms(
    df_avg: pd.DataFrame,
    scenario_levels: list[str],
    metric_map: dict[str, str],
) -> pd.DataFrame:
    """For each parameter combo, find the algorithm with the best (lowest) value per result metric.

    Parameters
    ----------
    df_avg          : DataFrame averaged over replicates; must contain an
                      'algorithm' column plus all metric columns
    scenario_levels : columns that define a parameter combination, EXCLUDING 'algorithm'
    metric_map      : {output_column: result_metric_column}

    Returns
    -------
    DataFrame with parameter combos + any extra numeric columns (e.g. modularity)
    + one 'alg_*' column per metric holding the best algorithm's name.

    """
    grouped = df_avg.groupby(scenario_levels)

    # Carry forward the mean of numeric columns that are neither combo
    # nor metrics (e.g. modularity when grouping by rewire_steps).
    metric_vals  = set(metric_map.values())
    extra_numeric = [
        c for c in df_avg.select_dtypes("number").columns
        if c not in scenario_levels and c not in metric_vals
    ]

    if extra_numeric:
        best = grouped[extra_numeric].mean().reset_index()
    else:
        best = grouped.size().reset_index(name="_n").drop(columns="_n")

    for out_col, metric in metric_map.items():
        best_idx      = grouped[metric].idxmin()          # integer row labels
        best[out_col] = df_avg.loc[best_idx, "algorithm"].to_numpy()

    return best

##########################################
# and the visualisation funcs
##########################################

def save_legend(
    legend_handles: list,
    title: str = "Algorithm",
    save_path: str = "figures/algo_legend.png",
    ncol: int = None,
    fontsize: int = 10,
) -> plt.Figure:
    """Render just the legend (no axes/data) as its own figure, so it can be
    saved once and reused across multiple heatmap figures.

    Parameters
    ----------
    legend_handles : list of Patch objects, e.g. from make_color_encoding
    title          : legend title
    save_path      : where to save the standalone legend image
    ncol           : number of columns in the legend (default: all in one row)
    fontsize       : legend text size
    """
    fig = plt.figure(figsize=(0.1, 0.1))  # dummy canvas, legend defines the real bbox
    legend = fig.legend(
        handles=legend_handles,
        title=title,
        loc="center",
        ncol=ncol or len(legend_handles),
        frameon=True,
        fontsize=fontsize,
    )
    fig.canvas.draw()
    # trim the figure to just the legend's bounding box
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(save_path, dpi=300, bbox_inches=bbox.expanded(1.05, 1.05))
    plt.show()
    return fig


def make_color_encoding(algorithms: list[str], palette = None):
    """Make categorical colour encoding for a list of algorithms to be able to plot them
    as a heatmap with seaborn.

    Parameters
    ----------
    algorithms     : list of algorithm names for heatmap
    Returns
    -------
    algo_to_code   : {algorithm_name: integer_code}
    palette        : list of RGB tuples
    cmap           : ListedColormap
    norm           : BoundaryNorm
    legend_handles : list of Patch objects suitable for ax.legend(handles=...)
    """
    algorithms   = sorted(set(algorithms))
    n            = len(algorithms)
    algo_to_code = {alg: i for i, alg in enumerate(algorithms)}
    if palette is None:
        palette      = sns.color_palette("tab10", n_colors=n)
    cmap         = mcolors.ListedColormap(palette)
    norm         = mcolors.BoundaryNorm(np.arange(-0.5, n + 0.5, 1), ncolors=n)
    legend_handles = [
        mpatches.Patch(facecolor=palette[i], label=algorithms[i])
        for i in range(n)
    ]
    return algo_to_code, palette, cmap, norm, legend_handles


def _pivot_column(
    df: pd.DataFrame,
    col: str,
    algo_to_code: dict,
    row: str = "coverage",
    col_: str = "modularity",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot a best-algorithm column into numeric-code and label DataFrames,
    to bring it in the format needed for the seaborn heatmap
    """
    df = df.copy()
    df[col + "_code"] = df[col].map(algo_to_code)
    pivot_code  = df.pivot(index=row, columns=col_, values=col + "_code").iloc[::-1]
    pivot_label = df.pivot(index=row, columns=col_, values=col).iloc[::-1]
    return pivot_code, pivot_label


def plot_algo_heatmaps(
    best_filtered: pd.DataFrame,
    algo_to_code: dict,
    cmap,
    norm,
    legend_handles: list,
    metric_map: dict[str, str],
    col_titles: list[str],
    suptitle: str = "Best-performing algorithm by metric",
    save_path: str | None = None,
    annotate = True,
) -> plt.Figure:
    """Make vertically-stacked figure with the best-algorithm heatmaps for each result metric.

    Parameters
    ----------
    best_filtered  : output of compute_best_algorithms, filtered to a single
                     (beta, gamma) slice; must contain 'coverage' and
                     'modularity' (already rounded) with unique combinations.
    algo_to_code   : from make_color_encoding
    cmap / norm    : from make_color_encoding
    legend_handles : from make_color_encoding
    metric_map     : same mapping passed to compute_best_algorithms
    col_titles     : human-readable panel titles (same order as metric_map keys)
    save_path      : if given, save figure to this path at 1500 dpi
    annotate       : Whether to annotate the heatmap panels.
    """
    col_names = list(metric_map.keys())
    n_panels  = len(col_names)
    fig, axes = plt.subplots(n_panels, 1, figsize=(4.5, 2 * n_panels))  
    if n_panels == 1:
        axes = [axes]

    for ax, col, title in zip(axes, col_names, col_titles):
        pivot_code, pivot_label = _pivot_column(best_filtered, col, algo_to_code)
        sns.heatmap(
            pivot_code,
            ax=ax,
            cmap=cmap,
            norm=norm,
            linewidths=0.5,
            linecolor="black",
            annot=pivot_label if annotate else None,
            fmt="",
            annot_kws={"size": 10, "color": "black"},
            cbar=False,
        )
        ax.set_title(f"Best algorithm — {title}", fontsize=12, pad=10)
        ax.set_xlabel("Modularity", fontsize=12)
        ax.set_ylabel("Coverage",   fontsize=12)
        ax.legend(
            handles=legend_handles,
            title="Algorithm",
            loc="upper right",
            bbox_to_anchor=(1.3, 1),
            frameon=True,
            fontsize=10,
        )

    plt.suptitle(suptitle, fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def plot_metrics(
    data: pd.DataFrame,
    x_col: str,
    slice_col: str,
    slice_val: float,
    *,
    x_label: str,
    slice_label: str,
    metrics: list[str],
    metric_labels: dict,
    save_path: str | None = None,
) -> plt.Figure:
    """Draw a figure with one panel per result metric using seaborn lineplot.

    Parameters
    ----------
    data         : pre-filtered DataFrame (restricted to a single slice)
    x_col        : column for the x-axis ('modularity' or 'coverage')
    slice_col    : column used to select the slice (used in suptitle)
    slice_val    : value of that column (used in suptitle)
    x_label      : human-readable x-axis label
    slice_label  : human-readable slice description
    save_path    : if given, save figure to this path at 150 dpi
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5), sharey=False)
    fig.suptitle(f"{slice_label} = {slice_val:.2f}", fontsize=14, y=1.01)

    for ax, metric in zip(axes, metrics):
        sns.lineplot(
            data=data, x=x_col, y=metric,
            hue="algorithm", marker="o", ax=ax,
        )
        ax.set_xlabel(x_label,               fontsize=14)
        ax.set_ylabel(metric_labels[metric], fontsize=14)
        ax.set_title(metric_labels[metric],  fontsize=14)
        ax.get_legend().remove()
        ax.grid(alpha=0.3)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Algorithm",
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.12),
        frameon=True,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def pairwise_comparison(df, algo_A, algo_B, metric_map, metric_var_map=None, row="coverage", col_="modularity"):
    """Perform pairwise comparison of two algorithms across all parameter combos. Produces one plot per beta-gamma slice
    found in the df, showing the difference in each metric between the two algorithms as a heatmap over the (coverage, modularity) grid.

    Parameters
    ----------
    df      : DataFrame with columns 'algorithm', 'coverage', 'modularity', and metric columns
    algo_A  : name of first algorithm (must be present in df['algorithm'])
    algo_B  : name of second algorithm (must be present in df['algorithm'])
    metric_map : dict mapping output column names to metric column names, e.g. {"alg_final": "final_attack_ratio"}
    metric_var_map : dict mapping metric names to their LaTeX variable representations
    row      : column name to use as heatmap rows (default 'coverage')
    col_     : column name to use as heatmap columns (default 'modularity')

    Returns
    -------
    """
    metrics = metric_map.keys()
    algo_list = [algo_A, algo_B]
    # filter to only the two algorithms we're comparing
    df_algs = df.query("algorithm in @algo_list")
    print(f"Comparing algorithms: {algo_A} vs {algo_B}")

    # normalize beta/gamma to two decimals before iterating; avoids float noise in labels
    slices = (
        df_algs[["beta", "gamma"]]
               .round({"beta": 2, "gamma": 2})
               .drop_duplicates()
               .reset_index(drop=True)
    )

    for beta, gamma in slices.itertuples(index=False):
        beta = float(round(beta, 2))
        gamma = float(round(gamma, 2))

        df_slice = df_algs.loc[
            df_algs["beta"].round(2).eq(beta) &
            df_algs["gamma"].round(2).eq(gamma)
        ]

        pivot = df_slice.pivot_table(
            index=["coverage", "modularity"],
            columns="algorithm",
            values=metrics,
            aggfunc="mean"
        ).reset_index()

        fig, axes = plt.subplots(ncols=len(metrics), figsize=(15, 5))
        fig.suptitle(f"{algo_A} vs {algo_B} (β={beta:.2f}, γ={gamma:.2f})", fontsize=14)
        for i, metric in enumerate(metrics):
            pivot[f"{metric}_diff"] = pivot[(metric, algo_A)] - pivot[(metric, algo_B)]
            ax = axes[i]
            sns.heatmap(
                pivot.pivot(index=row, columns=col_, values=f"{metric}_diff"),
                cmap="RdBu_r",
                center=0,
                #annot=True,
                fmt=".2f",
                ax=ax
            )
            ax.invert_yaxis() # bc otherwise lowest coverage is at the top somehow??
            if metric_var_map is None:
                ax.set_title(f"Difference in {metric_map[metric]} ({algo_A} - {algo_B})")
            else:
                ax.set_title(f"Difference in {metric_map[metric]} (${metric_var_map[metric]}_{{\\mathrm{{{algo_A}}}}}$ - ${metric_var_map[metric]}_{{\\mathrm{{{algo_B}}}}}$)")
            ax.set_xlabel(col_)
            ax.set_ylabel(row)

        fig.tight_layout()
        fig.savefig(f"figures/pairwise_comparison_{algo_A}_vs_{algo_B}_beta{beta:.2f}_gamma{gamma:.2f}.png", dpi=300)   
        plt.show()


def pairwise_comparison_by_metric(df, algo_A, algo_B, metric_map, metric_var_map=None, row="coverage", col_="modularity"):
    """Perform pairwise comparison of two algorithms across all parameter combos. Produces one plot per
    result metric, with each subplot showing the difference in that metric between the two algorithms
    as a heatmap over the (coverage, modularity) grid, for a given beta/gamma slice.

    Parameters
    ----------
    df      : DataFrame with columns 'algorithm', 'coverage', 'modularity', and metric columns
    algo_A  : name of first algorithm (must be present in df['algorithm'])
    algo_B  : name of second algorithm (must be present in df['algorithm'])
    metric_map : dict mapping output column names to metric column names, e.g. {"alg_final": "final_attack_ratio"}
    metric_var_map : dict mapping metric names to their LaTeX variable representations
    row      : column name to use as heatmap rows (default 'coverage')
    col_     : column name to use as heatmap columns (default 'modularity')

    Returns
    -------
    """
    metrics = list(metric_map.keys())
    algo_list = [algo_A, algo_B]
    df_algs = df.query("algorithm in @algo_list")
    print(f"Comparing algorithms: {algo_A} vs {algo_B}")

    # normalize beta/gamma to two decimals before iterating; avoids float noise in labels
    slices = (
        df_algs[["beta", "gamma"]]
               .round({"beta": 2, "gamma": 2})
               .drop_duplicates()
               .sort_values(["gamma", "beta"], ascending=[True, True])
               .reset_index(drop=True)
    )

    # --- pass 1: compute one pivot (with all metric diffs) per beta/gamma slice, store for reuse ---
    slice_pivots = []
    for beta, gamma in slices.itertuples(index=False):
        beta = float(round(beta, 2))
        gamma = float(round(gamma, 2))

        df_slice = df_algs.loc[
            df_algs["beta"].round(2).eq(beta) &
            df_algs["gamma"].round(2).eq(gamma)
        ]

        pivot = df_slice.pivot_table(
            index=["coverage", "modularity"],
            columns="algorithm",
            values=metrics,
            aggfunc="mean"
        ).reset_index()

        for metric in metrics:
            pivot[f"{metric}_diff"] = pivot[(metric, algo_A)] - pivot[(metric, algo_B)]

        slice_pivots.append((beta, gamma, pivot))

    # --- pass 2: one figure per metric, one subplot per beta/gamma slice ---
    for metric in metrics:
        n_slices = len(slice_pivots)
        fig, axes = plt.subplots(ncols=n_slices, figsize=(4.55 * n_slices, 3.5), squeeze=False)
        axes = axes[0]  # squeeze=False always gives 2D array; flatten to 1D

        if metric_var_map is None:
            fig.suptitle(f"Difference in {metric_map[metric]} ({algo_A} - {algo_B})", fontsize=14)
        else:
            fig.suptitle(
                f"Difference in {metric_map[metric]} "
                f"(${metric_var_map[metric]}_{{\\mathrm{{{algo_A}}}}}$ - ${metric_var_map[metric]}_{{\\mathrm{{{algo_B}}}}}$)",
                fontsize=14,
            )

        for ax, (beta, gamma, pivot) in zip(axes, slice_pivots):
            sns.heatmap(
                pivot.pivot(index=row, columns=col_, values=f"{metric}_diff"),
                cmap="RdBu_r",
                center=0,
                #annot=True,
                fmt=".2f",
                ax=ax,
            )
            ax.invert_yaxis()  # bc otherwise lowest coverage is at the top somehow??
            ax.set_title(f"β={beta:.2f}, γ={gamma:.2f}")
            ax.set_xlabel(col_)
            ax.set_ylabel(row)
            plt.yticks(rotation=0)

        fig.tight_layout()
        fig.savefig(f"figures/pairwise_comparison_{algo_A}_vs_{algo_B}_{metric}.png", dpi=300)
        plt.show()
