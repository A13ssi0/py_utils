from rich.console import Console
from rich.table import Table
from rich import box
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

def plot_confusion_matrix(matrix, labels=None):
    console = Console()
    n = len(matrix)
    table = Table(show_header=True, header_style="bold bright_cyan", box=box.SIMPLE_HEAVY)

    # Add column headers
    table.add_column(" ", style="bold bright_cyan")
    for i in range(n):
        label = labels[i] if labels else str(i)
        table.add_column(f"P_{label}", justify="center")

    # Add matrix rows
    max_val = matrix.max()
    for i in range(n):
        label = labels[i] if labels else str(i)
        row = [f"[bright_yellow]T_{label}[/]"]
        for j in range(n):
            val = matrix[i][j]
            ratio = val / max_val if max_val else 0

            # Bright color gradient
            if i == j:
                color = "bold bright_green"
            elif ratio > 0.66:
                color = "bold bright_red"
            elif ratio > 0.33:
                color = "bright_magenta"
            else:
                color = "bright_black"

            row.append(f"[{color}]{val}[/]")
        table.add_row(*row)

    console.print(table)


def fmt(x, decimals=3):
    arr = np.array(x)

    # Create format string like ".3f", ".5f", etc.
    fs = f"{{:.{decimals}f}}"

    # 0-D array → scalar
    if arr.ndim == 0:
        return fs.format(arr.item())

    # 1-D array → [a, b]
    if arr.ndim == 1:
        return "[" + ", ".join(fs.format(v) for v in arr) + "]"

    # 2-D array → [[a, b], [c, d]]
    if arr.ndim == 2:
        return "[" + ", ".join(
            "[" + ", ".join(fs.format(v) for v in row) + "]"
            for row in arr
        ) + "]"

    # fallback
    return str(arr)

def plot_array_runs_grid(
    data_array,
    lbl=None,
    y_col_name="Y",
    xlim=None, ylim=None,
    fitLine=False,
    scatter=True,
    mergeTask=False,
    averageOnTask=False,
    day_start_idx=None,
    day_labels=None,
    useHuber=False,
    idxStop=None,
    idxRec=None,
    title=None,
    accuracy=None,
    rejection=None,
):

    n_bands, n_runs, n_tasks = (
        data_array.shape if len(data_array.shape) == 3
        else (*data_array.shape, 1)
    )

    show_top = accuracy is not None or rejection is not None
    top_rows = 1 if show_top else 0
    data_rows = 1 if mergeTask or averageOnTask else n_tasks

    height_ratios = [0.3] * top_rows + [1.0] * data_rows
    fig = plt.figure(figsize=(16, 1.5*top_rows + 6*data_rows))
    gs = gridspec.GridSpec(
        top_rows + data_rows,
        n_bands,
        height_ratios=height_ratios,
        hspace=0.3
    )

    axes = np.empty((top_rows + data_rows, n_bands), dtype=object)
    for i in range(top_rows + data_rows):
        for j in range(n_bands):
            axes[i, j] = fig.add_subplot(gs[i, j])

    # ---- FORCE SHARE X BETWEEN TOP AND BOTTOM ----
# ---- FORCE SAME X LIMITS FOR TOP AND BOTTOM ----
    if show_top:
        for j in range(n_bands):
            axes[0, j].set_xlim(0, n_runs - 1)


    base_runs = np.arange(n_runs)
    cmap = plt.get_cmap("tab10")
    color_backgrounLines = "gainsboro"

    # ---------------- TOP ROW ----------------
    if show_top:
        for idxBand in range(n_bands):
            ax = axes[0, idxBand]

            if accuracy is not None:
                ax.scatter(base_runs, accuracy,
                           s=15, color='g', label='Accuracy', edgecolors='darkgreen')
            if rejection is not None:
                ax.scatter(base_runs, rejection,
                           s=15, color='r', label='Rejection', edgecolors='darkred')

            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', labelbottom=False)
            ax.set_xticks(day_start_idx)

            ax.tick_params(axis='both',
                           direction='in', length=5)



            if idxBand == 0:
                ax.set_ylabel("Performance")

            if day_start_idx is not None:
                for k in day_start_idx:
                    ax.axvline(k, color='k', lw=0.5, alpha=0.3, zorder=0)

            if idxStop is not None:
                for k in idxStop:
                    ax.axvline(k, color='r', lw=1.5, zorder=0)

            if idxRec is not None:
                for k in idxRec:
                    ax.axvline(k, color='orange', lw=1.5, zorder=0)

            # ---- HORIZONTAL GRID (TOP) ----
            yticks = ax.get_yticks()
            for y in yticks:
                ax.axhline(y, color=color_backgrounLines,
                           lw=1, zorder=-10)

            if idxBand == n_bands - 1:
                ax.legend(
                    loc='upper center',
                    bbox_to_anchor=(0.5, 1.25),
                    ncol=2,
                    frameon=False
                )


    # ---------------- MAIN DATA ----------------
    for idxTask in range(data_rows):
        for idxBand in range(n_bands):
            ax = axes[top_rows + idxTask, idxBand]

            # Vertical background
            if day_start_idx is not None:
                for k in day_start_idx:
                    ax.axvline(k, color=color_backgrounLines,
                               lw=1, zorder=0)

            # Stop / Rec with labels only once
            if idxStop is not None:
                for i, k in enumerate(idxStop):
                    ax.axvline(
                        k, color="red", lw=1.5, zorder=1,
                        label="Break" if (i == 0 and
                        idxTask == data_rows-1 and
                        idxBand == n_bands-1) else None
                    )

            if idxRec is not None:
                for i, k in enumerate(idxRec):
                    ax.axvline(
                        k, color="orange", lw=1.5, zorder=1,
                        label="Recalibration" if (i == 0 and
                        idxTask == data_rows-1 and
                        idxBand == n_bands-1) else None
                    )

            ax.set_xticks(day_start_idx)
                # ax.set_xticklabels(day_labels,
                #                    rotation=60, ha='right')

            # -------- SELECT DATA --------
            if not mergeTask and not averageOnTask:
                y_base = data_array[idxBand, :, idxTask]
                runs = base_runs
                lbl_local = lbl
            else:
                band_data = data_array[idxBand, :, :]
                y_base = (
                    np.nanmean(band_data, axis=1)
                    if averageOnTask
                    else band_data.reshape(-1)
                )
                runs = (
                    base_runs if averageOnTask
                    else np.tile(base_runs, n_tasks)
                )
                lbl_local = (
                    lbl if averageOnTask
                    else np.tile(lbl, n_tasks)
                    if lbl is not None else None
                )

            mask = ~np.isnan(y_base)
            y_base, runs = y_base[mask], runs[mask]
            if lbl_local is not None:
                lbl_local = lbl_local[mask]

            # -------- SCATTER --------
            if lbl_local is not None:
                segments = np.unique(lbl_local)
                i_color = -1
                for i, seg in enumerate(segments):
                    i_color += 1
                    if i_color == 1:  i_color += 1
                    if i_color == 3:  i_color += 1
                    color = cmap(i_color % cmap.N)
                    # color = cmap(i % cmap.N)
                    xs = runs[lbl_local == seg]
                    ys = y_base[lbl_local == seg]
                    ax.scatter(xs, ys, color=color,
                               edgecolors='k', alpha=0.85)
                    if useHuber and len(xs) > 1:
                        X = xs.reshape(-1, 1)

                        x_scaler = StandardScaler()
                        y_scaler = StandardScaler()

                        Xs = x_scaler.fit_transform(X)
                        Ys = y_scaler.fit_transform(
                            ys.reshape(-1, 1)
                        ).ravel()

                        huber = HuberRegressor(epsilon=2).fit(Xs, Ys)

                        xs_ext = np.append(xs, xs.max() + 1)
                        Xs_ext = x_scaler.transform(xs_ext.reshape(-1, 1))
                        pred_scaled = huber.predict(Xs_ext)
                        pred_ext = y_scaler.inverse_transform(
                            pred_scaled.reshape(-1, 1)
                        ).ravel()

                        ax.plot(xs_ext, pred_ext,
                                color=color, linewidth=2)

            else:
                ax.scatter(runs, y_base,
                           edgecolors='k', alpha=0.7)

            ax.set_xlim([0, n_runs-1] if xlim is None else xlim)
            ax.set_ylim(
                [0, np.nanmax(y_base)*1.05]
                if ylim is None else ylim
            )

            if idxBand == 0:
                ax.set_ylabel(y_col_name)

            if day_start_idx is not None and day_labels is not None:
                ax.set_xticks(day_start_idx)
                ax.set_xticklabels(day_labels,
                                   rotation=60, ha='right')

            # ---- HORIZONTAL GRID (BOTTOM) ----
            yticks = ax.get_yticks()
            for y in yticks:
                ax.axhline(y, color=color_backgrounLines,
                           lw=1, zorder=0)

            ax.tick_params(axis='both',
                           direction='in', length=5)

            # Legend only on bottom-right
            if (idxTask == data_rows-1 and
                idxBand == n_bands-1):
                ax.legend(loc='upper right',
                          frameon=True)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.show()
    return fig