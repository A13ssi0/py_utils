from rich import box
from rich.console import Console
from rich.table import Table

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
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

def get_x(scatter_data):
    return np.arange(len(scatter_data))

def plot_similarity_matrices(
    matrix_angleCos,
    matrix_distance,
    n_classes=None,
    day_start_idx=[],
    stop_idx=[],
    rec_idx=[],
    trial_accuracy=None,
    accuracy=None,
    d_max=None,
    rejection=None,
    cmap_name='PuBu',
    cell_size=2.2,  
    saveFigure=False,
    savingPath='.',
    filename='figure.svg'
):

    # ---------------- DATA ----------------
    angle_data = np.array(matrix_angleCos)
    dist_data  = np.array(matrix_distance)

    shape = angle_data.shape

    if len(shape) == 4:
        n_bands, n_run, _, n_classes_ = shape
    elif len(shape) == 3:
        n_bands, n_run, _ = shape
        n_classes_ = 1
    else:
        raise ValueError("Unsupported matrix shape")

    if n_classes is None:
        n_classes = n_classes_

    # ---------------- NORMALIZATION ----------------
    angle_min, angle_max = 0,1
    dist_min = 0
    # dist_min, dist_max   = np.min(dist_data), np.max(dist_data)
    q1 = np.nanpercentile(dist_data, 25)
    q3 = np.nanpercentile(dist_data, 75)
    iqr = q3 - q1

    dist_max = q3 + 2 * iqr

    cmap = plt.get_cmap(cmap_name)

    # ---------------- LAYOUT ----------------
    show_top = accuracy is not None or rejection is not None or trial_accuracy is not None
    top_rows = 1 if show_top else 0

    n_rows = top_rows + 2
    n_cols = n_classes

    fig_width  = n_cols * cell_size
    fig_height = (1.5 * cell_size) + (0.5 * top_rows * cell_size)

    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = gridspec.GridSpec(
        n_rows,
        n_cols + 1,
        height_ratios=([0.25] if show_top else []) + [1, 1],
        width_ratios=[1]*n_cols + [0.05],
        hspace=0.05,
        wspace=0.2
    )

    axes = np.empty((n_rows, n_cols), dtype=object)

    for i in range(n_rows):
        for j in range(n_cols):
            axes[i, j] = fig.add_subplot(gs[i, j])

    base_runs = np.arange(n_run)
    color_bg = "gainsboro"

    # ---------------- TOP ROW ----------------
    if show_top:
        for c in range(n_classes):
            ax = axes[0, c]

            if trial_accuracy is not None:
                mask = ~np.isnan(trial_accuracy[:,c])
                ax.bar(base_runs[mask], trial_accuracy[mask,c],
                       color='b', alpha=0.4,    
                       label='Trial Acc.',
                       edgecolor='darkblue')

            if rejection is not None:
                mask = ~np.isnan(rejection[:,c])
                ax.scatter(base_runs[mask], rejection[mask,c],
                           s=12, color='r',
                           label='Rejection',
                           edgecolors='darkred')
                
            if accuracy is not None:
                mask = ~np.isnan(accuracy[:,c])
                ax.scatter(base_runs[mask], accuracy[mask,c],
                           s=12, color='g',
                           label='Accuracy',
                           edgecolors='darkgreen')
                


            ax.set_xlim(-0.5, n_run - 0.5)
            ax.set_ylim(0, 1)

            ax.set_xticks([])
            ax.set_yticks(np.linspace(0, 1, 5))



            ax.set_title(f'Class {c}', fontsize=11, pad=8)

            if c == 0:
                ax.set_ylabel("Performance")


            # vertical markers
            for k in day_start_idx:
                ax.axvline(k - 0.5, color='k', lw=0.5, alpha=0.5, zorder=-10)

            for k in stop_idx:
                ax.axvline(k - 0.5, color='r', lw=1, zorder=-10)

            for k in rec_idx:
                ax.axvline(k - 0.5, color='orange', lw=1, zorder=-10)

            # horizontal grid
            for y in ax.get_yticks():
                ax.axhline(y, color=color_bg, lw=1, zorder=-10)

            if c > 0:
                ax.set_yticks([])

            ax.legend(loc='upper center',
                        bbox_to_anchor=(0.5, 0.1),
                        ncol=3 if trial_accuracy is not None else 2,
                        frameon=False)

    # ---------------- MATRICES ----------------
    row_offset = top_rows

    for c in range(n_classes):

        ax_a = axes[row_offset, c]
        ax_d = axes[row_offset + 1, c]

        # if n_classes > 1:
        angle_mat = angle_data[0, :, :, c] if len(shape) == 4 else angle_data
        dist_mat  = dist_data[0, :, :, c]  if len(shape) == 4 else dist_data
        # else:
        #     angle_mat = angle_data[0] if len(shape) == 3 else angle_data
        #     dist_mat  = dist_data[0]  if len(shape) == 3 else dist_data

        im_a = ax_a.imshow(
            angle_mat,
            cmap=cmap.reversed(),
            vmin=angle_min,
            vmax=angle_max,
            aspect='auto',
            extent=[-0.5, n_run - 0.5, n_run - 0.5, -0.5]
        )

        im_d = ax_d.imshow(
            dist_mat,
            cmap=cmap,
            vmin=dist_min,
            vmax=dist_max,
            aspect='auto',
            extent=[-0.5, n_run - 0.5, n_run - 0.5, -0.5]
        )

        # ---- FORCE SQUARE AXES ----
        ax_a.set_box_aspect(1)
        ax_d.set_box_aspect(1)

        for ax in (ax_a, ax_d):
            ax.set_xticks([])
            ax.set_yticks([])

            # markers
            for k in day_start_idx:
                ax.axvline(k - 0.5, color='k', lw=0.5, alpha=0.5)
                ax.axhline(k - 0.5, color='k', lw=0.5, alpha=0.5)

            for k in stop_idx:
                ax.axvline(k - 0.5, color='r', lw=0.5)
                ax.axhline(k - 0.5, color='r', lw=0.5)

            for k in rec_idx:
                ax.axvline(k - 0.5, color='orange', lw=0.5)
                ax.axhline(k - 0.5, color='orange', lw=0.5)

        if c == 0:
            ax_a.set_ylabel("Angle")
            ax_d.set_ylabel("Distance")

        # ---- SHARE X WITH TOP ----
        if show_top:
            ax_a.sharex(axes[0, c])
            ax_d.sharex(axes[0, c])

        for ax in axes.flatten():
            ax.margins(0)

    # ---------------- COLORBARS ----------------
    cax_a = fig.add_subplot(gs[row_offset, -1])
    cax_d = fig.add_subplot(gs[row_offset + 1, -1])

    fig.colorbar(cm.ScalarMappable(
        norm=mcolors.Normalize(angle_min, angle_max),
        cmap=cmap.reversed()), cax=cax_a)

    fig.colorbar(cm.ScalarMappable(
        norm=mcolors.Normalize(dist_min, dist_max),
        cmap=cmap), cax=cax_d)

    plt.subplots_adjust(
        top=0.95,
        bottom=0.08,
        left=0.15,
        right=0.9,
        hspace=0.5
    )
    # plt.tight_layout()
    plt.show()

    # ---------------- SAVE ----------------
    if saveFigure:
        path = f"{savingPath}/{filename}"
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f"Saved to {path}")
