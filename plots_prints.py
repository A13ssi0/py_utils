from rich.console import Console
from rich.table import Table
from rich import box
import numpy as np

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

