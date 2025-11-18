import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from xskillscore.core.probabilistic import rank_histogram


def plot_rank_histogram(rank_hist, normalize=True, title=None):
    """
    Plot a Rank Histogram (Talagrand Diagram) from the direct output of
    xskillscore.rank_histogram().

    Parameters
    ----------
    rank_hist : xr.DataArray or xr.Dataset
        Output of xskillscore.rank_histogram(...).
        Must include a 'rank' dimension.

    normalize : bool, default=True
        If True, converts counts to frequencies (sum to 1).

    title : str or None
        Optional custom plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """

    # ----------------------------------------
    # If Dataset → assume exactly one variable
    # ----------------------------------------
    if isinstance(rank_hist, xr.Dataset):
        if len(rank_hist.data_vars) != 1:
            raise ValueError("Dataset must contain exactly one variable.")
        rank_hist = next(iter(rank_hist.data_vars.values()))

    if "rank" not in rank_hist.dims:
        raise ValueError("Input must contain a 'rank' dimension.")

    # --------------------------------------------------
    # Collapse all non-rank dimensions → get 1D histogram
    # --------------------------------------------------
    reduce_dims = [d for d in rank_hist.dims if d != "rank"]
    hist_1d = rank_hist.sum(dim=reduce_dims)

    counts = hist_1d.values.astype(float)

    # Normalize to frequencies if requested
    if normalize:
        total = counts.sum()
        if total > 0:
            counts = counts / total

    ranks = hist_1d["rank"].values

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(ranks, counts, width=0.8, edgecolor="black")

    ax.set_xlabel("Rank of Observation Among Ensemble Members")
    ax.set_ylabel("Frequency" if normalize else "Count")

    if title is None:
        title = "Rank Histogram (Talagrand Diagram)"
    ax.set_title(title)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xticks(ranks)

    return fig, ax


def plot_rank_histogram_map_cartopy(
    observations: xr.DataArray,
    forecasts: xr.DataArray,
    member_dim: str = "member",
    dim: str = "time",
    rank: int = 1,
    normalize: bool = True,
    projection=ccrs.Robinson(),
    cmap="viridis",
    figsize=(14, 6),
):
    """
    Plot a proper global spatial map of rank-histogram frequency using Cartopy.

    Parameters
    ----------
    observations : xr.DataArray
        Observations with dims including `time`, `lat`, `lon`.
    forecasts : xr.DataArray
        Forecasts with `member_dim`.
    member_dim : str
        Ensemble member dimension name.
    dim : str
        Dimension over which histogram is computed (usually "time").
    rank : int
        Which rank to plot (1 ... n_members+1).
    normalize : bool
        If True, show frequency; else show counts.
    projection : cartopy.crs
        Cartopy projection (default: Robinson).
    cmap : str
        Colormap.
    figsize : tuple
        Figure size.

    Returns
    -------
    freq : xr.DataArray
        The plotted spatial frequency/count field.
    """

    # ---------------------------------------------------------
    # (1) Compute rank histogram
    # ---------------------------------------------------------
    rh = rank_histogram(
        observations=observations,
        forecasts=forecasts,
        dim=dim,
        member_dim=member_dim,
        keep_attrs=False,
    )
    n_ranks = rh.sizes["rank"]

    if not (1 <= rank <= n_ranks):
        raise ValueError(f"Rank must be in [1, {n_ranks}], got {rank}")

    # rank index = rank - 1
    freq = rh.isel(rank=rank - 1)

    if normalize:
        total = rh.sum("rank")
        freq = freq / total

    # ---------------------------------------------------------
    # (2) Prepare grid for Cartopy
    # ---------------------------------------------------------
    lat = freq["lat"].values
    lon = freq["lon"].values

    # Meshgrid
    lon2d, lat2d = np.meshgrid(lon, lat)

    # ---------------------------------------------------------
    # (3) Plot with Cartopy
    # ---------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)
    ax.set_global()

    # Add coastlines, borders, and gridlines
    ax.coastlines(linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.gridlines(draw_labels=False, linewidth=0.2, alpha=0.5)

    # Plot using PlateCarree coordinates
    pcm = ax.pcolormesh(
        lon2d,
        lat2d,
        freq.values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        shading="auto",
    )

    plt.colorbar(
        pcm,
        orientation="horizontal",
        pad=0.04,
        fraction=0.05,
        label=f"Rank {rank} Frequency" if normalize else f"Rank {rank} Count",
    )

    plt.title(f"Rank Histogram Spatial Map (Rank={rank}, dim='{dim}')")
    plt.tight_layout()
    plt.show()

    return freq
