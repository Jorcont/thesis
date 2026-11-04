import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde
import geopandas as gpd
from shapely.geometry import Point

# ── Load data ─────────────────────────────────────────────────────────
plants = pd.read_parquet(
    "/mnt/data/kes/data/jdomingo/nwp_parquet/plant_nwp_mapping.parquet"
)
grid = pd.read_parquet(
    "/mnt/data/kes/data/jdomingo/nwp_parquet/grid_coords.parquet"
)

# ── Aggregate duplicate asset_ids ─────────────────────────────────────
plants = (
    plants.groupby("asset_id")
    .agg(
        LATITUDE=("LATITUDE", "first"),
        LONGITUDE=("LONGITUDE", "first"),
        capacity_kW=("LEISTUNG [kW]", "sum"),
    )
    .reset_index()
)

print(f"Unique plants : {len(plants)}")
print(f"Total capacity: {plants['capacity_kW'].sum() / 1e3:.1f} MW")
print(f"Capacity range: {plants['capacity_kW'].min():.1f} – "
      f"{plants['capacity_kW'].max():.1f} kW")

# ── Germany border (requires geopandas + naturalearth data) ───────────
# pip install geopandas geodatasets
try:
    # import geodatasets
    # world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
    # germany = gpd.read_file(geodatasets.get_path("naturalearth.land"))
    # # Better: use a dedicated Germany shapefile
    # world = gpd.read_file(
    #     "https://raw.githubusercontent.com/datasets/geo-countries/"
    #     "master/data/countries.geojson"
    # )
    # germany = world[world["ADMIN"] == "Germany"]
    # has_border = True
    # import geodatasets
    # path = geodatasets.get_path("naturalearth countries")
    # germany = gpd.read_file(path)
    # germany = germany[germany["NAME"] == "Germany"]
    # has_border = True
    import geopandas as gpd

    germany = gpd.read_file(
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )
    germany = germany[germany["NAME"] == "Germany"]
    has_border = True
except Exception:
    has_border = False
    print("⚠️  Germany border not available — install geopandas + geodatasets")

# ── Helper: draw Germany border ───────────────────────────────────────
def draw_germany(ax, color="black", lw=0.8):
    if has_border:
        germany.boundary.plot(ax=ax, color=color, linewidth=lw)

# ── Axis limits (Germany bounding box) ────────────────────────────────
LON_MIN, LON_MAX = 5.8, 15.2
LAT_MIN, LAT_MAX = 47.2, 55.1

def set_germany_limits(ax):
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linestyle="--")

# ═══════════════════════════════════════════════════════════════════════
# PLOT 1 — Location scatter (simple dots)
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 8))
draw_germany(ax)

# NWP grid as background
ax.scatter(
    grid["lons"], grid["lats"],
    s=4, color="lightblue", alpha=0.3, zorder=1, label="NWP grid"
)
# Plant locations
ax.scatter(
    plants["LONGITUDE"], plants["LATITUDE"],
    s=15, color="darkorange", alpha=0.7, edgecolors="none",
    zorder=3, label=f"PV plants (n={len(plants)})"
)

set_germany_limits(ax)
ax.set_title("PV Plant Locations — Dataset Coverage", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("plot1_plant_locations.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: plot1_plant_locations.png")

# ═══════════════════════════════════════════════════════════════════════
# PLOT 2 — Bubble map (size ∝ installed capacity)
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 8))
draw_germany(ax)

cap = plants["capacity_kW"].values
# Normalize bubble sizes: min=10, max=300
sizes = 10 + 290 * (cap - cap.min()) / (cap.max() - cap.min() + 1e-8)

scatter = ax.scatter(
    plants["LONGITUDE"], plants["LATITUDE"],
    s=sizes,
    c=cap,
    cmap="YlOrRd",
    alpha=0.6,
    edgecolors="gray",
    linewidths=0.3,
    zorder=3,
)
cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label("Installed Capacity (kW)", fontsize=10)

# Legend for bubble sizes
for ref_kw in [100, 500, 1000]:
    if ref_kw <= cap.max():
        s = 10 + 290 * (ref_kw - cap.min()) / (cap.max() - cap.min() + 1e-8)
        ax.scatter([], [], s=s, c="gray", alpha=0.5, label=f"{ref_kw} kW")

set_germany_limits(ax)
ax.set_title("PV Plant Locations — Installed Capacity", fontsize=13, fontweight="bold")
ax.legend(title="Capacity", loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("plot2_capacity_bubbles.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: plot2_capacity_bubbles.png")

# ═══════════════════════════════════════════════════════════════════════
# PLOT 3 — KDE density heatmap
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 8))
draw_germany(ax)

# Compute 2D KDE
lon_vals = plants["LONGITUDE"].values
lat_vals = plants["LATITUDE"].values
xy = np.vstack([lon_vals, lat_vals])
kde = gaussian_kde(xy, bw_method=0.15)

# Evaluate on grid
lon_grid = np.linspace(LON_MIN, LON_MAX, 200)
lat_grid = np.linspace(LAT_MIN, LAT_MAX, 200)
LON_G, LAT_G = np.meshgrid(lon_grid, lat_grid)
Z = kde(np.vstack([LON_G.ravel(), LAT_G.ravel()])).reshape(LON_G.shape)

# Plot
cf = ax.contourf(LON_G, LAT_G, Z, levels=20, cmap="YlOrRd", alpha=0.8, zorder=2)
ax.contour(LON_G, LAT_G, Z, levels=10, colors="white", alpha=0.3,
           linewidths=0.5, zorder=3)
draw_germany(ax, color="black", lw=1.2)

cbar = plt.colorbar(cf, ax=ax, pad=0.02)
cbar.set_label("Plant Density (KDE)", fontsize=10)

set_germany_limits(ax)
ax.set_title("PV Plant Density Distribution", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot3_kde_density.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: plot3_kde_density.png")

# ═══════════════════════════════════════════════════════════════════════
# PLOT 4 — NWP grid resolution + plant coverage overlay
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 8))
draw_germany(ax)

# NWP grid tiles
ax.scatter(
    grid["lons"], grid["lats"],
    s=6, color="steelblue", alpha=0.4, zorder=1, label="NWP tile center"
)
# Plants
ax.scatter(
    plants["LONGITUDE"], plants["LATITUDE"],
    s=12, color="darkorange", alpha=0.8,
    edgecolors="none", zorder=3, label="PV plant"
)

# Draw NWP tile boxes (0.0625° grid spacing — adjust if different)
tile_spacing = 0.0625
for _, row in grid.sample(min(300, len(grid))).iterrows():
    rect = plt.Rectangle(
        (row["lons"] - tile_spacing / 2, row["lats"] - tile_spacing / 2),
        tile_spacing, tile_spacing,
        linewidth=0.3, edgecolor="steelblue", facecolor="none", alpha=0.3,
        zorder=2,
    )
    ax.add_patch(rect)

set_germany_limits(ax)
ax.set_title("NWP Grid Resolution vs PV Plant Locations", fontsize=13,
             fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("plot4_nwp_grid_overlay.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: plot4_nwp_grid_overlay.png")

# ═══════════════════════════════════════════════════════════════════════
# PLOT 5 — Capacity histogram (north vs south)
# ═══════════════════════════════════════════════════════════════════════
LAT_MID = 51.0  # approximate Germany midpoint
north = plants[plants["LATITUDE"] >= LAT_MID]["capacity_kW"]
south = plants[plants["LATITUDE"] <  LAT_MID]["capacity_kW"]

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.logspace(
    np.log10(plants["capacity_kW"].min() + 1),
    np.log10(plants["capacity_kW"].max() + 1),
    30
)
ax.hist(north, bins=bins, alpha=0.6, color="steelblue",
        label=f"North (lat≥{LAT_MID}°, n={len(north)})")
ax.hist(south, bins=bins, alpha=0.6, color="darkorange",
        label=f"South (lat<{LAT_MID}°, n={len(south)})")

ax.set_xscale("log")
ax.set_xlabel("Installed Capacity (kW, log scale)", fontsize=11)
ax.set_ylabel("Number of Plants", fontsize=11)
ax.set_title("Capacity Distribution — North vs South Germany", fontsize=13,
             fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot5_capacity_histogram.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: plot5_capacity_histogram.png")
