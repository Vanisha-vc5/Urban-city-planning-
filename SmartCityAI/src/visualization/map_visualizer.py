"""
map_visualizer.py
=================
Phase 8 - Visualization: Interactive Folium City Map

PURPOSE:
    Creates interactive Folium maps with layered overlays for:
    - City base map with OSM tiles
    - Population density choropleth on H3 hexagons
    - Existing hospital/school/EV/fire station markers
    - Recommended new sites (pulsing markers)
    - Service area coverage circles
    - Emergency response routes (from A*)

DESIGN:
    - Folium LayerControl: toggle each layer on/off
    - Choropleth: color zones by any feature (density, gap, priority)
    - Custom icons: hospital=🏥, school=🏫, ev=⚡, fire=🚒
    - Popup cards: click any zone for explanation panel
    - Dark tile layers for premium aesthetic

USAGE:
    from map_visualizer import CityMapVisualizer
    viz = CityMapVisualizer(zones_df=df, zones_gdf=gdf)
    m = viz.create_base_map()
    viz.add_population_layer(m)
    viz.add_hospital_markers(m, hospitals_gdf)
    viz.add_recommendations(m, ranked_df)
    m.save("city_map.html")
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import folium
import folium.plugins as fp
import geopandas as gpd
import numpy as np
import pandas as pd
from branca.colormap import LinearColormap

# ── Configuration ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("map_visualizer")

BASE_DIR      = Path(__file__).resolve().parents[3]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR       = BASE_DIR / "data" / "raw"
MODELS_DIR    = BASE_DIR / "models"

# Map tile options (premium look)
TILE_OPTIONS = {
    "dark":         "CartoDB dark_matter",
    "light":        "CartoDB positron",
    "standard":     "OpenStreetMap",
    "satellite":    "Esri WorldImagery",
}

# Facility marker configs
FACILITY_CONFIGS = {
    "hospital":    {"icon": "plus-sign", "color": "red",    "prefix": "glyphicon", "emoji": "🏥"},
    "school":      {"icon": "education", "color": "blue",   "prefix": "glyphicon", "emoji": "🏫"},
    "ev_station":  {"icon": "flash",     "color": "green",  "prefix": "glyphicon", "emoji": "⚡"},
    "fire_station":{"icon": "fire",      "color": "orange", "prefix": "glyphicon", "emoji": "🚒"},
    "recommended": {"icon": "star",      "color": "purple", "prefix": "glyphicon", "emoji": "⭐"},
}

# Priority class colors
PRIORITY_COLORS = {
    "High":   "#E74C3C",  # Red
    "Medium": "#F39C12",  # Orange
    "Low":    "#27AE60",  # Green
}


class CityMapVisualizer:
    """
    Interactive city map builder using Folium.

    Creates layered maps with choropleth, markers, and popups
    that can be embedded directly in the Streamlit dashboard.
    """

    def __init__(
        self,
        zones_df:   pd.DataFrame,
        zones_gdf:  Optional[gpd.GeoDataFrame] = None,
        city_center: Tuple[float, float] = (19.0, 72.85),
        zoom_start:  int = 12,
    ):
        """
        Initialize visualizer.

        Args:
            zones_df:    Zone features DataFrame
            zones_gdf:   H3 hexagon GeoDataFrame (with geometry)
            city_center: (lat, lon) of map center
            zoom_start:  Initial zoom level
        """
        self.df          = zones_df
        self.gdf         = zones_gdf
        self.city_center = city_center
        self.zoom_start  = zoom_start

        if self.gdf is None:
            self._try_load_gdf()

    def _try_load_gdf(self) -> None:
        """Attempt to load zones GeoDataFrame from disk."""
        gpkg = PROCESSED_DIR / "zones.gpkg"
        if gpkg.exists():
            self.gdf = gpd.read_file(str(gpkg))
            log.info(f"Loaded zones GeoDataFrame: {len(self.gdf)} zones")
        else:
            log.warning("No zones GeoPackage found. Choropleth layer unavailable.")

    def create_base_map(self, tile_style: str = "dark") -> folium.Map:
        """
        Create base Folium map.

        Args:
            tile_style: Map style key from TILE_OPTIONS

        Returns:
            folium.Map instance
        """
        tiles = TILE_OPTIONS.get(tile_style, "CartoDB dark_matter")

        m = folium.Map(
            location=list(self.city_center),
            zoom_start=self.zoom_start,
            tiles=tiles,
            control_scale=True,
            prefer_canvas=True,
        )

        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)

        # Add fullscreen button
        fp.Fullscreen(
            position="topleft",
            title="Fullscreen",
            title_cancel="Exit Fullscreen",
        ).add_to(m)

        # Add minimap
        fp.MiniMap(tile_layer=TILE_OPTIONS["light"], zoom_level_offset=-5).add_to(m)

        log.info(f"Base map created: center={self.city_center}, zoom={self.zoom_start}")
        return m

    def add_population_choropleth(
        self,
        m: folium.Map,
        feature_col: str = "population_density",
    ) -> folium.Map:
        """
        Add population density choropleth layer over H3 hexagons.

        Colors: Light yellow (low density) → Deep red (high density)
        Viridis-inspired colormap for perceptual uniformity.

        Args:
            m:           Folium map
            feature_col: Column to visualize

        Returns:
            Updated map
        """
        if self.gdf is None:
            log.warning("No GeoDataFrame — skipping choropleth")
            return m

        # Merge feature data
        gdf_merged = self.gdf.copy()
        if feature_col in self.df.columns and "h3_id" in self.df.columns:
            gdf_merged = gdf_merged.merge(
                self.df[["h3_id", feature_col]].rename(columns={feature_col: "viz_val"}),
                on="h3_id",
                how="left",
            )
            gdf_merged["viz_val"] = gdf_merged["viz_val"].fillna(0)
        elif feature_col in gdf_merged.columns:
            gdf_merged["viz_val"] = gdf_merged[feature_col]
        else:
            log.warning(f"Column {feature_col} not found")
            return m

        # Color scale
        min_val = gdf_merged["viz_val"].min()
        max_val = gdf_merged["viz_val"].max()
        colormap = LinearColormap(
            colors=["#FFFDE7", "#FF8F00", "#B71C1C"],
            vmin=min_val,
            vmax=max_val,
            caption=f"{feature_col.replace('_', ' ').title()}",
        )

        layer = folium.FeatureGroup(name=f"Population — {feature_col}", show=True)

        for _, row in gdf_merged.iterrows():
            if row.geometry is None:
                continue

            val = row.get("viz_val", 0)
            color = colormap(val)

            folium.GeoJson(
                data=row.geometry.__geo_interface__,
                style_function=lambda _, c=color: {
                    "fillColor":   c,
                    "fillOpacity": 0.65,
                    "color":       "white",
                    "weight":      0.3,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["h3_id"] if "h3_id" in row.index else [],
                    aliases=["Zone ID"],
                    sticky=False,
                ),
            ).add_to(layer)

        layer.add_to(m)
        colormap.add_to(m)
        log.info(f"Population choropleth added: {feature_col}")
        return m

    def add_facility_markers(
        self,
        m: folium.Map,
        facilities_gdf: gpd.GeoDataFrame,
        facility_type: str = "hospital",
        show: bool = True,
    ) -> folium.Map:
        """
        Add facility markers to map.

        Args:
            m:              Folium map
            facilities_gdf: GeoDataFrame of facility points
            facility_type:  Type for icon selection
            show:           Whether layer is visible by default

        Returns:
            Updated map
        """
        config = FACILITY_CONFIGS.get(facility_type, FACILITY_CONFIGS["hospital"])
        layer_name = f"Existing {facility_type.replace('_', ' ').title()}s"
        layer = folium.FeatureGroup(name=layer_name, show=show)

        for _, row in facilities_gdf.iterrows():
            if row.geometry is None:
                continue

            geom = row.geometry
            if geom.geom_type == "Point":
                lat, lon = geom.y, geom.x
            else:
                lat, lon = geom.centroid.y, geom.centroid.x

            name = row.get("name", f"Unknown {facility_type}")

            popup_html = f"""
            <div style='font-family: Arial; min-width: 200px'>
                <h4 style='color: {config["color"]}; margin: 0'>{config["emoji"]} {name}</h4>
                <p style='margin: 4px 0'><b>Type:</b> {facility_type.replace('_',' ').title()}</p>
                <p style='margin: 4px 0'><b>Location:</b> {lat:.4f}, {lon:.4f}</p>
            </div>"""

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{config['emoji']} {name}",
                icon=folium.Icon(
                    color=config["color"],
                    icon=config["icon"],
                    prefix=config["prefix"],
                ),
            ).add_to(layer)

        layer.add_to(m)
        log.info(f"Added {len(facilities_gdf)} {facility_type} markers")
        return m

    def add_recommendation_markers(
        self,
        m: folium.Map,
        ranked_df: pd.DataFrame,
        infra_type: str = "hospital",
        top_n: int = 10,
    ) -> folium.Map:
        """
        Add recommended new site markers with priority scores.

        Recommended sites use pulsing CircleMarkers with popup cards
        showing the full explanation.

        Args:
            m:          Folium map
            ranked_df:  DataFrame from BestFirstSearch results
            infra_type: Infrastructure type
            top_n:      Show top N recommendations

        Returns:
            Updated map
        """
        layer = folium.FeatureGroup(name=f"Recommended {infra_type.replace('_',' ').title()} Sites", show=True)
        top_df = ranked_df.head(top_n)

        for _, row in top_df.iterrows():
            lat = row.get("lat", 19.0)
            lon = row.get("lon", 72.85)
            rank = row.get("rank", "?")
            score = row.get("priority_score", 0)
            gap   = row.get("coverage_gap", 0)
            pop   = row.get("population_total", 0)
            cost  = row.get("estimated_cost", 0)

            # Color by priority
            if score >= 75:   color = "#E74C3C"
            elif score >= 50: color = "#F39C12"
            else:             color = "#27AE60"

            popup_html = f"""
            <div style='font-family: Arial; min-width: 280px; padding: 4px'>
                <h4 style='color: {color}; margin: 0 0 8px 0'>
                    ⭐ Rank #{rank:.0f} — {infra_type.replace('_',' ').title()} Recommendation
                </h4>
                <table style='width: 100%; border-collapse: collapse'>
                    <tr><td><b>Priority Score</b></td><td style='color:{color};font-weight:bold'>{score:.1f}/100</td></tr>
                    <tr><td><b>Coverage Gap</b></td><td>{gap*100:.1f}%</td></tr>
                    <tr><td><b>Population</b></td><td>{pop:,.0f}</td></tr>
                    <tr><td><b>Estimated Cost</b></td><td>₹{cost/1e7:.1f} Cr</td></tr>
                    <tr><td><b>Zone ID</b></td><td style='font-size:0.8em'>{row.get('h3_id','')[:20]}</td></tr>
                </table>
            </div>"""

            # Outer pulsing ring
            folium.CircleMarker(
                location=[lat, lon],
                radius=18,
                color=color,
                fill=False,
                weight=2,
                opacity=0.4,
            ).add_to(layer)

            # Inner marker
            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=2,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"⭐ Rank #{rank:.0f} | Score: {score:.0f}",
            ).add_to(layer)

            # Rank label
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-weight:bold;font-size:11px;color:white;text-align:center;margin-top:-4px">#{rank:.0f}</div>',
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                ),
            ).add_to(layer)

        layer.add_to(m)
        log.info(f"Added {len(top_df)} recommended site markers")
        return m

    def add_service_coverage_circles(
        self,
        m: folium.Map,
        facilities_gdf: gpd.GeoDataFrame,
        radius_km: float = 5.0,
        facility_type: str = "hospital",
    ) -> folium.Map:
        """
        Add service area circles around existing facilities.

        Args:
            m:              Folium map
            facilities_gdf: Facility points
            radius_km:      Service radius
            facility_type:  For color selection

        Returns:
            Updated map
        """
        config = FACILITY_CONFIGS.get(facility_type, FACILITY_CONFIGS["hospital"])
        layer = folium.FeatureGroup(name=f"{facility_type.title()} Coverage Areas", show=False)

        color_map = {"hospital": "#E74C3C", "school": "#3498DB", "ev_station": "#27AE60", "fire_station": "#F39C12"}
        circle_color = color_map.get(facility_type, "#9B59B6")

        for _, row in facilities_gdf.iterrows():
            if row.geometry is None:
                continue
            geom = row.geometry
            lat  = geom.y if geom.geom_type == "Point" else geom.centroid.y
            lon  = geom.x if geom.geom_type == "Point" else geom.centroid.x

            folium.Circle(
                location=[lat, lon],
                radius=radius_km * 1000,  # Folium uses meters
                color=circle_color,
                fill=True,
                fill_color=circle_color,
                fill_opacity=0.08,
                weight=1.5,
                opacity=0.5,
                tooltip=f"{config['emoji']} Service area: {radius_km} km radius",
            ).add_to(layer)

        layer.add_to(m)
        return m

    def add_priority_heatmap(
        self,
        m: folium.Map,
        feature_col: str = "priority_composite_100",
    ) -> folium.Map:
        """
        Add priority score heatmap layer.

        Args:
            m:           Folium map
            feature_col: Priority feature column

        Returns:
            Updated map
        """
        if "lat" not in self.df.columns or "lon" not in self.df.columns:
            if self.gdf is not None:
                centroids = self.gdf.geometry.centroid
                df_merged = self.df.copy()
                if len(centroids) == len(df_merged):
                    df_merged["lat"] = centroids.y.values
                    df_merged["lon"] = centroids.x.values
            else:
                log.warning("No lat/lon for heatmap")
                return m
        else:
            df_merged = self.df

        col = feature_col if feature_col in df_merged.columns else "priority_score"
        if col not in df_merged.columns:
            return m

        heat_data = [
            [row["lat"], row["lon"], float(row[col]) / 100.0]
            for _, row in df_merged.iterrows()
            if pd.notna(row.get("lat")) and pd.notna(row.get("lon"))
        ]

        fp.HeatMap(
            heat_data,
            name="Priority Score Heatmap",
            min_opacity=0.3,
            max_zoom=16,
            radius=20,
            blur=15,
            gradient={0.2: "#2ECC71", 0.5: "#F39C12", 0.8: "#E74C3C", 1.0: "#8E0000"},
            show=True,
        ).add_to(m)

        log.info("Priority heatmap added")
        return m

    def create_full_city_map(
        self,
        include_hospitals: bool = True,
        include_recommendations: bool = True,
        infra_type: str = "hospital",
    ) -> folium.Map:
        """
        Create complete interactive city map with all layers.

        Args:
            include_hospitals:       Show existing hospitals
            include_recommendations: Show recommended sites
            infra_type:              Infrastructure type for recommendations

        Returns:
            Complete Folium map
        """
        m = self.create_base_map("dark")

        # Population choropleth
        self.add_population_choropleth(m, "population_density")

        # Priority heatmap
        self.add_priority_heatmap(m)

        # Existing facilities
        if include_hospitals:
            for ftype in ["hospital", "school", "ev_station", "fire_station"]:
                for fname in [RAW_DIR / f"osm_{ftype}s.gpkg",
                               RAW_DIR / f"healthsites_IN.gpkg",
                               RAW_DIR / f"ev_stations_IN.gpkg"]:
                    if fname.exists():
                        try:
                            fgdf = gpd.read_file(str(fname))
                            self.add_facility_markers(m, fgdf, ftype)
                            self.add_service_coverage_circles(m, fgdf, facility_type=ftype)
                        except Exception:
                            pass
                        break

        # Recommended sites
        if include_recommendations:
            ranked_path = PROCESSED_DIR / f"ranked_sites_{infra_type}.csv"
            if ranked_path.exists():
                ranked_df = pd.read_csv(ranked_path)
                self.add_recommendation_markers(m, ranked_df, infra_type)

        folium.LayerControl(collapsed=False).add_to(m)
        return m


def save_map(m: folium.Map, filename: str = "city_map.html") -> Path:
    """Save Folium map to HTML file."""
    output_path = BASE_DIR / filename
    m.save(str(output_path))
    log.info(f"Map saved → {output_path}")
    return output_path


if __name__ == "__main__":
    # Quick demo
    csv = PROCESSED_DIR / "zone_features.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        viz = CityMapVisualizer(df)
        m = viz.create_full_city_map()
        save_map(m, "demo_city_map.html")
    else:
        print("Run create_zones.py first.")
