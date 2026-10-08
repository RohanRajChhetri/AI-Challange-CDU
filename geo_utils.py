"""Geospatial utilities, Pydeck 3D layers, and Folium map builder for Northern Territory housing dispatch."""
from __future__ import annotations

import re
import folium
import pandas as pd
import pydeck as pdk

# Depot location: Main NT Housing Fleet Depot (Darwin)
DARWIN_DEPOT = {
    "name": "Darwin Central Fleet Depot",
    "location": "Darwin (Depot)",
    "lat": -12.4450,
    "lon": 130.8500, 
}


NT_COMMUNITY_COORDS = {
    "darwin": (-12.4580, 130.8430),       
    "darwin city": (-12.4580, 130.8430),
    "casuarina": (-12.3735, 130.8800),       
    "nightcliff": (-12.3780, 130.8600),     
    "berrimah": (-12.4333, 130.9333),         
    "palmerston": (-12.4859, 130.9833),       
    "batchelor": (-13.0667, 131.0167),
    "adelaide river": (-13.2403, 131.1075),
    "pine creek": (-13.8236, 131.8264),
    "katherine": (-14.4652, 132.2635),
    "jabiru": (-12.6711, 132.8364),
    "wadeye": (-14.2380, 129.5260),          
    "port keats": (-14.2380, 129.5260),
    "maningrida": (-12.0575, 134.2347),      
    "tennant creek": (-19.6481, 134.1906),
    "alice springs": (-23.6980, 133.8807),
    "yuendumu": (-22.2536, 131.7944),
    "papunya": (-23.2056, 131.9056),
    "kintore": (-23.2847, 128.3078),
    "nhulunbuy": (-12.1825, 136.7800),
    "yirrkala": (-12.2533, 136.8867),
    "groote eylandt": (-13.8447, 136.4192),   
    "alyangula": (-13.8447, 136.4192),
    "daly river": (-13.7547, 130.7078),
    "nauiyu": (-13.7547, 130.7078),
    "gunbalanya": (-12.3278, 133.0500),
    "oenpelli": (-12.3278, 133.0500),
    "ramingining": (-12.3561, 134.9083),
    "galiwinku": (-12.0236, 135.5683),
    "elcho island": (-12.0236, 135.5683),
    "milingimbi": (-12.0964, 134.8967),
    "wurrumiyanga": (-11.7583, 130.6306),
    "tiwi islands": (-11.7583, 130.6306),
    "ngukurr": (-14.7333, 134.7333),
    "borroloola": (-16.0711, 136.3072),
    "lajamanu": (-18.3333, 130.6333),
    "kalkarindji": (-17.4333, 130.8333),
    "hermannsburg": (-23.9528, 132.7778),
}


def clean_location_name(loc: str) -> str:
    """Normalize location string, stripping suffixes like (Urban) or (Remote)."""
    if not isinstance(loc, str):
        return "darwin"
    cleaned = re.sub(r"\s*\((urban|remote)\)", "", loc, flags=re.IGNORECASE).strip().lower()
    return cleaned


def get_coords(location: str) -> tuple[float, float]:
    """Retrieve latitude and longitude for a Northern Territory community."""
    cleaned = clean_location_name(location)
    if cleaned in NT_COMMUNITY_COORDS:
        return NT_COMMUNITY_COORDS[cleaned]
    # Fuzzy match prefix
    for key, coords in NT_COMMUNITY_COORDS.items():
        if key in cleaned or cleaned in key:
            return coords
    # Default to Darwin depot
    return NT_COMMUNITY_COORDS["darwin"]


def wait_time_color(days: float) -> list[int]:
    """Color-code node by waiting duration for Deck.gl:
    - < 14 days: Mint Green (Fresh)
    - 14 - 45 days: Amber (Ageing)
    - > 45 days: Glowing Crimson (Breached Guardrail / SLA Overdue)
    """
    if days > 45:
        return [235, 59, 90, 240]    # Glowing Crimson
    elif days >= 14:
        return [250, 130, 49, 230]   # Amber / Orange
    else:
        return [46, 204, 113, 210]   # Mint Green


def build_geospatial_dataframe(scored_df: pd.DataFrame, jobs_per_week: int) -> pd.DataFrame:
    """Enrich scored dataframe with coordinates, route membership, and styling."""
    df = scored_df.copy()
    
    coords = df["location"].apply(get_coords)
    df["lat"] = [c[0] for c in coords]
    df["lon"] = [c[1] for c in coords]
    
    # Safe street-level residential micro-offsets (~80-120m lot separation)
    # For coastal communities (Darwin, Nightcliff, Wadeye), step strictly INLAND so pins NEVER enter water
    INLAND_VECTORS = {
        "darwin": (0.0010, 0.0008),      # Step northeast towards Stuart Park
        "darwin city": (0.0010, 0.0008),
        "nightcliff": (0.0008, 0.0012),   # Step east/northeast towards Millner
        "wadeye": (0.0006, 0.0012),       # Step east inland away from coast
        "port keats": (0.0006, 0.0012),
        "maningrida": (-0.0008, 0.0012),  # Step southeast inland
    }

    counts = {}
    for idx, row in df.iterrows():
        key = (row["lat"], row["lon"])
        count = counts.get(key, 0)
        counts[key] = count + 1
        if count > 0:
            clean_loc = clean_location_name(row["location"])
            if clean_loc in INLAND_VECTORS:
                v_lat, v_lon = INLAND_VECTORS[clean_loc]
                df.at[idx, "lat"] = row["lat"] + v_lat * count
                df.at[idx, "lon"] = row["lon"] + v_lon * count
            else:
                # Standard inland community micro-offset (~100m street dispersion)
                d_lat = 0.0010 * count * (1 if count % 2 == 1 else -1)
                d_lon = 0.0010 * count * (1 if count % 3 == 0 else -1)
                df.at[idx, "lat"] = row["lat"] + d_lat
                df.at[idx, "lon"] = row["lon"] + d_lon

    df["color"] = df["days_waiting"].apply(wait_time_color)
    df["radius"] = df["days_waiting"].apply(lambda d: max(14000, min(45000, int(15000 + d * 500))))
    
    # Safe string fields for Pydeck tooltip (Pydeck JS template renderer requires plain keys without format specifiers)
    df["distance_km_str"] = df["distance_km"].apply(lambda d: f"{d:.0f} km")
    df["safety_prob_str"] = df["safety_prob"].apply(lambda s: f"{s:.0%}")
    df["days_waiting_str"] = df["days_waiting"].apply(lambda w: f"{int(w)} days")
    df["urgency_str"] = df["base_urgency"].apply(lambda u: f"{u:.1f}")
    df["clean_description"] = (
        df["text"]
        .astype(str)
        .str.replace('"', '&quot;')
        .str.replace("'", "&#39;")
        .str.slice(0, 140)
    )

    # Route membership flags
    df["in_equity_route"] = df["rank"] <= jobs_per_week
    df["in_efficiency_route"] = df["rank_efficiency"] <= jobs_per_week
    
    return df


def generate_route_links(geo_df: pd.DataFrame, route_type: str = "both") -> pd.DataFrame:
    """Generate link records connecting Darwin Depot to dispatched communities."""
    links = []
    depot_lat, depot_lon = DARWIN_DEPOT["lat"], DARWIN_DEPOT["lon"]

    if route_type in ("efficiency", "both"):
        eff_jobs = geo_df[geo_df["in_efficiency_route"]].copy()
        for _, row in eff_jobs.iterrows():
            links.append({
                "route_type": "Efficiency-First (Cheapest)",
                "source_name": DARWIN_DEPOT["name"],
                "source_lat": depot_lat,
                "source_lon": depot_lon,
                "target_name": row["location"],
                "target_lat": row["lat"],
                "target_lon": row["lon"],
                "ticket_id": row["ticket_id"],
                "location": row["location"],
                "distance_km": row["distance_km"],
                "distance_km_str": row.get("distance_km_str", f"{row['distance_km']:.0f} km"),
                "category": row["category"].title(),
                "days_waiting_str": row.get("days_waiting_str", f"{int(row['days_waiting'])} days"),
                "safety_prob_str": row.get("safety_prob_str", "N/A"),
                "urgency_str": row.get("urgency_str", "N/A"),
                "rank": row["rank"],
                "rank_shift": row["rank_shift"],
                "rank_efficiency": row["rank_efficiency"],
                "clean_description": f"Cheapest trip route connecting Darwin Depot to {row['location']}",
                "color": [0, 180, 255, 200],  # Cyan / Blue
                "width": 3.5,
            })

    if route_type in ("equity", "both"):
        eq_jobs = geo_df[geo_df["in_equity_route"]].copy()
        for _, row in eq_jobs.iterrows():
            links.append({
                "route_type": "EquiTriage (Fairness-Adjusted)",
                "source_name": DARWIN_DEPOT["name"],
                "source_lat": depot_lat,
                "source_lon": depot_lon,
                "target_name": row["location"],
                "target_lat": row["lat"],
                "target_lon": row["lon"],
                "ticket_id": row["ticket_id"],
                "location": row["location"],
                "distance_km": row["distance_km"],
                "distance_km_str": row.get("distance_km_str", f"{row['distance_km']:.0f} km"),
                "category": row["category"].title(),
                "days_waiting_str": row.get("days_waiting_str", f"{int(row['days_waiting'])} days"),
                "safety_prob_str": row.get("safety_prob_str", "N/A"),
                "urgency_str": row.get("urgency_str", "N/A"),
                "rank": row["rank"],
                "rank_shift": row["rank_shift"],
                "rank_efficiency": row["rank_efficiency"],
                "clean_description": f"Fairness-adjusted dispatch corridor reaching {row['location']}",
                "color": [255, 191, 0, 230],  # Glowing Gold / Amber
                "width": 4.5,
            })

    return pd.DataFrame(links)


def build_pydeck_chart(
    geo_df: pd.DataFrame,
    jobs_per_week: int,
    route_view: str = "both",
    use_arcs: bool = True
) -> pdk.Deck:
    """Construct an interactive Pydeck Deck visualization using free Carto Dark GL tiles."""
    layers = []

    # 1. Depot Hub Marker with tooltip fallback attributes
    depot_record = dict(DARWIN_DEPOT)
    depot_record.update({
        "distance_km_str": "0 km",
        "ticket_id": "DEPOT-HQ",
        "category": "Central Fleet Base",
        "days_waiting_str": "Headquarters",
        "safety_prob_str": "Active",
        "urgency_str": "HQ",
        "rank": "0",
        "rank_shift": "–",
        "rank_efficiency": "0",
        "clean_description": "Darwin Central Public Housing Fleet Dispatch Hub",
    })
    depot_df = pd.DataFrame([depot_record])
    depot_layer = pdk.Layer(
        "ScatterplotLayer",
        data=depot_df,
        get_position=["lon", "lat"],
        get_color=[255, 255, 255, 255],
        get_line_color=[255, 191, 0, 255],
        line_width_min_pixels=3,
        stroked=True,
        get_radius=22000,
        pickable=True,
        auto_highlight=True,
    )
    layers.append(depot_layer)

    # 2. Dispatch Route Arcs / Lines
    links_df = generate_route_links(geo_df, route_type=route_view)
    if not links_df.empty:
        if use_arcs:
            route_layer = pdk.Layer(
                "ArcLayer",
                data=links_df,
                get_source_position=["source_lon", "source_lat"],
                get_target_position=["target_lon", "target_lat"],
                get_source_color="color",
                get_target_color="color",
                get_width="width",
                pickable=True,
                auto_highlight=True,
            )
        else:
            route_layer = pdk.Layer(
                "LineLayer",
                data=links_df,
                get_source_position=["source_lon", "source_lat"],
                get_target_position=["target_lon", "target_lat"],
                get_color="color",
                get_width="width",
                pickable=True,
                auto_highlight=True,
            )
        layers.append(route_layer)

    # 3. Community Ticket Nodes
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=geo_df,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius="radius",
        stroked=True,
        get_line_color=[20, 20, 20, 200],
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
    )
    layers.append(scatter_layer)

    # Central NT ViewState
    view_state = pdk.ViewState(
        latitude=-16.5,
        longitude=133.5,
        zoom=4.8,
        pitch=35 if use_arcs else 0,
        bearing=0,
    )

    tooltip = {
        "html": """
        <div style="font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 11px; line-height: 1.4; color: #FFF; width: 100%; box-sizing: border-box;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; gap: 12px; border-bottom: 1px solid rgba(255, 191, 0, 0.5); padding-bottom: 6px; margin-bottom: 8px;">
                <span style="color: #FFBF00; font-weight: 700; font-size: 13px; letter-spacing: 0.2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">📍 {location}</span>
                <span style="color: #9CA3AF; font-size: 11px; font-weight: 500; white-space: nowrap; margin-right: 2px;">{distance_km_str}</span>
            </div>
            
            <div style="background: rgba(255, 191, 0, 0.08); border-left: 3px solid #FFBF00; padding: 6px 10px; margin-bottom: 8px; font-size: 11px; color: #E5E7EB; line-height: 1.4; border-radius: 0 4px 4px 0; box-sizing: border-box;">
                <b style="color: #FFBF00; font-size: 10px; letter-spacing: 0.5px;">DESCRIPTION:</b><br/>{clean_description}
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px 10px; font-size: 11px; color: #9CA3AF;">
                <div style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"><b style="color: #D1D5DB;">Ticket:</b> <code style="color: #F3F4F6; background: rgba(255,255,255,0.06); padding: 1px 4px; border-radius: 2px; font-family: monospace;">{ticket_id}</code></div>
                <div style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"><b style="color: #D1D5DB;">Issue:</b> <span style="color: #F3F4F6;">{category}</span></div>
                <div style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"><b style="color: #D1D5DB;">Wait:</b> <span style="color: #FFBF00; font-weight: 600;">{days_waiting_str}</span></div>
                <div style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"><b style="color: #D1D5DB;">Safety:</b> <span style="color: #F3F4F6;">{safety_prob_str}</span></div>
                <div style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"><b style="color: #D1D5DB;">Policy Rank:</b> <b style="color: #FFBF00;">#{rank}</b></div>
                <div style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"><b style="color: #D1D5DB;">Cheapest Rank:</b> <span style="color: #F3F4F6;">#{rank_efficiency}</span></div>
            </div>
        </div>
        """,
        "style": {
            "top": "100%",
            "backgroundColor": "#121212",
            "color": "#FFFFFF",
            "fontFamily": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            "fontSize": "11px",
            "padding": "10px 14px",
            "borderRadius": "6px",
            "border": "1px solid #FFBF00",
            "boxShadow": "0 8px 24px rgba(0, 0, 0, 0.95)",
            "zIndex": "99999999",
            "pointerEvents": "none",
            "boxSizing": "border-box",
            "width": "320px",
            "maxWidth": "320px",
            "wordWrap": "break-word",
        }
    }

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="dark",
        tooltip=tooltip,
    )


def build_folium_map(
    geo_df: pd.DataFrame,
    jobs_per_week: int,
    route_view: str = "both"
) -> folium.Map:
    """Build a rich, interactive geographical OpenStreetMap of the Northern Territory."""
    m = folium.Map(
        location=[-16.5, 133.5],
        zoom_start=5,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    depot_lat, depot_lon = DARWIN_DEPOT["lat"], DARWIN_DEPOT["lon"]

    # 1. Depot Headquarters Marker
    folium.Marker(
        location=[depot_lat, depot_lon],
        popup=folium.Popup(f"<b>{DARWIN_DEPOT['name']}</b><br/>Central Housing Dispatch Hub", max_width=250),
        tooltip="Darwin Fleet Depot",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(m)

    # 2. Route Corridors
    depot_pt = [depot_lat, depot_lon]

    if route_view in ("efficiency", "both"):
        eff_jobs = geo_df[geo_df["in_efficiency_route"]]
        for _, row in eff_jobs.iterrows():
            folium.PolyLine(
                locations=[depot_pt, [row["lat"], row["lon"]]],
                color="#00B4D8",
                weight=3.5,
                opacity=0.85,
                tooltip=f"Efficiency Run: {row['location']} (#{int(row['rank_efficiency'])})",
            ).add_to(m)

    if route_view in ("equity", "both"):
        eq_jobs = geo_df[geo_df["in_equity_route"]]
        for _, row in eq_jobs.iterrows():
            folium.PolyLine(
                locations=[depot_pt, [row["lat"], row["lon"]]],
                color="#FFBF00",
                weight=4.5,
                opacity=0.95,
                tooltip=f"EquiTriage Run: {row['location']} (#{int(row['rank'])})",
            ).add_to(m)

    # 3. Community Nodes with Wait-Time Styling
    for _, row in geo_df.iterrows():
        days = row["days_waiting"]
        if days > 45:
            color = "#E74C3C"       # Red
            label = "Breached (>45d)"
        elif days >= 14:
            color = "#F39C12"       # Amber
            label = "Ageing (14-45d)"
        else:
            color = "#2ECC71"       # Green
            label = "Fresh (<14d)"

        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 12px; width: 220px; line-height: 1.4;">
            <h4 style="margin: 0 0 5px 0; color: #1A1A1A; border-bottom: 2px solid {color}; padding-bottom: 2px;">
                {row['location']} ({row['distance_km']:.0f} km)
            </h4>
            <b>Ticket:</b> <code>{row['ticket_id']}</code><br/>
            <b>Category:</b> {row['category'].title()}<br/>
            <b>Days Waiting:</b> <b>{int(days)} days</b> ({label})<br/>
            <b>Safety Risk:</b> {row['safety_prob']:.0%}<br/>
            <b>EquiTriage Rank:</b> #{int(row['rank'])} (Shift: {row['rank_shift']})<br/>
            <b>Efficiency Rank:</b> #{int(row['rank_efficiency'])}<br/>
            <b>Manifest Status:</b> <span style="color: #2980B9;">{row['bundle_status']}</span>
        </div>
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=8 + min(10, int(days / 6)),
            color="#111111",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['location']} | {row['ticket_id']} ({int(days)}d)",
        ).add_to(m)

    return m
