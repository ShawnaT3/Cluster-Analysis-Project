
import streamlit as st
import pandas as pd
import plotly.express as px
from geopy.distance import geodesic
import requests
import json
import numpy as np
from scipy.spatial import cKDTree
from functools import lru_cache

# Cache geocoding results
@lru_cache(maxsize=1024)
def geocode_address(address):
    # Format the address for Google Maps
    formatted_address = f"{address}, Ontario, Canada"
    # Use Google Maps Geocoding API
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={formatted_address}&key=AIzaSyBlgcjpOrH6FxIczUSRz4hD4tRV4Qt56Kg"
    try:
        response = requests.get(url)
        data = response.json()
        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            return {
                'latitude': location['lat'],
                'longitude': location['lng'],
                'address': data['results'][0]['formatted_address']
            }
        else:
            return None
    except Exception as e:
        st.sidebar.error(f"Error during geocoding: {str(e)}")
        return None

# Cache the distance calculation
@lru_cache(maxsize=1024)
def cached_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers
# Set page configuration
st.set_page_config(page_title="Location Density Tool", layout="wide")

# Title
st.title("Location Density Tool")

# Address search and geocoding logic at the top
st.subheader("Address Search")
address = st.text_input("Enter an address to search (include city and province):", key="address_input")
location_data = None
if address:
    location_data = geocode_address(address)
    if not location_data:
        st.error("Address not found. Please try a different address.")
        st.stop()

# Add file upload option
st.markdown("## Input Options")
upload_option = st.radio("Choose input method:", ["Single Address Search", "Upload Excel File"])

# Constants for allowed distances (in kilometers)
ALLOWED_DISTANCE_REGULAR = 0.01524  # 50 feet in kilometers
ALLOWED_DISTANCE_FARM = 0.024384    # 80 feet in kilometers



def create_base_map(residential_df, commercial_df, center_lat=43.5448, center_lon=-80.2482, zoom=7):
    """Create the base map with residential and commercial properties."""
    # Create the base map with residential properties
    fig = px.scatter_mapbox(residential_df, 
                          lat="RISK_LAT", 
                          lon="RISK_LNG", 
                          size="TIV",
                          color="TIV",
                          size_max=50,
                          center=dict(lat=center_lat, lon=center_lon),
                          zoom=zoom,
                          mapbox_style="open-street-map",
                          height=900,
                          color_continuous_scale=[
                              [0, "rgb(94, 201, 98)"],    # Light green
                              [0.25, "rgb(33, 145, 140)"],  # Teal
                              [0.5, "rgb(59, 82, 139)"],   # Blue
                              [0.75, "rgb(68, 1, 84)"],    # Dark purple
                              [1, "rgb(35, 0, 40)"]        # Darker purple
                          ],
                          opacity=0.9,
                          hover_data={
                              "Policy + Risk": True,
                              "TIV": True,
                              "Risk Location": True
                          })

    # Add density layers based on selection
    density_options = ["Residential", "Commercial"]
    selected_layers = st.sidebar.multiselect(
        "Select Density Layers",
        options=density_options,
        default=density_options
    )

    # Add residential density layer if selected
    if "Residential" in selected_layers:
        fig.add_densitymapbox(
            lat=residential_df['RISK_LAT'],
            lon=residential_df['RISK_LNG'],
            z=residential_df['TIV'],
            radius=30,
            colorscale=[
                [0, "rgba(94, 201, 98, 0)"],     # Transparent light green
                [0.2, "rgba(94, 201, 98, 0.2)"],  # Light green with low opacity
                [0.4, "rgba(33, 145, 140, 0.4)"], # Teal with medium opacity
                [0.6, "rgba(59, 82, 139, 0.6)"],  # Blue with higher opacity
                [0.8, "rgba(68, 1, 84, 0.8)"],    # Dark purple with high opacity
                [1, "rgba(35, 0, 40, 1)"]         # Darker purple with full opacity
            ],
            showscale=True,
            name="Residential Density",
            opacity=0.7,
            colorbar=dict(
                title="Residential Density",
                x=1.15,
                y=0.5,
                len=0.4,
                thickness=20,
                outlinewidth=1,
                outlinecolor='black'
            )
        )

    # Add commercial density layer if selected and available
    if "Commercial" in selected_layers and commercial_df is not None and not commercial_df.empty:
        fig.add_densitymapbox(
            lat=commercial_df['RISK_LAT'],
            lon=commercial_df['RISK_LNG'],
            z=commercial_df['TIV'],
            radius=30,
            colorscale=[
                [0, "rgba(255, 0, 255, 0)"],     # Transparent magenta
                [0.2, "rgba(255, 0, 255, 0.2)"],  # Magenta with low opacity
                [0.4, "rgba(200, 0, 200, 0.4)"],  # Darker magenta with medium opacity
                [0.6, "rgba(150, 0, 150, 0.6)"],  # Even darker magenta with higher opacity
                [0.8, "rgba(100, 0, 100, 0.8)"],  # Very dark magenta with high opacity
                [1, "rgba(50, 0, 50, 1)"]         # Darkest magenta with full opacity
            ],
            showscale=True,
            name="Commercial Density",
            opacity=0.7,
            colorbar=dict(
                title="Commercial Density",
                x=1.25,
                y=0.5,
                len=0.4,
                thickness=20,
                outlinewidth=1,
                outlinecolor='black'
            )
        )

    # Update scattermapbox properties
    fig.update_traces(
        marker=dict(
            sizemode='area',
            sizeref=2*max(residential_df['TIV'])/(40**2),
            sizemin=6
        ),
        hovertemplate="<b>Policy + Risk:</b> %{customdata[0]}<br>" +
                    "<b>TIV:</b> $%{customdata[1]:,.2f}<br>" +
                    "<b>Risk Location:</b> %{customdata[2]}<br>" +
                    "<extra></extra>",
        name="Residential Properties",
        showlegend=True,
        selector=dict(type='scattermapbox')
    )
    
    # Add commercial properties with a different color scale
    if commercial_df is not None and not commercial_df.empty:
        fig.add_scattermapbox(
            lat=commercial_df['RISK_LAT'],
            lon=commercial_df['RISK_LNG'],
            mode='markers',
            marker=dict(
                size=commercial_df['TIV'],
                sizemode='area',
                sizeref=2*max(commercial_df['TIV'])/(40**2),
                sizemin=6,
                color=commercial_df['TIV'],
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(
                    title="Commercial",
                    x=1.25,
                    y=0.5,
                    len=0.4,
                    thickness=20,
                    outlinewidth=1,
                    outlinecolor='black'
                )
            ),
            opacity=0.9,
            customdata=commercial_df[['Policy + Risk', 'TIV', 'Risk Location']],
            hovertemplate="<b>Policy + Risk:</b> %{customdata[0]}<br>" +
                        "<b>TIV:</b> $%{customdata[1]:,.2f}<br>" +
                        "<b>Risk Location:</b> %{customdata[2]}<br>" +
                        "<extra></extra>",
            name="Commercial Properties",
            showlegend=True
        )

    # Update layout
    fig.update_layout(
        mapbox=dict(
            zoom=zoom,
            center=dict(lat=center_lat, lon=center_lon)
        ),
        coloraxis_colorbar=dict(
            title="Residential",
            x=1.15,
            y=0.5,
            len=0.4,
            thickness=20,
            outlinewidth=1,
            outlinecolor='black'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.1,
            bgcolor="rgba(255, 255, 255, 0)",
            bordercolor="black",
            borderwidth=1
        ),
        margin=dict(r=200)
    )

    # Add density legend below the colorbars
    if "Residential" in selected_layers:
        fig.add_annotation(
            text="Density Legend",
            xref="paper",
            yref="paper",
            x=1.15,
            y=0.05,
            showarrow=False,
            font=dict(size=12, color="black")
        )
    
    if "Commercial" in selected_layers and commercial_df is not None and not commercial_df.empty:
        fig.add_annotation(
            text="Density Legend",
            xref="paper",
            yref="paper",
            x=1.25,
            y=0.05,
            showarrow=False,
            font=dict(size=12, color="black")
        )

    return fig

def add_uploaded_locations(fig, locations):
    """Add uploaded locations to the map."""
    if locations:
        uploaded_lats = [loc['latitude'] for loc in locations]
        uploaded_lons = [loc['longitude'] for loc in locations]
        uploaded_addresses = [loc['address'] for loc in locations]
        
        fig.add_scattermapbox(
            lat=uploaded_lats,
            lon=uploaded_lons,
            mode='markers',
            marker=dict(
                size=20,
                color='#B3D44E',  # HD Green
                opacity=0.8
            ),
            text=uploaded_addresses,
            name='Uploaded Locations',
            hovertemplate="<b>Uploaded Location:</b><br>%{text}<br><extra></extra>"
        )
    return fig

# Initialize dataframes as None
residential_df = None
commercial_df = None

try:
    # Read the Excel file
    df = pd.read_excel(r"C:\Users\sthomson\OneDrive - HD Mutual Insurance Company\Documents\Halwell TIVs.xlsx", sheet_name="Locations + TIV")
    # Rename columns to match expected names (except Policy + Risk)
    df = df.rename(columns={
        'TIV': 'TIV',
        'Risk Location': 'Risk Location',
        'Lat': 'RISK_LAT',
        'Long': 'RISK_LNG'
    })
    
    # Clean and validate the data
    df['RISK_LAT'] = pd.to_numeric(df['RISK_LAT'], errors='coerce')
    df['RISK_LNG'] = pd.to_numeric(df['RISK_LNG'], errors='coerce')
    df['TIV'] = pd.to_numeric(df['TIV'], errors='coerce')
    
    # Convert negative TIVs to positive for visualization
    df['TIV'] = df['TIV'].abs()
    
    # Filter out rows with invalid coordinates
    df = df[
        (df['RISK_LAT'].notna()) & 
        (df['RISK_LNG'].notna()) & 
        (df['RISK_LAT'] >= -90) & 
        (df['RISK_LAT'] <= 90) & 
        (df['RISK_LNG'] >= -180) & 
        (df['RISK_LNG'] <= 180) &
        (df['TIV'].notna())
    ].copy()
    
    # Pre-calculate farm status
    df['is_farm'] = df['Policy + Risk'].str.upper().str.contains('F')
    
    # Consolidate data before plotting
    df['is_commercial'] = df['Policy + Risk'].str.upper().str.contains('C')
    
    # Group by policy number and calculate sums
    agg_dict = {
        'TIV': 'sum',
        'RISK_LAT': 'first',
        'RISK_LNG': 'first',
        'Risk Location': 'first',
        'is_farm': 'first',
        'is_commercial': 'first'
    }
    
    consolidated_df = df.groupby('Policy + Risk').agg(agg_dict).reset_index()
    consolidated_df['TIV'] = consolidated_df['TIV'].abs()
    
    # Split into commercial and residential dataframes
    commercial_df = consolidated_df[consolidated_df['is_commercial']]
    residential_df = consolidated_df[~consolidated_df['is_commercial']]
    
    # Display data quality information
    st.sidebar.write("Data Quality Information:")
    st.sidebar.write(f"Total rows in dataset: {len(df)}")
    st.sidebar.write(f"Rows with valid coordinates: {len(df)}")
    st.sidebar.write("Data Consolidation Information:")
    st.sidebar.write(f"Total unique policies: {len(consolidated_df)}")
    st.sidebar.write(f"Commercial policies: {len(commercial_df)}")
    st.sidebar.write(f"Residential policies: {len(residential_df)}")
    
except Exception as e:
    st.error(f"Error reading the file: {str(e)}")
    st.error("Please check that your Excel file contains valid latitude and longitude values.")
    st.error("Latitude must be between -90 and 90, and longitude must be between -180 and 180.")

if upload_option == "Single Address Search":
    st.subheader("Address Search")
    location_data = None
    if not address:
        st.info("Input address to begin.")
    else:
        location_data = geocode_address(address)
        if not location_data:
            st.error("Address not found. Please try a different address.")
        else:
            # Filter policies within 10km radius
            search_radius_km = 10.0
            warning_radius_km = 0.1  # 100 meters
            
            # Filter residential policies within radius
            if residential_df is not None:
                residential_distances = [
                    cached_distance(
                        location_data['latitude'],
                        location_data['longitude'],
                        row['RISK_LAT'],
                        row['RISK_LNG']
                    )
                    for _, row in residential_df.iterrows()
                ]
                residential_df['Distance'] = residential_distances
                filtered_residential_df = residential_df[residential_df['Distance'] <= search_radius_km].copy()
                nearby_residential_df = residential_df[residential_df['Distance'] <= warning_radius_km].copy()
            else:
                filtered_residential_df = None
                nearby_residential_df = None

            # Filter commercial policies within radius
            if commercial_df is not None:
                commercial_distances = [
                    cached_distance(
                        location_data['latitude'],
                        location_data['longitude'],
                        row['RISK_LAT'],
                        row['RISK_LNG']
                    )
                    for _, row in commercial_df.iterrows()
                ]
                commercial_df['Distance'] = commercial_distances
                filtered_commercial_df = commercial_df[commercial_df['Distance'] <= search_radius_km].copy()
                nearby_commercial_df = commercial_df[commercial_df['Distance'] <= warning_radius_km].copy()
            else:
                filtered_commercial_df = None
                nearby_commercial_df = None

            # Check for nearby policies (within 100m)
            nearby_policies_count = (len(nearby_residential_df) if nearby_residential_df is not None else 0) + \
                                  (len(nearby_commercial_df) if nearby_commercial_df is not None else 0)
            
            if nearby_policies_count > 0:
                st.warning(f"⚠️ WARNING: Found {nearby_policies_count} policies within 100m of the searched location!")
                if nearby_residential_df is not None and not nearby_residential_df.empty:
                    st.write("Nearby Residential Policies:")
                    st.dataframe(nearby_residential_df[['Policy + Risk', 'Risk Location', 'Distance']].rename(
                        columns={'Policy + Risk': 'Policy Number', 'Risk Location': 'Address', 'Distance': 'Distance (km)'}
                    ))
                if nearby_commercial_df is not None and not nearby_commercial_df.empty:
                    st.write("Nearby Commercial Policies:")
                    st.dataframe(nearby_commercial_df[['Policy + Risk', 'Risk Location', 'Distance']].rename(
                        columns={'Policy + Risk': 'Policy Number', 'Risk Location': 'Address', 'Distance': 'Distance (km)'}
                    ))

            # Create the base map with filtered data
            fig = create_base_map(filtered_residential_df, filtered_commercial_df, 
                                center_lat=location_data['latitude'], 
                                center_lon=location_data['longitude'],
                                zoom=12)  # Increased zoom for better visibility of nearby policies
            
            # Add the searched location marker
            fig.add_scattermapbox(
                lat=[location_data['latitude']],
                lon=[location_data['longitude']],
                mode='markers',
                marker=dict(
                    size=20,
                    color='#B57EDC',  # Lavender
                    opacity=1,
                    symbol='circle'
                ),
                text=['Searched Location'],
                name='Searched Location'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display summary of nearby policies
            total_policies = len(filtered_residential_df) if filtered_residential_df is not None else 0
            total_policies += len(filtered_commercial_df) if filtered_commercial_df is not None else 0
            
            st.write(f"### Policies within {search_radius_km}km radius")
            st.write(f"Total policies found: {total_policies}")
            
            # Add distance slider for detailed analysis
            st.write("### Detailed Analysis")
            
            # Add unit toggle
            use_imperial = st.toggle("Use Imperial Units (feet)", value=True)
            
            if use_imperial:
                min_value_ft = 50
                max_value_ft = 6562
                step_ft = 50
                
                distance_ft = st.slider(
                    "Select search radius",
                    min_value=min_value_ft,
                    max_value=max_value_ft,
                    value=min_value_ft,
                    step=step_ft,
                    format="%d ft"
                )
                distance_km = distance_ft / 3280.84
            else:
                min_value_km = 0.01524
                max_value_km = 2.0
                step_km = 0.01524
                
                distance_km = st.slider(
                    "Select search radius",
                    min_value=min_value_km,
                    max_value=max_value_km,
                    value=min_value_km,
                    step=step_km,
                    format="%.3f km"
                )
            
            # Calculate nearby policies
            if location_data:
                all_policies = pd.concat([
                    residential_df.assign(Property_Type='Residential'),
                    commercial_df.assign(Property_Type='Commercial')
                ]) if commercial_df is not None else residential_df.assign(Property_Type='Residential')
                
                distances = []
                for _, row in all_policies.iterrows():
                    dist = cached_distance(
                        location_data['latitude'],
                        location_data['longitude'],
                        row['RISK_LAT'],
                        row['RISK_LNG']
                    )
                    distances.append(dist)
                
                all_policies['Distance'] = distances
                nearby_policies = all_policies[all_policies['Distance'] <= distance_km].copy()
                nearby_policies = nearby_policies.sort_values('Distance')
                
                if not nearby_policies.empty:
                    nearby_policies['Distance (ft)'] = (nearby_policies['Distance'] * 3280.84).round(0).astype(int)
                    
                    display_df = nearby_policies[[
                        'Property_Type',
                        'Policy + Risk',
                        'Risk Location',
                        'TIV',
                        'Distance (ft)'
                    ]].rename(columns={
                        'Property_Type': 'Property Type',
                        'Policy + Risk': 'Policy Number',
                        'Risk Location': 'Address',
                        'TIV': 'TIV'
                    })
                    
                    display_df['TIV'] = display_df['TIV'].apply(lambda x: f"${x:,.2f}")
                    
                    # Calculate summary statistics
                    total_tiv = nearby_policies['TIV'].sum()
                    residential_policies = nearby_policies[nearby_policies['Property_Type'] == 'Residential']
                    commercial_policies = nearby_policies[nearby_policies['Property_Type'] == 'Commercial']
                    
                    residential_count = len(residential_policies)
                    commercial_count = len(commercial_policies)
                    residential_tiv = residential_policies['TIV'].sum()
                    commercial_tiv = commercial_policies['TIV'].sum()
                    
                    # Create three columns for the summary metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Policies", f"{len(nearby_policies):,}")
                        st.metric("Total TIV", f"${total_tiv:,.2f}")
                    
                    with col2:
                        st.metric("Residential Policies", f"{residential_count:,}")
                        st.metric("Residential TIV", f"${residential_tiv:,.2f}")
                    
                    with col3:
                        st.metric("Commercial Policies", f"{commercial_count:,}")
                        st.metric("Commercial TIV", f"${commercial_tiv:,.2f}")
                    
                    st.divider()
                    
                    st.dataframe(
                        display_df,
                        column_config={
                            "Property Type": st.column_config.TextColumn(
                                "Property Type",
                                width="medium"
                            ),
                            "Policy Number": st.column_config.TextColumn(
                                "Policy Number",
                                width="medium"
                            ),
                            "Address": st.column_config.TextColumn(
                                "Address",
                                width="large"
                            ),
                            "TIV": st.column_config.TextColumn(
                                "TIV",
                                width="medium"
                            ),
                            "Distance (ft)": st.column_config.NumberColumn(
                                "Distance (ft)",
                                width="small",
                                format="%d"
                            )
                        },
                        hide_index=True
                    )
                else:
                    st.write(f"No policies found within {distance_km*3280.84:.0f} feet")
            
            # Debug information
            st.sidebar.write("Debug Info:")
            st.sidebar.write(f"Found location: {location_data['address']}")
            st.sidebar.write(f"Coordinates: {location_data['latitude']}, {location_data['longitude']}")
else:
    # File upload functionality
    st.sidebar.title("Upload File")
    uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV file with addresses", type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file is not None:
        try:
            # Read the file based on its type
            if uploaded_file.name.endswith('.csv'):
                df_uploaded = pd.read_csv(uploaded_file)
            else:
                df_uploaded = pd.read_excel(uploaded_file)
            
            # Validate required columns
            required_columns = ['Address']
            if not all(col in df_uploaded.columns for col in required_columns):
                st.error("File must contain an 'Address' column")
            else:
                # Geocode all addresses
                st.write("Geocoding addresses...")
                locations = []
                for address in df_uploaded['Address']:
                    location = geocode_address(address)
                    if location:
                        locations.append(location)
                    else:
                        st.warning(f"Could not geocode address: {address}")
                
                if locations and residential_df is not None:
                    # Create the base map
                    fig = create_base_map(residential_df, commercial_df)
                    
                    # Add uploaded locations
                    fig = add_uploaded_locations(fig, locations)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display summary of uploaded locations
                    st.write("### Uploaded Locations Summary")
                    st.write(f"Total addresses in file: {len(df_uploaded)}")
                    st.write(f"Successfully geocoded: {len(locations)}")
                    st.write(f"Failed to geocode: {len(df_uploaded) - len(locations)}")
                    
                    # Display the uploaded locations in a table
                    if locations:
                        uploaded_df = pd.DataFrame([
                            {
                                'Address': loc['address'],
                                'Latitude': loc['latitude'],
                                'Longitude': loc['longitude']
                            }
                            for loc in locations
                        ])
                        
                        st.dataframe(
                            uploaded_df,
                            column_config={
                                "Address": st.column_config.TextColumn(
                                    "Address",
                                    width="large"
                                ),
                                "Latitude": st.column_config.NumberColumn(
                                    "Latitude",
                                    width="medium",
                                    format="%.6f"
                                ),
                                "Longitude": st.column_config.NumberColumn(
                                    "Longitude",
                                    width="medium",
                                    format="%.6f"
                                )
                            },
                            hide_index=True
                        )
                else:
                    st.error("No valid locations found in the uploaded file")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Please ensure the file contains the required columns and valid data.")
 