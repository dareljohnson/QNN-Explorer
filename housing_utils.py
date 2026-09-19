#!/usr/bin/env python
"""
Housing prediction utility functions.

This module provides functions for rendering housing price prediction UI components
separate from the main app to allow for better testability.
"""

import streamlit as st

def render_housing_price_form():
    """Render the housing price prediction form for testing purposes."""
    # Lists of US states
    states = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
    ]
    
    with st.form("housing_prediction_form_test"):
        st.write("Enter house details to predict the price:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sqft = st.number_input("Square Footage", min_value=500, max_value=10000, value=2000)
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
            bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=7.0, value=2.0, step=0.5)
        
        with col2:
            state = st.selectbox("State", states, index=states.index('CA'))
            year_built = st.number_input("Year Built", min_value=1900, max_value=2023, value=1990)
            condition = st.slider("Condition (1-5)", min_value=1, max_value=5, value=3)
        
        with col3:
            lot_size = st.number_input("Lot Size (sqft)", min_value=1000, max_value=100000, value=10000)
            waterfront = st.checkbox("Waterfront Property", value=False)
            floors = st.selectbox("Number of Floors", [1, 1.5, 2, 2.5, 3], index=1)
        
        # Use checkbox for optional features instead of expander
        show_optional = st.checkbox("Show Optional Features", value=False)
        if show_optional:
            st.write("**Optional Features:**")
            col1, col2 = st.columns(2)
            with col1:
                view = st.slider("View Quality (0-4)", min_value=0, max_value=4, value=2)
                grade = st.slider("Grade (1-13)", min_value=1, max_value=13, value=7)
            with col2:
                yr_renovated = st.number_input("Year Renovated (0=None)", min_value=0, max_value=2023, value=0)
                basement = st.checkbox("Has Basement", value=True)
        else:
            # Default values when optional features are hidden
            view = 2
            grade = 7
            yr_renovated = 0
            basement = True
        
        # Calculate derivative fields
        if basement:
            sqft_basement = 200  # Default value
            sqft_above = sqft - sqft_basement
        else:
            sqft_basement = 0
            sqft_above = sqft
        
        # Use checkbox for geolocation instead of expander
        show_geolocation = st.checkbox("Show Advanced Geolocation", value=False)
        if show_geolocation:
            st.write("**Advanced Geolocation:**")
            lat = st.number_input("Latitude", min_value=24.0, max_value=50.0, value=47.5)
            long = st.number_input("Longitude", min_value=-125.0, max_value=-66.0, value=-122.3)
        else:
            # Default values when geolocation is hidden
            lat = 47.5
            long = -122.3
        
        submitted = st.form_submit_button("Predict Price")
        
        return submitted

def render_prediction_details(price, state, sqft, bedrooms, bathrooms, year_built, condition, 
                             multiplier, age_factor, condition_factor, waterfront):
    """Render the prediction details for testing purposes."""
    st.success(f"# ${int(price):,}")
    st.write(f"Estimated house price in {state}")
    
    # Show details with checkbox instead of expander
    show_details = st.checkbox("Show Prediction Details", value=True)
    if show_details:
        st.write("**House Features:**")
        
        # Show key features in a nicer format
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Square Feet", f"{sqft:,}")
            st.metric("Bedrooms", bedrooms)
        with col2:
            st.metric("Bathrooms", bathrooms)
            st.metric("Year Built", year_built)
        with col3:
            st.metric("State", state)
            st.metric("Condition", f"{condition}/5")
        
        # Explanation of factors affecting the price
        st.write("**Factors Affecting Price:**")
        st.write(f"- **Location:** {state} {'(premium market)' if multiplier > 1 else '(affordable market)' if multiplier < 1 else ''}")
        st.write(f"- **Size:** {sqft:,} sq ft (larger homes are more expensive)")
        st.write(f"- **Age:** {2023-year_built} years old ({age_factor:.2f}x multiplier)")
        st.write(f"- **Condition:** {condition}/5 ({condition_factor:.2f}x multiplier)")
        st.write(f"- **Waterfront:** {'Yes (+50%)' if waterfront else 'No'}")
        
        # Add a disclaimer
        st.info("This is a demonstration model using synthetic data. Real estate prices vary significantly based on many factors beyond those considered here.") 