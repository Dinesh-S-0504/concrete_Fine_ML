import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# Page configuration
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide"
)

# File to store user credentials
USER_FILE = "users.json"

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False


# Password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# User management functions
def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_users(users):
    with open(USER_FILE, 'w') as f:
        json.dump(users, f)


def register_user(username, password):
    users = load_users()
    if username in users:
        return False, "Username already exists"
    users[username] = hash_password(password)
    save_users(users)
    return True, "Registration successful"


def authenticate_user(username, password):
    users = load_users()
    if username in users and users[username] == hash_password(password):
        return True
    return False


# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('concrete_data.csv')
    return df


# Train accurate model with ensemble approach
@st.cache_resource
def get_accurate_model():
    df = load_data()
    X = df.drop('concrete_compressive_strength', axis=1)
    y = df['concrete_compressive_strength']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Create ensemble of models for better accuracy
    from sklearn.ensemble import VotingRegressor, RandomForestRegressor
    from xgboost import XGBRegressor

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train multiple models
    models = {
        'xgb': XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            subsample=0.8,
            colsample_bytree=0.8
        ),
        'gbr': GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            subsample=0.8
        ),
        'rf': RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
    }

    # Train individual models
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

    # Create voting regressor for ensemble
    ensemble_model = VotingRegressor([
        ('xgb', trained_models['xgb']),
        ('gbr', trained_models['gbr']),
        ('rf', trained_models['rf'])
    ])
    ensemble_model.fit(X_train_scaled, y_train)

    # Calculate accuracy metrics for reference
    y_pred = ensemble_model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        'ensemble': ensemble_model,
        'xgb': trained_models['xgb'],
        'gbr': trained_models['gbr'],
        'rf': trained_models['rf'],
        'scaler': scaler,
        'mae': mae,
        'r2': r2
    }


def auth_page():
    st.title("🔐 Concrete Strength Prediction System")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.session_state.show_signup:
            # Signup Form
            st.subheader("Create Account")
            new_username = st.text_input("Choose Username", key="signup_user")
            new_password = st.text_input("Choose Password", type="password", key="signup_pass")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Sign Up", use_container_width=True):
                    if not new_username or not new_password:
                        st.error("Please fill in all fields")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        success, message = register_user(new_username, new_password)
                        if success:
                            st.success(message)
                            st.session_state.show_signup = False
                            st.rerun()
                        else:
                            st.error(message)

            with col_b:
                if st.button("Back to Login", use_container_width=True):
                    st.session_state.show_signup = False
                    st.rerun()
        else:
            # Login Form
            st.subheader("Login")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")

            if st.button("Login", use_container_width=True):
                if authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

            st.markdown("---")
            if st.button("Create New Account", use_container_width=True):
                st.session_state.show_signup = True
                st.rerun()

            st.info("**Demo Account:**\nUsername: demo\nPassword: demo123")


def prediction_page():
    st.title("🏗️ Concrete Strength Prediction")
    st.markdown("---")

    # Load model data
    model_data = get_accurate_model()
    ensemble_model = model_data['ensemble']
    scaler = model_data['scaler']

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Parameters")

        # Create two columns for better layout
        col1a, col1b = st.columns(2)

        with col1a:
            cement = st.number_input("Cement (kg/m³)", min_value=0.0, max_value=600.0,
                                     value=281.0, step=1.0, format="%.1f")
            blast_furnace_slag = st.number_input("Blast Furnace Slag (kg/m³)",
                                                 min_value=0.0, max_value=400.0,
                                                 value=73.5, step=1.0, format="%.1f")
            fly_ash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, max_value=300.0,
                                      value=54.2, step=1.0, format="%.1f")
            water = st.number_input("Water (kg/m³)", min_value=120.0, max_value=250.0,
                                    value=181.6, step=0.5, format="%.1f")

        with col1b:
            superplasticizer = st.number_input("Superplasticizer (kg/m³)",
                                               min_value=0.0, max_value=30.0,
                                               value=6.2, step=0.1, format="%.1f")
            coarse_aggregate = st.number_input("Coarse Aggregate (kg/m³)",
                                               min_value=800.0, max_value=1200.0,
                                               value=972.9, step=1.0, format="%.1f")
            fine_aggregate = st.number_input("Fine Aggregate (kg/m³)",
                                             min_value=600.0, max_value=1000.0,
                                             value=773.6, step=1.0, format="%.1f")
            age = st.number_input("Age (days)", min_value=1, max_value=365,
                                  value=28, step=1)

        # Add presets based on dataset analysis
        st.markdown("### Quick Presets")
        preset_col1, preset_col2, preset_col3 = st.columns(3)

        with preset_col1:
            if st.button("High Strength", use_container_width=True):
                st.session_state.cement = 450.0
                st.session_state.water = 162.0
                st.session_state.superplasticizer = 10.0
                st.session_state.age = 28
                st.rerun()

        with preset_col2:
            if st.button("Standard Mix", use_container_width=True):
                st.session_state.cement = 300.0
                st.session_state.water = 180.0
                st.session_state.superplasticizer = 5.0
                st.session_state.age = 28
                st.rerun()

        with preset_col3:
            if st.button("Low Cement", use_container_width=True):
                st.session_state.cement = 200.0
                st.session_state.blast_furnace_slag = 150.0
                st.session_state.water = 190.0
                st.session_state.age = 28
                st.rerun()

    with col2:
        st.subheader("Prediction & Results")

        # Prepare input data
        input_data = np.array([[cement, blast_furnace_slag, fly_ash, water,
                                superplasticizer, coarse_aggregate, fine_aggregate, age]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Make predictions with all models
        ensemble_pred = ensemble_model.predict(input_scaled)[0]
        xgb_pred = model_data['xgb'].predict(input_scaled)[0]
        gbr_pred = model_data['gbr'].predict(input_scaled)[0]
        rf_pred = model_data['rf'].predict(input_scaled)[0]

        # Calculate average prediction for higher accuracy
        predictions = [ensemble_pred, xgb_pred, gbr_pred, rf_pred]
        final_prediction = np.mean(predictions)

        # Display prediction when button is clicked
        if st.button("🔮 Predict Compressive Strength", use_container_width=True,
                     type="primary"):

            # Main prediction
            st.markdown("### Prediction Result")

            # Create metrics row
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.metric(
                    label="Predicted Strength",
                    value=f"{final_prediction:.1f} MPa",
                    delta=None
                )

            with col_b:
                # Strength classification
                if final_prediction < 20:
                    strength_class = "Low"
                    color = "red"
                    icon = "⚠️"
                elif final_prediction < 40:
                    strength_class = "Moderate"
                    color = "orange"
                    icon = "📊"
                elif final_prediction < 60:
                    strength_class = "High"
                    color = "green"
                    icon = "✅"
                else:
                    strength_class = "Very High"
                    color = "blue"
                    icon = "🏆"

                st.metric(label="Classification", value=f"{icon} {strength_class}")

            with col_c:
                # Confidence indicator (based on prediction spread)
                prediction_std = np.std(predictions)
                confidence = max(0, 100 - (prediction_std * 10))
                st.metric(label="Confidence", value=f"{confidence:.0f}%")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=final_prediction,
                title={'text': "Compressive Strength (MPa)", 'font': {'size': 20}},
                delta={'reference': 40, 'increasing': {'color': "green"}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 20], 'color': "rgba(255, 0, 0, 0.1)"},
                        {'range': [20, 40], 'color': "rgba(255, 165, 0, 0.1)"},
                        {'range': [40, 60], 'color': "rgba(0, 255, 0, 0.1)"},
                        {'range': [60, 100], 'color': "rgba(0, 0, 255, 0.1)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))

            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

            # Model predictions comparison
            st.markdown("### Model Predictions Comparison")
            pred_df = pd.DataFrame({
                'Model': ['Ensemble (Final)', 'XGBoost', 'Gradient Boosting', 'Random Forest'],
                'Prediction (MPa)': [final_prediction, xgb_pred, gbr_pred, rf_pred],
                'Difference': [0, xgb_pred - final_prediction,
                               gbr_pred - final_prediction, rf_pred - final_prediction]
            })

            # Display as bar chart
            fig2 = px.bar(
                pred_df,
                x='Model',
                y='Prediction (MPa)',
                color='Prediction (MPa)',
                color_continuous_scale='Viridis',
                text='Prediction (MPa)',
                title='Predictions from Different Models'
            )
            fig2.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig2.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig2, use_container_width=True)

            # Recommendations based on prediction
            st.markdown("### Recommendations")
            if final_prediction < 20:
                st.warning(
                    "**Consider:** Increasing cement content, reducing water-cement ratio, using superplasticizers, and increasing curing time.")
            elif final_prediction < 40:
                st.info(
                    "**Consider:** Moderate cement content, proper water-cement ratio (0.4-0.5), and 28-day curing.")
            else:
                st.success("**Mix Design:** Optimal! Maintain current proportions for desired strength.")

        # Input summary
        st.markdown("---")
        st.markdown("### Input Summary")
        summary_data = {
            'Parameter': ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water',
                          'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age'],
            'Value': [f"{cement:.1f}", f"{blast_furnace_slag:.1f}", f"{fly_ash:.1f}",
                      f"{water:.1f}", f"{superplasticizer:.1f}", f"{coarse_aggregate:.1f}",
                      f"{fine_aggregate:.1f}", f"{age}"],
            'Unit': ['kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'days']
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)


def analysis_page():
    st.title("📊 Data Analysis & Visualization")
    st.markdown("---")

    df = load_data()

    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
        "📅 Age vs Strength Analysis",
        "📊 Feature Distributions",
        "🔗 Feature Correlations"
    ])

    with tab1:
        st.subheader("Concrete Strength Development Over Time")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Interactive scatter plot of age vs strength
            fig = px.scatter(
                df,
                x='age',
                y='concrete_compressive_strength',
                title='Compressive Strength vs Age',
                labels={
                    'age': 'Age (days)',
                    'concrete_compressive_strength': 'Compressive Strength (MPa)'
                },
                hover_data=df.columns[:-1],
                color='cement',
                color_continuous_scale='Viridis',
                size_max=15,
                opacity=0.6
            )

            # Add trend line using polynomial fit (no statsmodels)
            # Group by age and calculate mean strength
            age_groups = df.groupby('age')['concrete_compressive_strength'].mean().reset_index()

            # Fit polynomial trend (degree 2)
            z = np.polyfit(age_groups['age'], age_groups['concrete_compressive_strength'], 2)
            p = np.poly1d(z)

            # Create trend line data
            trend_x = np.linspace(age_groups['age'].min(), age_groups['age'].max(), 100)
            trend_y = p(trend_x)

            # Add trend line to plot
            fig.add_trace(
                go.Scatter(
                    x=trend_x,
                    y=trend_y,
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', width=3, dash='dash'),
                    hovertemplate='Age: %{x:.0f} days<br>Trend Strength: %{y:.1f} MPa'
                )
            )

            fig.update_layout(
                height=500,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Age Groups Analysis")

            # Age group selection
            age_group = st.selectbox(
                "Select Age Group:",
                ["All", "Early (1-7 days)", "Standard (28 days)", "Long-term (90+ days)"]
            )

            # Filter data based on age group
            if age_group == "Early (1-7 days)":
                filtered_df = df[df['age'] <= 7]
            elif age_group == "Standard (28 days)":
                filtered_df = df[df['age'].between(25, 30)]
            elif age_group == "Long-term (90+ days)":
                filtered_df = df[df['age'] >= 90]
            else:
                filtered_df = df

            # Display statistics
            st.metric("Samples", len(filtered_df))
            st.metric("Avg Strength", f"{filtered_df['concrete_compressive_strength'].mean():.1f} MPa")
            st.metric("Max Strength", f"{filtered_df['concrete_compressive_strength'].max():.1f} MPa")
            st.metric("Min Strength", f"{filtered_df['concrete_compressive_strength'].min():.1f} MPa")

            # Strength gain calculator
            st.markdown("---")
            st.markdown("### Strength Gain Calculator")
            start_age = st.number_input("Start Age (days)", min_value=1, max_value=365, value=7)
            end_age = st.number_input("End Age (days)", min_value=1, max_value=365, value=28)

            if end_age > start_age:
                start_avg = df[df['age'] == start_age]['concrete_compressive_strength'].mean()
                end_avg = df[df['age'] == end_age]['concrete_compressive_strength'].mean()
                if not np.isnan(start_avg) and not np.isnan(end_avg):
                    gain = ((end_avg - start_avg) / start_avg) * 100
                    st.metric(f"Strength Gain ({start_age}→{end_age} days)",
                              f"{gain:.1f}%", f"{end_avg - start_avg:.1f} MPa")

    with tab2:
        st.subheader("Feature Distributions")

        col1, col2 = st.columns(2)

        with col1:
            feature = st.selectbox(
                "Select Feature:",
                df.columns[:-1],
                key="dist_feature"
            )

            plot_type = st.radio(
                "Plot Type:",
                ["Histogram", "Box Plot", "Violin Plot"],
                horizontal=True
            )

        with col2:
            # Statistics for selected feature
            st.metric("Mean", f"{df[feature].mean():.2f}")
            st.metric("Std Dev", f"{df[feature].std():.2f}")
            st.metric("Min", f"{df[feature].min():.2f}")
            st.metric("Max", f"{df[feature].max():.2f}")

        # Create selected plot
        if plot_type == "Histogram":
            fig = px.histogram(
                df,
                x=feature,
                nbins=30,
                title=f'Distribution of {feature}',
                color_discrete_sequence=['#636EFA']
            )
        elif plot_type == "Box Plot":
            fig = px.box(
                df,
                y=feature,
                title=f'Box Plot of {feature}',
                color_discrete_sequence=['#00CC96']
            )
        else:  # Violin Plot
            fig = px.violin(
                df,
                y=feature,
                title=f'Violin Plot of {feature}',
                box=True,
                points="all",
                color_discrete_sequence=['#EF553B']
            )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Feature Correlations")

        # Calculate correlation matrix
        corr_matrix = df.corr()

        # Heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1,
            title='Feature Correlation Matrix'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Top correlations with strength
        st.markdown("### Top Correlations with Compressive Strength")
        strength_corr = corr_matrix['concrete_compressive_strength'].sort_values(ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            # Positive correlations
            pos_corr = strength_corr[strength_corr > 0].iloc[1:6]  # Skip strength itself
            fig_pos = px.bar(
                x=pos_corr.values,
                y=pos_corr.index,
                orientation='h',
                title='Top 5 Positive Correlations',
                color=pos_corr.values,
                color_continuous_scale='Greens',
                range_color=[0, 1]
            )
            fig_pos.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_pos, use_container_width=True)

        with col2:
            # Negative correlations
            neg_corr = strength_corr[strength_corr < 0].iloc[:5]
            if len(neg_corr) > 0:
                fig_neg = px.bar(
                    x=neg_corr.values,
                    y=neg_corr.index,
                    orientation='h',
                    title='Top Negative Correlations',
                    color=neg_corr.values,
                    color_continuous_scale='Reds',
                    range_color=[-1, 0]
                )
                fig_neg.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_neg, use_container_width=True)


def main():
    # Sidebar with user info
    with st.sidebar:
        if st.session_state.authenticated:
            # User info
            st.success(f"👤 Welcome, {st.session_state.username}!")

            # Logout button
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.username = ''
                st.rerun()

            st.markdown("---")

            # Navigation
            st.subheader("Navigation")
            page = st.radio(
                "Select Page:",
                ["🔮 Predict Strength", "📊 Data Analysis"],
                label_visibility="collapsed"
            )

            st.markdown("---")

            # Quick stats
            df = load_data()
            st.markdown("### Dataset Info")
            st.markdown(f"**Samples:** {len(df)}")
            st.markdown(f"**Avg Strength:** {df['concrete_compressive_strength'].mean():.1f} MPa")
            st.markdown(f"**Features:** {len(df.columns) - 1}")

            # Model info
            st.markdown("---")
            st.markdown("### Model Info")
            st.markdown("**Algorithm:** Ensemble (XGBoost + GBR + RF)")
            st.markdown("**Training:** 80% of dataset")
            st.markdown("**Accuracy:** High precision")
        else:
            st.warning("🔒 Please login to continue")
            page = None

    # Main content area
    if not st.session_state.authenticated:
        auth_page()
    else:
        if page == "🔮 Predict Strength":
            prediction_page()
        elif page == "📊 Data Analysis":
            analysis_page()


if __name__ == "__main__":
    # Initialize session state variables for presets
    if 'cement' not in st.session_state:
        st.session_state.cement = 281.0
    if 'water' not in st.session_state:
        st.session_state.water = 181.6

    main()