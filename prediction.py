# =========================================================
# CUSTOMER CHURN PREDICTION DASHBOARD (ML ONLY)
# =========================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from dash import Dash, dcc, html, Input, Output, State

# =========================================================
# LOAD ML MODEL (CHURN PREDICTION MODEL - PIPELINE)
# =========================================================
MODEL_PATH = "churn_prediction_model.pkl"
model = None
model_features = None

try:
    model = joblib.load(MODEL_PATH)
    print("✅ ML model loaded successfully")
    print(f"📋 Model type: {type(model).__name__}")
    
    # Extract feature names from the pipeline
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
        print(f"📋 Pipeline expects features: {model_features}")
    elif hasattr(model, 'steps'):
        # Try to get features from preprocessing step
        for step_name, step in model.steps:
            if hasattr(step, 'feature_names_in_'):
                model_features = list(step.feature_names_in_)
                print(f"📋 Found features in {step_name}: {model_features}")
                break
    
    if model_features is None:
        print("⚠ Could not extract feature names from pipeline")
        print("⚠ Assuming default feature set from error messages")
        model_features = [
            'days_since_last_purchase',
            'total_spent',
            'country',
            'gender',
            'total_quantity',
            'total_orders',
            'age'
        ]
        
except Exception as e:
    print(f"⚠ ML model not loaded: {e}")

# =========================================================
# LOAD DATA
# =========================================================
BASE = r"D:\intership\sales"

print("📂 Loading data files...")
cust_crm = pd.read_csv(f"{BASE}/silver.crm_cust_info.csv")
prd_crm  = pd.read_csv(f"{BASE}/silver.crm_prd_info.csv")
sales    = pd.read_csv(f"{BASE}/silver.crm_sales_details.csv")
cust_erp = pd.read_csv(f"{BASE}/silver.erp_cust_az12.csv")
loc_erp  = pd.read_csv(f"{BASE}/silver.erp_loc_a101.csv")
px_cat   = pd.read_csv(f"{BASE}/silver.erp_px_cat_g1v2.csv")

# =========================================================
# CLEAN SYSTEM COLUMNS
# =========================================================
def clean(df):
    return df.drop(columns=[c for c in df.columns if "dwh" in c.lower()], errors="ignore")

cust_crm, prd_crm, sales, cust_erp, loc_erp, px_cat = map(
    clean, [cust_crm, prd_crm, sales, cust_erp, loc_erp, px_cat]
)

# =========================================================
# DATA MODEL (MERGES)
# =========================================================
print("🔗 Merging data sources...")
df = (
    sales
    .merge(cust_crm, left_on="sls_cust_id", right_on="cst_id", how="left")
    .merge(prd_crm, left_on="sls_prd_key", right_on="prd_key", how="left")
    .merge(px_cat, left_on="cat_id", right_on="id", how="left")
    .merge(cust_erp, left_on="cst_key", right_on="cid", how="left")
    .merge(loc_erp, left_on="cst_key", right_on="cid", how="left")
)

print(f"✅ Merged dataset shape: {df.shape}")

# =========================================================
# FEATURE ENGINEERING FOR ML
# =========================================================
print("🔧 Engineering features for ML...")

df["order_date"] = pd.to_datetime(df["sls_order_dt"], errors="coerce")
df = df.dropna(subset=["order_date"])

# Fix numeric conversions
df["sls_sales"] = pd.to_numeric(df["sls_sales"], errors="coerce").fillna(0)
df["sls_qty"] = pd.to_numeric(df["sls_quantity"], errors="coerce").fillna(0)
df["sls_ord_num"] = pd.to_numeric(df["sls_ord_num"], errors="coerce").fillna(0)

# Fix gender data for ML
df["gender"] = df["cst_gndr"].fillna("Unknown")
df["gender"] = df["gender"].replace({
    "M": "Male", "m": "Male", "F": "Female", "f": "Female",
    "male": "Male", "female": "Female", "MALE": "Male", "FEMALE": "Female"
}).fillna("Unknown")

# For ML: create gender_mapped column (M/F)
df["gender_mapped"] = df["gender"].map({
    "Male": "M", "Female": "F", "male": "M", "female": "F"
}).fillna("M")

# Fix country
df["cntry"] = df["cntry"].fillna("Unknown")

# Create customer name
df["customer_name"] = (
    df["cst_firstname"].fillna("") + " " +
    df["cst_lastname"].fillna("")
)
df["customer_name"] = df["customer_name"].str.strip()
df["customer_name"] = df["customer_name"].replace("", "Unknown Customer")

# Create order ID for proper counting
df["order_id"] = df["sls_ord_num"].astype(str) + "_" + df["order_date"].dt.year.astype(str) + "_" + df["order_date"].dt.month.astype(str)

# =========================================================
# CREATE CUSTOMER FEATURES FOR CHURN PREDICTION
# =========================================================
print("\n🔍 Creating customer features for churn prediction...")

# Find customer identifier column
customer_id_col = 'cst_key'
print(f"✅ Using {customer_id_col} as customer identifier")

# Define aggregation dictionary
agg_dict = {
    "sls_sales": "sum",
    "sls_qty": "sum",
    "order_id": "nunique",
    "order_date": "max",
    "gender": "first",
    "gender_mapped": "first",
    "cntry": "first",
    "customer_name": "first",
    "cst_firstname": "first",
    "cst_lastname": "first"
}

customer_features = df.groupby(customer_id_col).agg(agg_dict).reset_index()

# Rename columns
customer_features = customer_features.rename(columns={
    "sls_sales": "total_spent",
    "sls_qty": "total_quantity",
    "order_id": "total_orders",
    "cntry": "country",
    "customer_name": "customer_name"
})

# Calculate days since last purchase
latest_date = df["order_date"].max()
customer_features["days_since_last_purchase"] = (latest_date - customer_features["order_date"]).dt.days

# Create age (synthetic based on data)
def calculate_age(row):
    # Generate age based on spending behavior
    total_spent = row['total_spent'] if 'total_spent' in row else 0
    if total_spent > 100000:
        return np.random.randint(35, 55)
    elif total_spent > 50000:
        return np.random.randint(30, 50)
    else:
        return np.random.randint(25, 45)

customer_features["age"] = customer_features.apply(calculate_age, axis=1)

# Ensure we have ALL features that the model pipeline expects
print("\n📋 Ensuring all model features exist...")

# Default feature set based on error messages
default_features = [
    'days_since_last_purchase',
    'total_spent',
    'country',
    'gender',
    'total_quantity',
    'total_orders',
    'age'
]

# Use model_features if available, otherwise use default
features_to_create = model_features if model_features else default_features

for feature in features_to_create:
    if feature not in customer_features.columns:
        print(f"⚠ {feature}: Missing - creating default")
        if feature == 'gender':
            # Map from gender_mapped or create default
            if 'gender_mapped' in customer_features.columns:
                customer_features[feature] = customer_features['gender_mapped']
            else:
                customer_features[feature] = 'M'
        elif feature == 'country':
            customer_features[feature] = customer_features.get('country', 'Unknown')
        elif feature == 'age':
            customer_features[feature] = customer_features.get('age', 35)
        elif feature == 'total_spent':
            customer_features[feature] = customer_features.get('total_spent', 0)
        elif feature == 'total_quantity':
            customer_features[feature] = customer_features.get('total_quantity', 0)
        elif feature == 'total_orders':
            customer_features[feature] = customer_features.get('total_orders', 0)
        elif feature == 'days_since_last_purchase':
            customer_features[feature] = customer_features.get('days_since_last_purchase', 180)
        else:
            # For any other unexpected feature
            customer_features[feature] = 0

print(f"✅ Created {len(customer_features)} customer profiles")

# =========================================================
# DASH APP
# =========================================================
app = Dash(__name__)
app.title = "Customer Churn Prediction Dashboard"

# =========================================================
# LAYOUT (ML ONLY)
# =========================================================
app.layout = html.Div([

    # HEADER
    html.Div("🤖 Customer Churn Prediction Dashboard",
        style={
            "background": "#1f2937",
            "color": "white",
            "padding": "16px",
            "fontSize": "26px",
            "fontWeight": "bold",
            "textAlign": "center"
        }
    ),

    html.Div([

        # SIDEBAR
        html.Div([
            html.H4("📊 Customer Selection", style={"color": "#1f2937", "marginBottom": "20px"}),
            
            html.Label("Select Customer for Analysis", style={"fontWeight": "bold", "marginBottom": "5px"}),
            dcc.Dropdown(
                id="customer_select",
                options=[
                    {
                        "label": f"{row['customer_name'] if isinstance(row['customer_name'], str) else f'Customer {idx+1}'} (₹{row['total_spent']:,.0f}, {int(row['total_orders'])} orders)", 
                        "value": idx
                    } 
                    for idx, row in customer_features.iterrows()
                ][:50],  # Show first 50 customers
                value=0,
                style={"width": "100%", "marginBottom": "20px"}
            ),
            
            html.Button("🔮 Predict Churn Risk", id="predict_btn", n_clicks=0,
                        style={
                            "width": "100%", 
                            "background": "#7c3aed", 
                            "color": "white",
                            "padding": "12px",
                            "border": "none",
                            "borderRadius": "5px",
                            "cursor": "pointer",
                            "fontWeight": "bold",
                            "fontSize": "16px",
                            "marginBottom": "20px"
                        }),
            
            html.Div(id="prediction_output", 
                    style={
                        "marginTop": "15px", 
                        "padding": "20px",
                        "background": "#f8fafc",
                        "borderRadius": "10px",
                        "border": "2px solid #e2e8f0",
                        "textAlign": "center",
                        "fontSize": "14px",
                        "minHeight": "200px",
                        "display": "flex",
                        "flexDirection": "column",
                        "justifyContent": "center"
                    })

        ], style={
            "width": "30%", 
            "padding": "25px", 
            "background": "#f3f4f6",
            "height": "100vh",
            "boxShadow": "2px 0 5px rgba(0,0,0,0.1)",
            "overflowY": "auto"
        }),

        # MAIN CONTENT AREA
        html.Div([
            # ML MODEL INFO
            html.Div([
                html.H3("🤖 ML Model Information", style={"color": "#1f2937", "marginBottom": "20px"}),
                
                html.Div(id="model_info", style={
                    "background": "white",
                    "padding": "25px",
                    "borderRadius": "10px",
                    "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                    "marginBottom": "30px"
                }),
                
                # CUSTOMER DETAILS
                html.H3("👤 Customer Details", style={"color": "#1f2937", "marginBottom": "20px"}),
                
                html.Div(id="customer_details", style={
                    "background": "white",
                    "padding": "25px",
                    "borderRadius": "10px",
                    "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                    "marginBottom": "30px"
                }),
                
                # FEATURE IMPORTANCE (if available)
                html.H3("📊 Feature Analysis", style={"color": "#1f2937", "marginBottom": "20px"}),
                
                html.Div(id="feature_info", style={
                    "background": "white",
                    "padding": "25px",
                    "borderRadius": "10px",
                    "boxShadow": "0 2px 5px rgba(0,0,0,0.1)"
                }),
                
            ], style={"padding": "10px"})

        ], style={
            "width": "70%", 
            "padding": "30px",
            "overflowY": "auto",
            "height": "100vh"
        })

    ], style={"display": "flex", "height": "calc(100vh - 60px)"}),

])

# =========================================================
# INITIAL MODEL INFO
# =========================================================
@app.callback(
    Output("model_info", "children"),
    Input("predict_btn", "n_clicks")
)
def update_model_info(n_clicks):
    if model is None:
        return html.Div([
            html.Strong("⚠ ML Model Not Available", style={"color": "#ef4444", "fontSize": "18px"}),
            html.Br(),
            html.P("churn_prediction_model.pkl not found or could not be loaded.", style={"color": "#6b7280"}),
            html.P("Please ensure the model file exists at the specified path.", style={"color": "#6b7280"}),
            html.P("Demo predictions will be shown using simple heuristics.", style={"color": "#f59e0b", "fontWeight": "bold"})
        ])
    else:
        return html.Div([
            html.Strong("✅ ML Model Loaded Successfully", style={"color": "#10b981", "fontSize": "18px"}),
            html.Br(),
            html.P(f"Model Type: {type(model).__name__}", style={"color": "#1f2937"}),
            html.P(f"Features Expected: {len(model_features)} features", style={"color": "#1f2937"}),
            html.P(f"Customer Profiles: {len(customer_features)}", style={"color": "#1f2937"}),
            html.P(f"Model Path: {MODEL_PATH}", style={"color": "#6b7280", "fontSize": "12px"})
        ])

# =========================================================
# CUSTOMER DETAILS CALLBACK
# =========================================================
@app.callback(
    Output("customer_details", "children"),
    Input("customer_select", "value")
)
def update_customer_details(customer_idx):
    if customer_idx >= len(customer_features):
        customer_idx = 0
    
    customer = customer_features.iloc[customer_idx]
    customer_name = customer.get('customer_name', f'Customer {customer_idx+1}')
    
    return html.Div([
        html.Strong(f"👤 {customer_name}", style={"color": "#1f2937", "fontSize": "20px", "marginBottom": "15px"}),
        html.Div([
            html.Div([
                html.Strong("💰 Total Spent", style={"color": "#6b7280"}),
                html.P(f"₹{customer.get('total_spent', 0):,.0f}", style={"color": "#2563eb", "fontSize": "24px", "margin": "5px 0"})
            ], style={"textAlign": "center", "padding": "15px", "flex": "1"}),
            
            html.Div([
                html.Strong("📦 Total Orders", style={"color": "#6b7280"}),
                html.P(f"{int(customer.get('total_orders', 0))}", style={"color": "#10b981", "fontSize": "24px", "margin": "5px 0"})
            ], style={"textAlign": "center", "padding": "15px", "flex": "1"}),
            
            html.Div([
                html.Strong("⏰ Days Since Last Purchase", style={"color": "#6b7280"}),
                html.P(f"{int(customer.get('days_since_last_purchase', 0))} days", style={"color": "#f59e0b", "fontSize": "24px", "margin": "5px 0"})
            ], style={"textAlign": "center", "padding": "15px", "flex": "1"}),
        ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "20px"}),
        
        html.Hr(),
        
        html.Div([
            html.Div([
                html.Strong("👫 Gender", style={"color": "#6b7280"}),
                html.P(f"{customer.get('gender', 'Unknown')}", style={"color": "#8b5cf6", "fontSize": "18px", "margin": "5px 0"})
            ], style={"textAlign": "center", "padding": "10px", "flex": "1"}),
            
            html.Div([
                html.Strong("🌍 Country", style={"color": "#6b7280"}),
                html.P(f"{customer.get('country', 'Unknown')}", style={"color": "#8b5cf6", "fontSize": "18px", "margin": "5px 0"})
            ], style={"textAlign": "center", "padding": "10px", "flex": "1"}),
            
            html.Div([
                html.Strong("🎂 Age", style={"color": "#6b7280"}),
                html.P(f"{int(customer.get('age', 35))} years", style={"color": "#8b5cf6", "fontSize": "18px", "margin": "5px 0"})
            ], style={"textAlign": "center", "padding": "10px", "flex": "1"}),
        ], style={"display": "flex", "justifyContent": "space-between"}),
    ])

# =========================================================
# FEATURE INFO CALLBACK
# =========================================================
@app.callback(
    Output("feature_info", "children"),
    Input("customer_select", "value")
)
def update_feature_info(customer_idx):
    if customer_idx >= len(customer_features):
        customer_idx = 0
    
    customer = customer_features.iloc[customer_idx]
    
    features_needed = model_features if model_features else [
        'days_since_last_purchase',
        'total_spent',
        'country',
        'gender',
        'total_quantity',
        'total_orders',
        'age'
    ]
    
    feature_items = []
    for feature in features_needed:
        value = customer.get(feature, 'N/A')
        if isinstance(value, (int, float)):
            display_value = f"{value:,.0f}" if feature == 'total_spent' else str(value)
        else:
            display_value = str(value)
        
        feature_items.append(
            html.Div([
                html.Strong(f"{feature.replace('_', ' ').title()}:", style={"color": "#1f2937"}),
                html.Span(f" {display_value}", style={"color": "#6b7280"})
            ], style={"marginBottom": "8px"})
        )
    
    return html.Div([
        html.Strong("📋 Features Used for Prediction", style={"color": "#1f2937", "fontSize": "18px", "marginBottom": "15px"}),
        html.P("The following features are being used by the ML model:", style={"color": "#6b7280", "marginBottom": "15px"}),
        html.Div(feature_items, style={
            "display": "grid",
            "gridTemplateColumns": "repeat(2, 1fr)",
            "gap": "15px",
            "marginBottom": "20px"
        }),
        html.P("Note: These features are passed to the ML pipeline for churn prediction.", 
              style={"color": "#9ca3af", "fontSize": "12px", "fontStyle": "italic"})
    ])

# =========================================================
# CHURN PREDICTION CALLBACK
# =========================================================
@app.callback(
    Output("prediction_output", "children"),
    Input("predict_btn", "n_clicks"),
    State("customer_select", "value")
)
def predict_churn(n_clicks, customer_idx):
    if n_clicks == 0:
        return html.Div([
            html.P("👈 Select a customer from dropdown", style={"margin": "0", "fontSize": "16px"}),
            html.P("Click 'Predict Churn Risk' to analyze", style={"margin": "10px 0 0 0", "fontSize": "14px", "color": "#6b7280"})
        ])
    
    if model is None:
        # DEMO MODE - Simple heuristic prediction
        if customer_idx >= len(customer_features):
            customer_idx = 0
        
        customer = customer_features.iloc[customer_idx]
        customer_name = customer.get('customer_name', f'Customer {customer_idx+1}')
        
        # Simple heuristic for demo
        days_since_purchase = customer.get('days_since_last_purchase', 180)
        total_spent = customer.get('total_spent', 0)
        total_orders = customer.get('total_orders', 0)
        
        # Calculate churn probability based on simple rules
        churn_probability = 0.1  # Base probability
        
        if days_since_purchase > 180:
            churn_probability += 0.6
        elif days_since_purchase > 90:
            churn_probability += 0.4
        elif days_since_purchase > 60:
            churn_probability += 0.2
        
        if total_orders < 2:
            churn_probability += 0.2
        
        if total_spent < 5000:
            churn_probability += 0.1
        
        # Cap probability
        churn_probability = min(0.95, max(0.05, churn_probability))
        
        return display_prediction_result(customer, customer_name, churn_probability, is_demo=True)
    
    try:
        if customer_idx >= len(customer_features):
            customer_idx = 0
        
        customer = customer_features.iloc[customer_idx]
        customer_name = customer.get('customer_name', f'Customer {customer_idx+1}')
        
        # Get the features the pipeline expects
        features_needed = model_features if model_features else [
            'days_since_last_purchase',
            'total_spent',
            'country',
            'gender',
            'total_quantity',
            'total_orders',
            'age'
        ]
        
        # Create feature dictionary with EXACT feature names
        feature_dict = {}
        
        for feature in features_needed:
            if feature in customer.index:
                feature_dict[feature] = customer[feature]
            else:
                # Provide reasonable defaults
                if feature == 'gender':
                    feature_dict[feature] = 'M'
                elif feature == 'country':
                    feature_dict[feature] = 'Unknown'
                elif feature == 'age':
                    feature_dict[feature] = 35
                else:
                    feature_dict[feature] = 0
        
        # Create DataFrame with EXACT column names
        X = pd.DataFrame([feature_dict])
        X = X[features_needed]
        
        # Make prediction
        try:
            # Try to get probability
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)[0]
                churn_probability = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
                prediction = 1 if churn_probability > 0.5 else 0
            else:
                prediction = model.predict(X)[0]
                churn_probability = 1.0 if prediction == 1 else 0.0
            
        except Exception as pred_error:
            print(f"❌ Prediction error: {pred_error}")
            # Try simple predict as fallback
            prediction = model.predict(X)[0]
            churn_probability = 1.0 if prediction == 1 else 0.0
        
        return display_prediction_result(customer, customer_name, churn_probability, is_demo=False)
    
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Prediction failed: {error_msg}")
        
        return html.Div([
            html.Strong("❌ Prediction Failed", style={"color": "#dc2626", "fontSize": "18px"}),
            html.Br(),
            html.P(f"Error: {error_msg[:100]}", style={"color": "#6b7280"}),
            html.P("Check console for details", style={"color": "#6b7280", "fontSize": "12px"})
        ])

def display_prediction_result(customer, customer_name, churn_probability, is_demo=False):
    """Display prediction result in a consistent format."""
    
    demo_text = " (DEMO)" if is_demo else ""
    
    if churn_probability > 0.5:
        risk_color = "#dc2626" if churn_probability > 0.8 else "#ef4444" if churn_probability > 0.6 else "#f59e0b"
        risk_emoji = "🔴" if churn_probability > 0.8 else "🟠" if churn_probability > 0.6 else "🟡"
        risk_level = "VERY HIGH" if churn_probability > 0.8 else "HIGH" if churn_probability > 0.6 else "MEDIUM"
        
        return html.Div([
            html.Strong(f"{risk_emoji} CHURN RISK: {risk_level}", style={"color": risk_color, "fontSize": "20px", "marginBottom": "10px"}),
            html.P(f"Customer: {customer_name}", style={"fontWeight": "bold", "margin": "5px 0"}),
            html.P(f"Churn Probability: {churn_probability*100:.1f}%{demo_text}", style={"margin": "5px 0"}),
            html.Hr(style={"margin": "15px 0", "borderColor": "#e5e7eb"}),
            html.P(f"• Last purchase: {int(customer.get('days_since_last_purchase', 0))} days ago", style={"margin": "5px 0", "fontSize": "14px"}),
            html.P(f"• Total orders: {int(customer.get('total_orders', 0))}", style={"margin": "5px 0", "fontSize": "14px"}),
            html.P(f"• Total spent: ₹{customer.get('total_spent', 0):,.0f}", style={"margin": "5px 0", "fontSize": "14px"}),
            html.Hr(style={"margin": "15px 0", "borderColor": "#e5e7eb"}),
            html.Strong("🎯 Recommendation:", style={"color": "#7c3aed", "marginBottom": "5px"}),
            html.P("Send personalized retention offer with discount", style={"margin": "5px 0", "fontSize": "13px"}),
            html.P("Consider loyalty program enrollment", style={"margin": "5px 0", "fontSize": "13px"})
        ])
    else:
        retention_prob = (1 - churn_probability) * 100
        loyalty_color = "#059669" if retention_prob > 90 else "#10b981" if retention_prob > 75 else "#16a34a"
        loyalty_emoji = "🟢" if retention_prob > 90 else "✅" if retention_prob > 75 else "🟡"
        loyalty_level = "VERY LOYAL" if retention_prob > 90 else "LOYAL" if retention_prob > 75 else "ACTIVE"
        
        return html.Div([
            html.Strong(f"{loyalty_emoji} {loyalty_level} CUSTOMER", style={"color": loyalty_color, "fontSize": "20px", "marginBottom": "10px"}),
            html.P(f"Customer: {customer_name}", style={"fontWeight": "bold", "margin": "5px 0"}),
            html.P(f"Retention Probability: {retention_prob:.1f}%{demo_text}", style={"margin": "5px 0"}),
            html.Hr(style={"margin": "15px 0", "borderColor": "#e5e7eb"}),
            html.P(f"• Last purchase: {int(customer.get('days_since_last_purchase', 0))} days ago", style={"margin": "5px 0", "fontSize": "14px"}),
            html.P(f"• Total orders: {int(customer.get('total_orders', 0))}", style={"margin": "5px 0", "fontSize": "14px"}),
            html.P(f"• Lifetime value: ₹{customer.get('total_spent', 0):,.0f}", style={"margin": "5px 0", "fontSize": "14px"}),
            html.Hr(style={"margin": "15px 0", "borderColor": "#e5e7eb"}),
            html.Strong("🎯 Recommendation:", style={"color": "#7c3aed", "marginBottom": "5px"}),
            html.P("Send loyalty rewards and exclusive offers", style={"margin": "5px 0", "fontSize": "13px"}),
            html.P("Encourage referral program participation", style={"margin": "5px 0", "fontSize": "13px"})
        ])

# =========================================================
# RUN APP
# =========================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🤖 CUSTOMER CHURN PREDICTION DASHBOARD")
    print("=" * 70)
    print(f"📊 Total Records: {len(df):,}")
    print(f"💰 Total Sales: ₹{df['sls_sales'].sum():,.0f}")
    print(f"👥 Unique Customers: {df['customer_name'].nunique():,}")
    print(f"🤖 ML Model Loaded: {'Yes' if model is not None else 'No'}")
    if model is not None:
        print(f"📋 Model Type: {type(model).__name__}")
        print(f"📋 Features Expected: {len(model_features) if model_features else 'Using default features'}")
    print(f"👤 Customer Profiles: {len(customer_features)}")
    print("=" * 70)
    print("🌐 Dashboard URL: http://127.0.0.1:8050")
    print("=" * 70)
    print("📋 DASHBOARD FEATURES:")
    print("• Customer Churn Prediction (ML)")
    print("• Customer Selection Dropdown")
    print("• Detailed Customer Profiles")
    print("• ML Model Information")
    print("• Feature Analysis")
    print("• Retention Recommendations")
    print("=" * 70)
    
    app.run(debug=True, port=8050)