import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ======================
# LOAD MODEL
# ======================
model, scaler, features, rules = pickle.load(open("model.pkl", "rb"))

st.title("🛒 E-Commerce Customer Segmentation & Recommendation System")

# ======================
# HISTOGRAM (WITH BORDER)
# ======================
st.subheader("Price Distribution")

data = np.random.randint(1000, 100000, 400)

fig, ax = plt.subplots()
ax.hist(data, bins=30, edgecolor='black')
ax.set_xlabel("Price")
ax.set_ylabel("Count")

st.pyplot(fig)

# ======================
# USER INPUT
# ======================
st.subheader("Enter Customer Details")

price = st.slider("Price", 1000, 100000, 5000)
quantity = st.slider("Quantity", 1, 5, 2)
discount = st.slider("Discount Applied", 0, 50, 10)
rating = st.slider("Rating", 1, 5, 3)
session = st.slider("Session Duration", 1, 60, 15)

device = st.selectbox("Device", ["Mobile", "Desktop", "Tablet"])
browser = st.selectbox("Browser", ["Chrome", "Safari", "Edge"])
shipping = st.selectbox("Shipping Type", ["Standard", "Express"])

# ======================
# ENCODING
# ======================
device_map = {"Mobile": 0, "Desktop": 1, "Tablet": 2}
browser_map = {"Chrome": 2, "Safari": 1, "Edge": 0}
shipping_map = {"Standard": 0, "Express": 1}

input_data = {
    "Price": price,
    "Quantity": quantity,
    "DiscountApplied": discount,
    "Rating": rating,
    "SessionDuration": session,
    "Device": device_map[device],
    "Browser": browser_map[browser],
    "ShippingType": shipping_map[shipping]
}

input_df = pd.DataFrame([input_data])

# Match training features
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[features]
input_scaled = scaler.transform(input_df)

# ======================
# DECODE FUNCTION
# ======================
def decode(item):
    item = str(item)

    if "ShippingType_1" in item:
        return "Shipping: Express"
    if "ShippingType_0" in item:
        return "Shipping: Standard"

    if "Device_0" in item:
        return "Device: Mobile"
    if "Device_1" in item:
        return "Device: Desktop"
    if "Device_2" in item:
        return "Device: Tablet"

    if "Browser_2" in item:
        return "Browser: Chrome"
    if "Browser_1" in item:
        return "Browser: Safari"
    if "Browser_0" in item:
        return "Browser: Edge"

    if "ReturnStatus_1" in item:
        return "Return: Yes"
    if "ReturnStatus_0" in item:
        return "Return: No"

    return None

# ======================
# ANALYZE BUTTON
# ======================
if st.button("🔍 Analyze Customer"):

    # SEGMENT
    segment = model.predict(input_scaled)[0]
    st.success(f"Customer Segment: {segment}")

    # ======================
    # RECOMMENDATION (FIXED)
    # ======================
    st.subheader("💡 Recommended for this customer:")

    # Store best per category
    best = {
        "Shipping": None,
        "Device": None,
        "Browser": None,
        "Return": None
    }

    if rules is not None and not rules.empty:

        sorted_rules = rules.sort_values(by="confidence", ascending=False)

        for _, row in sorted_rules.iterrows():
            for item in list(row["consequents"]):

                decoded = decode(item)

                if decoded is None:
                    continue

                category = decoded.split(":")[0]

                # fill only once (best confidence first)
                if category in best and best[category] is None:
                    best[category] = decoded

    # ======================
    # FALLBACK (IMPORTANT)
    # ======================
    if best["Shipping"] is None:
        best["Shipping"] = "Shipping: Express"

    if best["Device"] is None:
        best["Device"] = "Device: Mobile"

    if best["Browser"] is None:
        best["Browser"] = "Browser: Chrome"

    if best["Return"] is None:
        best["Return"] = "Return: No"

    # ======================
    # DISPLAY (CLEAN FORMAT)
    # ======================
    st.write(f"👉 {best['Shipping']}")
    st.write(f"👉 {best['Device']}")
    st.write(f"👉 {best['Browser']}")
    st.write(f"👉 {best['Return']}")