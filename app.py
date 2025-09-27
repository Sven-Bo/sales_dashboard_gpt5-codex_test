import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = Path("data")
PRODUCTS_PATH = DATA_DIR / "products.json"
SUPPLIERS_PATH = DATA_DIR / "suppliers.json"
SALES_PATH = DATA_DIR / "sales.json"


@st.cache_data(show_spinner=False)
def load_data():
    """Load and prepare products, suppliers, and sales data."""
    def _load_json(path: Path) -> pd.DataFrame:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return pd.DataFrame(payload["data"]) if payload["data"] else pd.DataFrame()

    products_df = _load_json(PRODUCTS_PATH)
    suppliers_df = _load_json(SUPPLIERS_PATH)
    sales_df = _load_json(SALES_PATH)

    if sales_df.empty:
        return products_df, suppliers_df, sales_df

    # Convert date/time fields
    sales_df["Transaction_Date"] = pd.to_datetime(sales_df["Transaction_Date"])
    sales_df["Transaction_Time"] = pd.to_timedelta(sales_df["Transaction_Time"])
    sales_df["Transaction_DateTime"] = sales_df["Transaction_Date"] + sales_df["Transaction_Time"]

    # Calculate discount-adjusted revenue
    sales_df["Revenue"] = (
        sales_df["Quantity"]
        * sales_df["Unit_Price"]
        * (1 - sales_df["Discount_Rate"].fillna(0))
    )

    # Enrich sales with product meta-data
    merged_df = sales_df.merge(
        products_df.rename(columns={"ID": "Product_Key"}),
        left_on="Product_ID",
        right_on="Product_Key",
        how="left",
        suffixes=("", "_Product"),
    )

    # Enrich with supplier details
    merged_df = merged_df.merge(
        suppliers_df.rename(columns={"ID": "Supplier_Key"}),
        left_on="Supplier_ID",
        right_on="Supplier_Key",
        how="left",
        suffixes=("", "_Supplier"),
    )

    # Remove duplicate column names introduced by merges
    if "Unit_Price" not in merged_df.columns and "Unit_Price_Product" in merged_df.columns:
        merged_df = merged_df.rename(columns={"Unit_Price_Product": "Unit_Price"})

    redundant_columns = [
        col
        for col in [
            "Product_Key",
            "Supplier_Key",
            "Unit_Price_Product",
            "Unit_Price_Supplier",
        ]
        if col in merged_df.columns
    ]
    if redundant_columns:
        merged_df = merged_df.drop(columns=redundant_columns)

    return products_df, suppliers_df, merged_df


def filter_sales(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    st.sidebar.header("Filters")

    min_date, max_date = df["Transaction_Date"].min(), df["Transaction_Date"].max()
    default_dates = (min_date, max_date)
    selected_dates = st.sidebar.date_input(
        "Date range",
        value=default_dates,
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
    # Handle single date returned by date_input
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates

    filtered_df = df[(df["Transaction_Date"].dt.date >= start_date) & (df["Transaction_Date"].dt.date <= end_date)]

    store_options = sorted(df["Store_Location"].dropna().unique())
    selected_stores = st.sidebar.multiselect("Store location", options=store_options, default=store_options)
    if selected_stores:
        filtered_df = filtered_df[filtered_df["Store_Location"].isin(selected_stores)]

    payment_options = sorted(df["Payment_Method"].dropna().unique())
    selected_payments = st.sidebar.multiselect(
        "Payment method", options=payment_options, default=payment_options
    )
    if selected_payments:
        filtered_df = filtered_df[filtered_df["Payment_Method"].isin(selected_payments)]

    category_options = sorted(df["Category"].dropna().unique())
    selected_categories = st.sidebar.multiselect(
        "Product category", options=category_options, default=category_options
    )
    if selected_categories:
        filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]

    supplier_location_options = sorted(df["Location"].dropna().unique())
    selected_supplier_locations = st.sidebar.multiselect(
        "Supplier location", options=supplier_location_options, default=supplier_location_options
    )
    if selected_supplier_locations:
        filtered_df = filtered_df[filtered_df["Location"].isin(selected_supplier_locations)]

    max_discount_pct = float((df["Discount_Rate"].fillna(0).max() * 100).round(2))
    discount_min_pct, discount_max_pct = st.sidebar.slider(
        "Discount rate (%)",
        min_value=0.0,
        max_value=max_discount_pct or 0.0,
        value=(0.0, max_discount_pct or 0.0),
        step=1.0 if max_discount_pct >= 1 else 0.1,
    )
    filtered_df = filtered_df[
        (filtered_df["Discount_Rate"].fillna(0) * 100 >= discount_min_pct)
        & (filtered_df["Discount_Rate"].fillna(0) * 100 <= discount_max_pct)
    ]

    return filtered_df


def render_kpis(df: pd.DataFrame) -> None:
    total_revenue = df["Revenue"].sum()
    total_quantity = df["Quantity"].sum()
    avg_discount = df["Discount_Rate"].mean() * 100 if not df["Discount_Rate"].empty else 0.0
    order_count = df["Order_ID"].nunique()

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total revenue", f"${total_revenue:,.2f}")
    kpi_cols[1].metric("Units sold", f"{int(total_quantity):,}")
    kpi_cols[2].metric("Avg. discount", f"{avg_discount:.1f}%")
    kpi_cols[3].metric("Unique orders", f"{order_count:,}")


def render_charts(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Adjust the filters to see insights.")
        return

    granularity = st.radio(
        "Revenue trend granularity",
        options=["Daily", "Weekly", "Monthly", "Quarterly"],
        horizontal=True,
        key="trend_granularity",
    )

    freq_map = {
        "Weekly": "W",
        "Monthly": "ME",
        "Quarterly": "QE",
    }

    if granularity == "Daily":
        revenue_by_period = (
            df.groupby("Transaction_Date", as_index=False)["Revenue"].sum()
        )
    else:
        freq = freq_map[granularity]
        revenue_by_period = (
            df.groupby(pd.Grouper(key="Transaction_Date", freq=freq))["Revenue"]
            .sum()
            .reset_index()
        )

    revenue_by_period = revenue_by_period.dropna(subset=["Transaction_Date"]).sort_values("Transaction_Date")
    revenue_by_period = revenue_by_period.rename(columns={"Transaction_Date": "Period"})

    fig_revenue_trend = px.line(
        revenue_by_period,
        x="Period",
        y="Revenue",
        title=f"Revenue trend ({granularity.lower()})",
        markers=True,
    )
    fig_revenue_trend.update_layout(yaxis_title="Revenue", xaxis_title=granularity)

    revenue_by_category = (
        df.groupby("Category", as_index=False)
        .agg({"Revenue": "sum", "Quantity": "sum"})
        .sort_values("Revenue", ascending=False)
    )
    fig_category = px.bar(
        revenue_by_category,
        x="Category",
        y="Revenue",
        hover_data={"Quantity": ":,"},
        title="Revenue by category",
        text_auto=".2s",
    )
    fig_category.update_layout(yaxis_title="Revenue", xaxis_title="Category")

    top_products = (
        df.groupby(["Product_ID", "Product_Name"], as_index=False)["Revenue"].sum()
        .sort_values("Revenue", ascending=False)
        .head(10)
    )
    fig_top_products = px.bar(
        top_products,
        x="Revenue",
        y="Product_Name",
        orientation="h",
        title="Top products by revenue",
        text_auto=".2s",
    )
    fig_top_products.update_layout(xaxis_title="Revenue", yaxis_title="Product")

    payment_mix = (
        df.groupby("Payment_Method", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
    )
    fig_payment = px.pie(
        payment_mix,
        names="Payment_Method",
        values="Revenue",
        title="Revenue share by payment method",
        hole=0.35,
    )

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_revenue_trend, use_container_width=True)
    col2.plotly_chart(fig_category, use_container_width=True)

    col3, col4 = st.columns(2)
    col3.plotly_chart(fig_top_products, use_container_width=True)
    col4.plotly_chart(fig_payment, use_container_width=True)


def render_correlation_analysis(df: pd.DataFrame) -> None:
    st.subheader("Correlation analysis")

    if df.empty:
        st.info("No transactions available for correlation analysis.")
        return

    numeric_columns = ["Quantity", "Unit_Price", "Discount_Rate", "Revenue"]
    numeric_columns = [col for col in numeric_columns if col in df.columns]

    if len(numeric_columns) < 2:
        st.info("Not enough numeric features to compute correlations.")
        return

    corr_matrix = df[numeric_columns].corr().round(3)
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig_corr.update_layout(title="Correlation heatmap", margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig_corr, use_container_width=True)

    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_pairs = upper_tri.stack().reset_index()
    if not corr_pairs.empty:
        corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
        corr_pairs["|Correlation|"] = corr_pairs["Correlation"].abs()
        top_pairs = corr_pairs.sort_values("|Correlation|", ascending=False).head(5)
        top_pairs = top_pairs.drop(columns="|Correlation|")
        st.markdown("#### Top correlated feature pairs")
        st.dataframe(top_pairs.reset_index(drop=True), hide_index=True, use_container_width=True)
    else:
        st.info("Correlations between numeric features are negligible.")


def render_supplier_insights(df: pd.DataFrame) -> None:
    st.subheader("Supplier performance insights")

    if df.empty:
        st.info("No supplier activity found for the current filters.")
        return

    supplier_df = df.copy()
    supplier_df["Supplier_Name"] = supplier_df["Supplier_Name"].fillna("Unknown Supplier")
    supplier_df["Location"] = supplier_df["Location"].fillna("Unknown")
    supplier_df["Specialization"] = supplier_df["Specialization"].fillna("Unknown")

    supplier_summary = (
        supplier_df.groupby(["Supplier_Name", "Location", "Specialization"], dropna=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Units_Sold=("Quantity", "sum"),
            Avg_Discount=("Discount_Rate", lambda x: x.fillna(0).mean()),
            Unique_Orders=("Order_ID", pd.Series.nunique),
            Distinct_Products=("Product_ID", pd.Series.nunique),
            Avg_Unit_Price=("Unit_Price", "mean"),
        )
        .reset_index()
    )

    if supplier_summary.empty:
        st.info("No supplier metrics available after applying the filters.")
        return

    supplier_summary["Avg_Discount"] = supplier_summary["Avg_Discount"].fillna(0) * 100
    supplier_summary = supplier_summary.sort_values("Revenue", ascending=False)

    top_suppliers_chart = px.bar(
        supplier_summary.head(10),
        x="Revenue",
        y="Supplier_Name",
        color="Location",
        orientation="h",
        title="Top suppliers by revenue",
        text_auto=".2s",
    )
    top_suppliers_chart.update_layout(xaxis_title="Revenue", yaxis_title="Supplier")

    st.plotly_chart(top_suppliers_chart, use_container_width=True)

    spotlight_options = supplier_summary["Supplier_Name"].tolist()
    default_supplier = spotlight_options[0] if spotlight_options else None
    selected_supplier = st.selectbox(
        "Supplier spotlight",
        options=spotlight_options,
        index=spotlight_options.index(default_supplier) if default_supplier else 0,
    )

    spotlight_df = supplier_df[supplier_df["Supplier_Name"] == selected_supplier]

    if spotlight_df.empty:
        st.info("No detailed data for the selected supplier.")
        return

    total_revenue = spotlight_df["Revenue"].sum()
    total_units = spotlight_df["Quantity"].sum()
    avg_discount = spotlight_df["Discount_Rate"].fillna(0).mean() * 100
    order_count = spotlight_df["Order_ID"].nunique()

    metrics = st.columns(4)
    metrics[0].metric("Revenue", f"${total_revenue:,.2f}")
    metrics[1].metric("Units sold", f"{int(total_units):,}")
    metrics[2].metric("Avg. discount", f"{avg_discount:.1f}%")
    metrics[3].metric("Unique orders", f"{order_count:,}")

    supplier_trend = (
        spotlight_df.groupby(pd.Grouper(key="Transaction_Date", freq="ME"))["Revenue"]
        .sum()
        .reset_index()
        .dropna(subset=["Transaction_Date"])
    )

    if not supplier_trend.empty:
        supplier_trend_fig = px.bar(
            supplier_trend,
            x="Transaction_Date",
            y="Revenue",
            title=f"Monthly revenue for {selected_supplier}",
        )
        supplier_trend_fig.update_layout(xaxis_title="Month", yaxis_title="Revenue")
        st.plotly_chart(supplier_trend_fig, use_container_width=True)

    top_products = (
        spotlight_df.groupby(["Product_ID", "Product_Name"], dropna=False)["Revenue"].sum()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )

    if not top_products.empty:
        st.markdown("#### Best-selling products for the selected supplier")
        top_products["Revenue"] = top_products["Revenue"].map(lambda x: f"${x:,.2f}")
        st.dataframe(top_products.rename(columns={"Product_ID": "Product ID", "Product_Name": "Product"}), hide_index=True, use_container_width=True)

    st.markdown("#### Supplier leaderboard")
    leaderboard_display = supplier_summary.copy()
    leaderboard_display["Revenue"] = leaderboard_display["Revenue"].map(lambda x: f"${x:,.2f}")
    leaderboard_display["Avg_Discount"] = leaderboard_display["Avg_Discount"].map(lambda x: f"{x:.1f}%")
    leaderboard_display["Avg_Unit_Price"] = leaderboard_display["Avg_Unit_Price"].map(lambda x: f"${x:,.2f}")
    st.dataframe(
        leaderboard_display.rename(
            columns={
                "Supplier_Name": "Supplier",
                "Location": "Supplier location",
                "Specialization": "Specialization",
                "Revenue": "Revenue",
                "Units_Sold": "Units sold",
                "Avg_Discount": "Avg. discount",
                "Unique_Orders": "Unique orders",
                "Distinct_Products": "Distinct products",
                "Avg_Unit_Price": "Avg. unit price",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


def render_detail_table(df: pd.DataFrame) -> None:
    st.subheader("Transaction detail")
    if df.empty:
        st.warning("No transactions for the current filter selection.")
        return

    display_columns = [
        "Transaction_Date",
        "Transaction_Time",
        "Store_Location",
        "Payment_Method",
        "Product_Name",
        "Category",
        "Supplier_Name",
        "Location",
        "Quantity",
        "Unit_Price",
        "Discount_Rate",
        "Revenue",
        "Order_ID",
    ]
    detail_df = df[display_columns].copy()
    detail_df["Transaction_Date"] = detail_df["Transaction_Date"].dt.date
    detail_df["Discount_Rate"] = (detail_df["Discount_Rate"].fillna(0) * 100).map(lambda x: f"{x:.1f}%")
    detail_df["Revenue"] = detail_df["Revenue"].map(lambda x: f"${x:,.2f}")
    detail_df["Unit_Price"] = detail_df["Unit_Price"].map(lambda x: f"${x:,.2f}")

    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data",
        data=csv_data,
        file_name="filtered_sales.csv",
        mime="text/csv",
    )


def main():
    st.set_page_config(
        page_title="Sales Performance Dashboard",
        page_icon="🛒",
        layout="wide",
    )

    st.title("Sales Performance Dashboard")
    st.caption(
        "Interact with the filters to explore revenue, product, and supplier insights across stores."  # noqa: E501
    )

    products_df, suppliers_df, sales_df = load_data()

    if sales_df.empty:
        st.error("Sales data not found or empty. Please verify the JSON source files.")
        return

    filtered_sales = filter_sales(sales_df)

    st.markdown("---")
    render_kpis(filtered_sales)
    st.markdown("---")
    render_charts(filtered_sales)
    st.markdown("---")
    render_correlation_analysis(filtered_sales)
    st.markdown("---")
    render_supplier_insights(filtered_sales)
    st.markdown("---")
    render_detail_table(filtered_sales)

    with st.expander("Data dictionary"):
        st.markdown(
            "- **Products**: Loaded from `data/products.json`, includes product ID, name, category, unit price, and supplier reference."
        )
        st.markdown(
            "- **Suppliers**: Loaded from `data/suppliers.json`, includes supplier ID, name, location, and specialization."
        )
        st.markdown(
            "- **Sales**: Loaded from `data/sales.json`, includes transaction date/time, store, payment method, product, quantity, unit price, discount rate, and order ID."
        )


if __name__ == "__main__":
    main()
