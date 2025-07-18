import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="🚀 Crypto Market Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e0e2e6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .crypto-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: white;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


# Functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_crypto_data(vs_currency='usd', per_page=50):
    """Fetch cryptocurrency data from CoinGecko API"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': vs_currency,
            'order': 'market_cap_desc',
            'per_page': per_page,
            'page': 1,
            'sparkline': True,
            'price_change_percentage': '1h,24h,7d,30d'
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return []


@st.cache_data(ttl=300)
def fetch_global_data():
    """Fetch global cryptocurrency market data"""
    try:
        url = "https://api.coingecko.com/api/v3/global"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()['data']
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching global data: {e}")
        return {}


@st.cache_data(ttl=300)
def fetch_coin_history(coin_id, days=7):
    """Fetch historical price data for a specific coin"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'hourly' if days <= 7 else 'daily'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching historical data: {e}")
        return {}


def format_currency(value):
    """Format currency values"""
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    elif value >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif value >= 1e6:
        return f"${value / 1e6:.2f}M"
    elif value >= 1e3:
        return f"${value / 1e3:.2f}K"
    else:
        return f"${value:.2f}"


def format_percentage(value):
    """Format percentage values with color"""
    if value is None:
        return "N/A"

    color = "green" if value > 0 else "red"
    return f"<span style='color: {color}'>{value:+.2f}%</span>"


# Main Dashboard
def main():
    # Header
    st.markdown("# 🚀 Crypto Market Dashboard")
    st.markdown("### Real-time cryptocurrency market data powered by CoinGecko API")

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Dashboard Settings")

        # Refresh button
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()

        # Currency selection
        currency = st.selectbox(
            "💱 Select Currency",
            options=['usd', 'eur', 'jpy', 'gbp', 'thb'],
            index=0,
            format_func=lambda x: {
                'usd': '🇺🇸 USD', 'eur': '🇪🇺 EUR',
                'jpy': '🇯🇵 JPY', 'gbp': '🇬🇧 GBP', 'thb': '🇹🇭 THB'
            }[x]
        )

        # Number of coins to display
        num_coins = st.slider("📊 Number of Coins", min_value=10, max_value=100, value=50, step=10)

        # Auto-refresh
        auto_refresh = st.checkbox("🔄 Auto Refresh (5 min)", value=False)

        st.markdown("---")
        st.markdown("### 📈 Market Categories")

        categories = {
            "🪙 All Coins": "all",
            "₿ Bitcoin": "bitcoin",
            "⟠ Ethereum": "ethereum",
            "🔗 DeFi": "defi",
            "🎮 Gaming": "gaming",
            "🎨 NFT": "nft"
        }

        selected_category = st.selectbox("Select Category", list(categories.keys()))

    # Fetch data
    with st.spinner("🔄 Loading cryptocurrency data..."):
        crypto_data = fetch_crypto_data(vs_currency=currency, per_page=num_coins)
        global_data = fetch_global_data()

    if not crypto_data:
        st.error("Failed to load cryptocurrency data. Please try again later.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(crypto_data)

    # Global Market Overview
    st.markdown("## 🌍 Global Market Overview")

    if global_data:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_market_cap = global_data.get('total_market_cap', {}).get(currency, 0)
            st.metric(
                label="🏦 Total Market Cap",
                value=format_currency(total_market_cap),
                delta=f"{global_data.get('market_cap_change_percentage_24h_usd', 0):.2f}%"
            )

        with col2:
            total_volume = global_data.get('total_volume', {}).get(currency, 0)
            st.metric(
                label="📊 24h Volume",
                value=format_currency(total_volume)
            )

        with col3:
            btc_dominance = global_data.get('market_cap_percentage', {}).get('btc', 0)
            st.metric(
                label="₿ BTC Dominance",
                value=f"{btc_dominance:.1f}%"
            )

        with col4:
            active_cryptos = global_data.get('active_cryptocurrencies', 0)
            st.metric(
                label="🪙 Active Cryptos",
                value=f"{active_cryptos:,}"
            )

    # Market Cap Treemap
    st.markdown("## 📊 Market Cap Treemap")

    # Prepare data for treemap
    top_20 = df.head(20).copy()
    top_20['market_cap_formatted'] = top_20['market_cap'].apply(format_currency)
    top_20['change_24h'] = top_20['price_change_percentage_24h'].fillna(0)

    fig_treemap = px.treemap(
        top_20,
        path=['name'],
        values='market_cap',
        color='change_24h',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        title="Market Capitalization Treemap (Top 20 Coins)",
        hover_data={
            'current_price': ':$,.2f',
            'market_cap_formatted': True,
            'change_24h': ':.2f'
        }
    )

    fig_treemap.update_layout(
        height=600,
        font_size=12,
        title_font_size=16
    )

    st.plotly_chart(fig_treemap, use_container_width=True)

    # Top Cryptocurrencies Table
    st.markdown("## 💰 Top Cryptocurrencies")

    # Prepare data for display
    display_df = df[['market_cap_rank', 'name', 'symbol', 'current_price',
                     'market_cap', 'price_change_percentage_1h_in_currency',
                     'price_change_percentage_24h', 'price_change_percentage_7d_in_currency',
                     'total_volume']].copy()

    display_df.columns = ['Rank', 'Name', 'Symbol', 'Price', 'Market Cap',
                          '1h Change', '24h Change', '7d Change', 'Volume']

    # Format columns
    currency_symbol = {'usd': '$', 'eur': '€', 'jpy': '¥', 'gbp': '£', 'thb': '฿'}[currency]

    display_df['Price'] = display_df['Price'].apply(
        lambda x: f"{currency_symbol}{x:,.4f}" if x < 1 else f"{currency_symbol}{x:,.2f}")
    display_df['Market Cap'] = display_df['Market Cap'].apply(format_currency)
    display_df['Volume'] = display_df['Volume'].apply(format_currency)

    # Add percentage formatting with colors
    for col in ['1h Change', '24h Change', '7d Change']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")

    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        hide_index=True
    )

    # Price Charts
    st.markdown("## 📈 Price Charts")

    # Coin selection for detailed chart
    col1, col2 = st.columns([3, 1])

    with col1:
        selected_coins = st.multiselect(
            "Select coins for price comparison",
            options=df['name'].tolist(),
            default=df['name'].head(5).tolist(),
            max_selections=10
        )

    with col2:
        chart_days = st.selectbox(
            "Time Period",
            options=[1, 7, 30, 90, 365],
            index=1,
            format_func=lambda x: f"{x} day{'s' if x > 1 else ''}"
        )

    if selected_coins:
        # Create price comparison chart
        fig_prices = go.Figure()

        for coin_name in selected_coins:
            coin_data = df[df['name'] == coin_name].iloc[0]
            coin_id = coin_data['id']

            # Fetch historical data
            with st.spinner(f"Loading {coin_name} price history..."):
                history = fetch_coin_history(coin_id, chart_days)

            if history and 'prices' in history:
                timestamps = [datetime.fromtimestamp(price[0] / 1000) for price in history['prices']]
                prices = [price[1] for price in history['prices']]

                fig_prices.add_trace(go.Scatter(
                    x=timestamps,
                    y=prices,
                    mode='lines',
                    name=coin_name,
                    line=dict(width=2),
                    hovertemplate=f'<b>{coin_name}</b><br>' +
                                  'Price: $%{y:,.2f}<br>' +
                                  'Date: %{x}<br>' +
                                  '<extra></extra>'
                ))

        fig_prices.update_layout(
            title=f"Price Comparison - Last {chart_days} Day{'s' if chart_days > 1 else ''}",
            xaxis_title="Date",
            yaxis_title=f"Price ({currency_symbol})",
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig_prices, use_container_width=True)

    # Market Analysis
    st.markdown("## 📊 Market Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Volume vs Market Cap
        fig_scatter = px.scatter(
            df.head(30),
            x='total_volume',
            y='market_cap',
            size='market_cap',
            color='price_change_percentage_24h',
            hover_name='name',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title="Volume vs Market Cap",
            labels={
                'total_volume': '24h Volume',
                'market_cap': 'Market Cap',
                'price_change_percentage_24h': '24h Change (%)'
            }
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        # Price Change Distribution
        price_changes = df['price_change_percentage_24h'].dropna()

        fig_hist = px.histogram(
            x=price_changes,
            nbins=30,
            title="24h Price Change Distribution",
            labels={'x': '24h Price Change (%)', 'y': 'Number of Coins'},
            color_discrete_sequence=['#667eea']
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

    # Market Movers
    st.markdown("## 🚀 Market Movers")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Top Gainers (24h)")
        gainers = df.nlargest(10, 'price_change_percentage_24h')[
            ['name', 'symbol', 'current_price', 'price_change_percentage_24h']
        ].copy()
        gainers['price_change_percentage_24h'] = gainers['price_change_percentage_24h'].apply(
            lambda x: f"+{x:.2f}%" if pd.notna(x) else "N/A"
        )
        gainers.columns = ['Name', 'Symbol', 'Price', '24h Change']
        st.dataframe(gainers, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("### 📉 Top Losers (24h)")
        losers = df.nsmallest(10, 'price_change_percentage_24h')[
            ['name', 'symbol', 'current_price', 'price_change_percentage_24h']
        ].copy()
        losers['price_change_percentage_24h'] = losers['price_change_percentage_24h'].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
        )
        losers.columns = ['Name', 'Symbol', 'Price', '24h Change']
        st.dataframe(losers, hide_index=True, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>📊 Data provided by <a href='https://coingecko.com' target='_blank'>CoinGecko API</a> | 
        🔄 Last updated: {}</p>
        <p>⚠️ This is for educational purposes only. Not financial advice.</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(300)  # Wait 5 minutes
        st.rerun()


if __name__ == "__main__":
    main()