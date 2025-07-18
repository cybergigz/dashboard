import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import numpy as np
import os
from typing import Dict, List, Optional, Union
import logging
from functools import wraps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🚀 Crypto Market Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize dark mode in session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Custom CSS with dark mode support
def apply_custom_css():
    if st.session_state.dark_mode:
        # Dark mode CSS
        st.markdown("""
        <style>
            .main > div {
                padding-top: 2rem;
                background-color: #1e1e1e;
                color: #ffffff;
            }
            .stMetric {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
                color: #ffffff;
            }
            .crypto-card {
                background: linear-gradient(135deg, #4a4a4a 0%, #2d2d2d 100%);
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
                color: white;
            }
            .stSelectbox {
                margin-bottom: 1rem;
            }
            h1 {
                color: #ffffff;
                text-align: center;
                margin-bottom: 2rem;
            }
            .sidebar .sidebar-content {
                background: linear-gradient(135deg, #2d2d2d 0%, #1e1e1e 100%);
            }
            .stDataFrame {
                background-color: #2d2d2d;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Light mode CSS
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

apply_custom_css()


# Enhanced API client with retry logic
class CryptoAPIClient:
    def __init__(self, base_url: str = "https://api.coingecko.com/api/v3"):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        self.session.headers.update({
            'User-Agent': 'Crypto-Dashboard/1.0',
            'Accept': 'application/json'
        })
    
    def _make_request(self, endpoint: str, params: Dict = None, timeout: int = 15) -> Optional[Dict]:
        """Make API request with error handling and rate limiting"""
        try:
            url = f"{self.base_url}/{endpoint}"
            logger.info(f"Making request to: {url}")
            
            response = self.session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout error for {endpoint}")
            st.error("⏰ Request timed out. Please try again.")
            return None
            
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error for {endpoint}")
            st.error("🔌 Connection error. Please check your internet connection.")
            return None
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.error(f"Rate limit exceeded for {endpoint}")
                st.error("⚠️ Rate limit exceeded. Please wait before making more requests.")
            else:
                logger.error(f"HTTP error {e.response.status_code} for {endpoint}")
                st.error(f"HTTP error: {e.response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error for {endpoint}: {str(e)}")
            st.error(f"An unexpected error occurred: {str(e)}")
            return None

# Initialize API client
api_client = CryptoAPIClient()

# Enhanced caching with different TTL based on data type
@st.cache_data(ttl=180)  # Cache for 3 minutes (more frequent updates)
def fetch_crypto_data(vs_currency='usd', per_page=50) -> List[Dict]:
    """Fetch cryptocurrency data from CoinGecko API"""
    params = {
        'vs_currency': vs_currency,
        'order': 'market_cap_desc',
        'per_page': min(per_page, 250),  # API limit
        'page': 1,
        'sparkline': True,
        'price_change_percentage': '1h,24h,7d,30d'
    }
    
    data = api_client._make_request("coins/markets", params)
    return data if data else []


@st.cache_data(ttl=600)  # Cache for 10 minutes (global data changes less frequently)
def fetch_global_data() -> Dict:
    """Fetch global cryptocurrency market data"""
    data = api_client._make_request("global")
    return data.get('data', {}) if data else {}


@st.cache_data(ttl=900)  # Cache for 15 minutes (historical data changes less frequently)
def fetch_coin_history(coin_id: str, days: int = 7) -> Dict:
    """Fetch historical price data for a specific coin"""
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'hourly' if days <= 7 else 'daily'
    }
    
    data = api_client._make_request(f"coins/{coin_id}/market_chart", params)
    return data if data else {}


def validate_numeric_data(data: List[Dict]) -> List[Dict]:
    """Validate and sanitize numeric data from API"""
    validated_data = []
    
    for item in data:
        try:
            # Validate required fields
            required_fields = ['id', 'name', 'symbol', 'current_price', 'market_cap']
            if not all(field in item for field in required_fields):
                logger.warning(f"Missing required fields in item: {item.get('name', 'Unknown')}")
                continue
                
            # Sanitize numeric values
            numeric_fields = [
                'current_price', 'market_cap', 'total_volume',
                'price_change_percentage_1h_in_currency',
                'price_change_percentage_24h',
                'price_change_percentage_7d_in_currency',
                'price_change_percentage_30d_in_currency'
            ]
            
            for field in numeric_fields:
                if field in item and item[field] is not None:
                    try:
                        item[field] = float(item[field])
                        # Check for reasonable bounds
                        if field == 'current_price' and (item[field] < 0 or item[field] > 1e10):
                            logger.warning(f"Suspicious price value: {item[field]} for {item['name']}")
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid numeric value for {field}: {item[field]}")
                        item[field] = None
                        
            # Sanitize string fields
            string_fields = ['name', 'symbol', 'id']
            for field in string_fields:
                if field in item and item[field]:
                    item[field] = str(item[field]).strip()
                    
            validated_data.append(item)
            
        except Exception as e:
            logger.error(f"Error validating item: {str(e)}")
            continue
            
    return validated_data

def safe_float(value: Union[str, int, float, None], default: float = 0.0) -> float:
    """Safely convert value to float with default"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def format_currency(value: Union[str, int, float, None]) -> str:
    """Format currency values with validation"""
    value = safe_float(value)
    
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


def format_percentage(value: Union[str, int, float, None]) -> str:
    """Format percentage values with color and validation"""
    value = safe_float(value)
    
    if value is None or (isinstance(value, float) and np.isnan(value)):
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
        
        # Dark mode toggle
        dark_mode_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
        if dark_mode_toggle != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode_toggle
            apply_custom_css()
            st.rerun()

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
        auto_refresh = st.checkbox("🔄 Auto Refresh (3 min)", value=False)
        
        # Portfolio tracking
        st.markdown("---")
        st.markdown("### 💼 Portfolio Tracker")
        
        # Initialize portfolio in session state
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {}
            
        # Add to portfolio
        with st.expander("➕ Add to Portfolio"):
            if 'df' in locals():
                coin_options = df['name'].tolist() if not df.empty else []
            else:
                coin_options = []
                
            selected_coin = st.selectbox(
                "Select Coin",
                options=coin_options,
                key="portfolio_coin_select"
            )
            
            amount = st.number_input(
                "Amount",
                min_value=0.0,
                step=0.1,
                key="portfolio_amount"
            )
            
            if st.button("➕ Add to Portfolio"):
                if selected_coin and amount > 0:
                    if selected_coin in st.session_state.portfolio:
                        st.session_state.portfolio[selected_coin] += amount
                    else:
                        st.session_state.portfolio[selected_coin] = amount
                    st.success(f"Added {amount} {selected_coin} to portfolio")
                    st.rerun()
                    
        # Display portfolio
        if st.session_state.portfolio:
            st.markdown("**Your Portfolio:**")
            for coin, amount in st.session_state.portfolio.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{coin}: {amount}")
                with col2:
                    if st.button("❌", key=f"remove_{coin}"):
                        del st.session_state.portfolio[coin]
                        st.rerun()
                        
            if st.button("Clear Portfolio"):
                st.session_state.portfolio = {}
                st.rerun()
                
            # Export portfolio
            if st.button("💾 Export Portfolio"):
                portfolio_json = json.dumps(st.session_state.portfolio, indent=2)
                st.download_button(
                    label="Download Portfolio",
                    data=portfolio_json,
                    file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

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

    # Validate and sanitize data
    crypto_data = validate_numeric_data(crypto_data)
    
    if not crypto_data:
        st.error("No valid cryptocurrency data available. Please try again later.")
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
    
    # Export functionality
    st.markdown("### 💾 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        # Export to CSV
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="💾 Download CSV",
            data=csv_data,
            file_name=f"crypto_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="csv_download"
        )
        
    with col2:
        # Export to JSON
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="💾 Download JSON",
            data=json_data,
            file_name=f"crypto_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="json_download"
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

    # Portfolio Performance Section
    if st.session_state.portfolio:
        st.markdown("## 💼 Portfolio Performance")
        
        portfolio_data = []
        total_value = 0
        
        for coin_name, amount in st.session_state.portfolio.items():
            coin_data = df[df['name'] == coin_name]
            if not coin_data.empty:
                current_price = coin_data.iloc[0]['current_price']
                value = amount * current_price
                change_24h = coin_data.iloc[0]['price_change_percentage_24h']
                
                portfolio_data.append({
                    'Coin': coin_name,
                    'Amount': amount,
                    'Price': current_price,
                    'Value': value,
                    '24h Change': change_24h
                })
                total_value += value
        
        if portfolio_data:
            portfolio_df = pd.DataFrame(portfolio_data)
            
            # Portfolio summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Portfolio Value", format_currency(total_value))
            with col2:
                total_change = sum(p['Value'] * p['24h Change'] / 100 for p in portfolio_data if p['24h Change'] is not None)
                st.metric("24h P&L", format_currency(total_change))
            with col3:
                num_coins = len(portfolio_data)
                st.metric("Number of Coins", num_coins)
            
            # Portfolio breakdown
            st.dataframe(
                portfolio_df.style.format({
                    'Amount': '{:.4f}',
                    'Price': '${:.4f}',
                    'Value': '${:.2f}',
                    '24h Change': '{:+.2f}%'
                }),
                use_container_width=True
            )
            
            # Portfolio allocation chart
            if len(portfolio_data) > 1:
                fig_pie = px.pie(
                    portfolio_df,
                    values='Value',
                    names='Coin',
                    title="Portfolio Allocation"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(180)  # Wait 3 minutes
        st.rerun()


if __name__ == "__main__":
    main()