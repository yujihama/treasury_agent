import streamlit as st
import pandas as pd
import os
from data_utils import DataProcessor, Visualizer
from agent_utils import BasicAnalyzer
from dotenv import load_dotenv

# ページ設定
st.set_page_config(
    page_title="Treasury Management Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS（シンプルなデザイン）
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: left;
        margin-top: -5rem !important; /* 上の余白を負の値で詰める */
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 1rem !important;
        margin: 0rem 0rem 1rem 0rem !important;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #ffffff;
        padding: 0.5rem 1rem;
        z-index: 1000;
        box-shadow: 0 -2px 4px rgba(0,0,0,0.05);
    }
    /* 画面下に固定された入力欄によってコンテンツが隠れないよう余白を追加 */
    .block-container {
        padding-bottom: 6rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # メインヘッダー
    st.markdown('<h1 class="main-header">Treasury Management Agent</h1>', unsafe_allow_html=True)
    
    # サイドバー設定
    with st.sidebar:
        st.markdown('<h2 class="section-header">設定</h2>', unsafe_allow_html=True)
        
        # OpenAI APIキー設定
        api_key = st.text_input("OpenAI API Key", type="password", key="openai_api_key")
        st.session_state.api_key = api_key.strip()
        
        st.markdown('<h2 class="section-header">データアップロード</h2>', unsafe_allow_html=True)
        
        # ファイルアップロード
        uploaded_files = {}
        file_configs = [
            ("balances", "口座残高CSV", "口座残高データをアップロードしてください"),
            ("transactions", "取引履歴CSV", "取引履歴データをアップロードしてください"),
            ("fx_rates", "為替レートCSV", "為替レートデータをアップロードしてください"),
            ("payables", "買掛金CSV", "買掛金データをアップロードしてください"),
            ("receivables", "売掛金CSV", "売掛金データをアップロードしてください"),
            ("loans", "借入金CSV", "借入金データをアップロードしてください"),
            ("investments", "投資CSV", "投資データをアップロードしてください"),
            ("derivatives", "デリバティブCSV", "デリバティブデータをアップロードしてください")
        ]
        
        for key, label, help_text in file_configs:
            uploaded_files[key] = st.file_uploader(
                label,
                type=["csv"],
                key=f"upload_{key}",
                help=help_text
            )
        
        # サンプルデータ使用オプション
        st.markdown("---")
        use_sample_data = st.checkbox(
            "サンプルデータを使用",
            help="テスト用のサンプルデータを使用します"
        )
    
    # データ読み込み
    dataframes = load_data(uploaded_files, use_sample_data)
    
    # メインコンテンツ
    if any(not df.empty for df in dataframes.values()):
        display_main_content(dataframes, st.session_state.api_key)
    else:
        st.info("データをアップロードするか、サンプルデータを使用してください。")

def load_data(uploaded_files, use_sample_data):
    """データを読み込む。アップロード > サンプルデータ > data フォルダの優先順で読み込む"""
    # 初期化
    dataframes = {
        'balances': pd.DataFrame(),
        'transactions': pd.DataFrame(),
        'fx_rates': pd.DataFrame(),
        # 追加データセット
        'payables': pd.DataFrame(),  # accounts payable
        'receivables': pd.DataFrame(),  # accounts receivable
        'loans': pd.DataFrame(),
        'investments': pd.DataFrame(),
        'derivatives': pd.DataFrame(),
    }

    # ------------------------ 1) サンプルデータ -------------------------
    if use_sample_data:
        DATA_DIR = 'data'
        sample_files = {
            'balances': os.path.join(DATA_DIR, 'account_balance.csv'),
            'transactions': os.path.join(DATA_DIR, 'transactions.csv'),
            'fx_rates': os.path.join(DATA_DIR, 'exchange_rates.csv'),
            'payables': os.path.join(DATA_DIR, 'accounts_payable.csv'),
            'receivables': os.path.join(DATA_DIR, 'accounts_receivable.csv'),
            'loans': os.path.join(DATA_DIR, 'loans.csv'),
            'investments': os.path.join(DATA_DIR, 'investments.csv'),
            'derivatives': os.path.join(DATA_DIR, 'derivatives.csv'),
        }
        for key, filename in sample_files.items():
            if os.path.exists(filename):
                try:
                    # data_utils の load_csv_data を使って読み込み、列名を変換
                    dataframes[key] = DataProcessor.load_csv_data(filename, key)
                except Exception as e:
                    st.error(f"サンプルデータ {filename} の読み込みに失敗: {e}")
        return dataframes  # サンプル使用時はここで終了

    # ------------------------ 2) アップロードファイル -------------------
    file_labels = {
        'balances': '口座残高',
        'transactions': '取引履歴',
        'fx_rates': '為替レート',
        'payables': '買掛金',
        'receivables': '売掛金',
        'loans': '借入金',
        'investments': '投資',
        'derivatives': 'デリバティブ',
    }
    for key, uploaded_file in uploaded_files.items():
        if uploaded_file:
            dataframes[key] = DataProcessor.load_csv_data(uploaded_file, file_labels.get(key, key))

    # ------------------------ 3) data フォルダ --------------------------
    DATA_DIR = 'data'
    local_files = {
        'balances': os.path.join(DATA_DIR, 'account_balance.csv'),
        'transactions': os.path.join(DATA_DIR, 'transactions.csv'),
        'fx_rates': os.path.join(DATA_DIR, 'exchange_rates.csv'),
        'payables': os.path.join(DATA_DIR, 'accounts_payable.csv'),
        'receivables': os.path.join(DATA_DIR, 'accounts_receivable.csv'),
        'loans': os.path.join(DATA_DIR, 'loans.csv'),
        'investments': os.path.join(DATA_DIR, 'investments.csv'),
        'derivatives': os.path.join(DATA_DIR, 'derivatives.csv'),
    }
    for key, path in local_files.items():
        # 既に読み込まれていなければローカルファイルを読む
        if dataframes[key].empty and os.path.exists(path):
            dataframes[key] = DataProcessor.load_csv_data(path, key)

    return dataframes

def display_main_content(dataframes, api_key):
    """メインコンテンツを表示"""
    
    # タブ作成
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Data Viewer", "Analysis", "Agent Chat"])
    
    with tab1:
        display_dashboard(dataframes)
    
    with tab2:
        display_data_viewer(dataframes)
    
    with tab3:
        display_analysis(dataframes)
    
    with tab4:
        display_chat_interface(dataframes, api_key)

def display_dashboard(dataframes):
    """ダッシュボードを表示"""
    st.markdown('<h2 class="section-header">Dashboard</h2>', unsafe_allow_html=True)
    
    df_balances = dataframes['balances']
    df_transactions = dataframes['transactions']
    
    if not df_balances.empty:
        # フィルタセクション
        st.markdown("### Filter Settings")
        
        # フィルタ用のカラム
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        
        with filter_col1:
            # 国別フィルタ
            countries = ['All'] + sorted(df_balances['国名'].unique().tolist()) if '国名' in df_balances.columns else ['All']
            selected_countries = st.multiselect("Country", countries, default=['All'])
        
        with filter_col2:
            # 口座別フィルタ
            accounts = ['All'] + sorted(df_balances['口座ID'].unique().tolist()) if '口座ID' in df_balances.columns else ['All']
            selected_accounts = st.multiselect("Account", accounts, default=['All'])
        
        with filter_col3:
            # 通貨別フィルタ
            currencies = ['All'] + sorted(df_balances['通貨'].unique().tolist()) if '通貨' in df_balances.columns else ['All']
            selected_currencies = st.multiselect("Currency", currencies, default=['All'])
        
        with filter_col4:
            # 期間フィルタ
            date_range = None # 初期化
            if '日付' in df_balances.columns:
                df_balances['日付'] = pd.to_datetime(df_balances['日付'])
                min_date = df_balances['日付'].min().date()
                max_date = df_balances['日付'].max().date()
                
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
        
        # データフィルタリング
        filtered_balances = df_balances.copy()
        filtered_transactions = df_transactions.copy()
        
        # 国フィルタ適用
        if 'All' not in selected_countries and selected_countries:
            filtered_balances = filtered_balances[filtered_balances['国名'].isin(selected_countries)]
        
        # 口座フィルタ適用
        if 'All' not in selected_accounts and selected_accounts:
            filtered_balances = filtered_balances[filtered_balances['口座ID'].isin(selected_accounts)]
            if not df_transactions.empty and '口座ID' in df_transactions.columns:
                filtered_transactions = filtered_transactions[filtered_transactions['口座ID'].isin(selected_accounts)]
        
        # 通貨フィルタ適用
        if 'All' not in selected_currencies and selected_currencies:
            filtered_balances = filtered_balances[filtered_balances['通貨'].isin(selected_currencies)]
        
        # 期間フィルタ適用
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_balances = filtered_balances[
                (filtered_balances['日付'].dt.date >= start_date) & 
                (filtered_balances['日付'].dt.date <= end_date)
            ]
            if not df_transactions.empty and '日付' in df_transactions.columns:
                df_transactions['日付'] = pd.to_datetime(df_transactions['日付'])
                filtered_transactions = filtered_transactions[
                    (filtered_transactions['日付'].dt.date >= start_date) & 
                    (filtered_transactions['日付'].dt.date <= end_date)
                ]
        
        st.markdown("---")
        
        # フィルタ結果の表示
        if not filtered_balances.empty:
            # 基本メトリクス
            latest_balances = DataProcessor.get_latest_balances(filtered_balances)
            total_balance = latest_balances['残高'].sum()
            account_count = len(latest_balances)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Balance", f"{total_balance:,.0f}")
            with col2:
                st.metric("Account Count", f"{account_count}")
            with col3:
                if '国名' in latest_balances.columns:
                    country_count = latest_balances['国名'].nunique()
                    st.metric("Country Count", f"{country_count}")
            with col4:
                data_count = len(filtered_balances)
                st.metric("Data Count", f"{data_count}")
            
            # ----------- JPY換算処理を先に実行 -------------
            if not dataframes['fx_rates'].empty:
                latest_balances = DataProcessor.convert_balances_to_jpy(latest_balances, dataframes['fx_rates'])
            
            # グラフ表示
            col1, col2 = st.columns(2)
            
            with col1:
                # 口座別残高円グラフ (JPY換算後)
                pie_fig = Visualizer.create_balance_pie_chart(latest_balances)
                st.plotly_chart(pie_fig, use_container_width=True)
            
            with col2:
                # 世界地図 (JPY換算後)
                if '国コード' in latest_balances.columns:
                    country_summary = DataProcessor.get_country_summary(latest_balances)
                    if not country_summary.empty:
                        map_fig = Visualizer.create_world_map(country_summary)
                        st.plotly_chart(map_fig, use_container_width=True)
            
            # キャッシュフロー分析
            if not filtered_transactions.empty:
                daily_flow = DataProcessor.calculate_daily_cashflow(filtered_transactions)
                if not daily_flow.empty:
                    cashflow_fig = Visualizer.create_cashflow_chart(daily_flow)
                    st.plotly_chart(cashflow_fig, use_container_width=True)

            # ----------- 日本円換算サマリー -------------
            if '残高JPY' not in latest_balances.columns:
                jpy_balances = latest_balances.copy()
                jpy_balances['残高JPY'] = jpy_balances['残高']
            else:
                jpy_balances = latest_balances

            # 通貨別
            currency_summary = (
                jpy_balances.groupby('通貨')['残高JPY']
                .sum()
                .reset_index()
                .sort_values('残高JPY', ascending=False)
            )

            # 銀行別
            bank_summary = (
                jpy_balances.groupby('銀行名')['残高JPY']
                .sum()
                .reset_index()
                .sort_values('残高JPY', ascending=False)
            )

            # 地域別
            jpy_balances = DataProcessor.add_region_column(jpy_balances)
            region_summary = (
                jpy_balances.groupby('地域')['残高JPY']
                .sum()
                .reset_index()
                .sort_values('残高JPY', ascending=False)
            )

            # グラフ作成
            currency_fig = Visualizer.create_bar_chart(currency_summary, '通貨', '残高JPY', '通貨別合計残高 (JPY換算)')
            bank_fig = Visualizer.create_bar_chart(bank_summary, '銀行名', '残高JPY', '銀行別残高 (JPY換算)')
            region_fig = Visualizer.create_bar_chart(region_summary, '地域', '残高JPY', '地域別残高 (JPY換算)')

            st.markdown("### 日本円換算サマリー")
            col_j1, col_j2, col_j3 = st.columns(3)
            with col_j1:
                st.plotly_chart(currency_fig, use_container_width=True)
            with col_j2:
                st.plotly_chart(bank_fig, use_container_width=True)
            with col_j3:
                st.plotly_chart(region_fig, use_container_width=True)
            # ----------- 追加ここまで -------------
        else:
            st.warning("No data found for the selected filter conditions.")
    else:
        st.info("No balance data found. Please upload data.")

def display_data_viewer(dataframes):
    """データビューアを表示"""
    st.markdown('<h2 class="section-header">Data Viewer</h2>', unsafe_allow_html=True)

    data_labels = {
        'balances': 'Balance Data',
        'transactions': 'Transaction Data',
        'fx_rates': 'FX Rate Data',
        # 追加データ
        'payables': 'Accounts Payable Data',
        'receivables': 'Accounts Receivable Data',
        'loans': 'Loans Data',
        'investments': 'Investments Data',
        'derivatives': 'Derivatives Data',
    }

    for key, label in data_labels.items():
        df = dataframes.get(key, pd.DataFrame())
        if df is not None and not df.empty:
            st.subheader(label)

            # データフレーム情報
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Row Count", len(df))
            with col2:
                st.metric("Column Count", len(df.columns))
            with col3:
                if '日付' in df.columns:
                    date_range = f"{df['日付'].min().strftime('%Y-%m-%d')} ～ {df['日付'].max().strftime('%Y-%m-%d')}"
                    st.metric("Date Range", date_range)

            # データ表示
            st.dataframe(df, use_container_width=True)
            st.markdown("---")

def display_analysis(dataframes):
    """分析画面を表示"""
    st.markdown('<h2 class="section-header">Analysis</h2>', unsafe_allow_html=True)
    
    df_balances = dataframes['balances']
    df_transactions = dataframes['transactions']
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not df_balances.empty:
            st.subheader("Balance Analysis")
            balance_analysis = BasicAnalyzer.analyze_balances(df_balances)
            st.markdown(balance_analysis)

            # 追加: 回帰分析によるトレンド
            trend_text, trend_fig = BasicAnalyzer.analyze_balance_trend(df_balances)
            st.markdown(trend_text)
            if trend_fig is not None:
                st.plotly_chart(trend_fig, use_container_width=True)
    
    with col2:
        if not df_transactions.empty:
            st.subheader("Transaction Analysis")
            transaction_analysis = BasicAnalyzer.analyze_transactions(df_transactions)
            st.markdown(transaction_analysis)

            # フィルタ UI
            unique_countries = sorted(set(df_balances['国名'].dropna().unique()).union(set(df_transactions['相手国名'].dropna().unique())))

            with st.expander("資金フローフィルタ", expanded=False):
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    selected_sources = st.multiselect("流出国 (source)", options=unique_countries, default=[])
                with col_f2:
                    selected_targets = st.multiselect("流入国 (target)", options=unique_countries, default=[])

            # 国間フローの可視化
            flow_text, flow_fig = BasicAnalyzer.analyze_country_flows(df_balances, df_transactions, selected_sources, selected_targets)
            st.markdown(flow_text)
            if flow_fig is not None:
                st.plotly_chart(flow_fig, use_container_width=True)

def display_chat_interface(dataframes, api_key):
    """チャットインターフェースを表示"""
    st.markdown('<h2 class="section-header">Agent Chat</h2>', unsafe_allow_html=True)
    
    if not api_key:
        st.warning("To use the chat feature, please enter the OpenAI API key in the sidebar.")
        st.info("Basic analysis features are available without an API key.")
        return

    # LangGraph 版チャットインターフェースを呼び出す
    st.session_state.api_key = api_key

    # .env ファイルを読み込む
    load_dotenv()

    from chat_langgraph import main as chat_main
    try:
        chat_main(dataframes)
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()

