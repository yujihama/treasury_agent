import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pycountry
import streamlit as st
from typing import Optional, Dict, List, Tuple

class DataProcessor:
    """データ処理とビジュアライゼーションのためのユーティリティクラス"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def load_csv_data(uploaded_file, data_type: str) -> pd.DataFrame:
        """CSVファイルを読み込み、基本的な前処理を行う"""
        if uploaded_file is None:
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(uploaded_file)
            
            # 日付カラムの処理
            if '日付' in df.columns:
                try:
                    df['日付'] = pd.to_datetime(df['日付'])
                except Exception as e:
                    st.warning(f"{data_type}の日付列の変換に失敗しました: {e}")
            
            # 国コードの標準化
            df = DataProcessor._standardize_country_codes(df)
            
            st.success(f"{data_type}データが正常に読み込まれました。({len(df)}件)")
            return df
            
        except Exception as e:
            st.error(f"{data_type}データの読み込みに失敗しました: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def _standardize_country_codes(df: pd.DataFrame) -> pd.DataFrame:
        """国名と国コードの標準化"""
        try:
            # 国名から国コードを生成
            if '国名' in df.columns and '国コード' not in df.columns:
                def get_country_code(country_name):
                    try:
                        return pycountry.countries.get(name=country_name).alpha_3
                    except:
                        # 一般的な国名のマッピング
                        mapping = {
                            'アメリカ': 'USA',
                            'イギリス': 'GBR',
                            'ドイツ': 'DEU',
                            'フランス': 'FRA',
                            '韓国': 'KOR'
                        }
                        return mapping.get(country_name, None)
                
                df['国コード'] = df['国名'].apply(get_country_code)
            
            # 国コードから国名を生成
            elif '国コード' in df.columns and '国名' not in df.columns:
                def get_country_name(country_code):
                    try:
                        return pycountry.countries.get(alpha_3=country_code).name
                    except:
                        return None
                
                df['国名'] = df['国コード'].apply(get_country_name)
                
        except Exception as e:
            st.warning(f"国コードの標準化中にエラーが発生しました: {e}")
        
        return df
    
    @staticmethod
    def get_latest_balances(df_balances: pd.DataFrame) -> pd.DataFrame:
        """各口座の最新残高を取得"""
        if df_balances.empty or '日付' not in df_balances.columns:
            return pd.DataFrame()
        
        latest_balances = df_balances.loc[
            df_balances.groupby('口座ID')['日付'].idxmax()
        ].reset_index(drop=True)
        
        return latest_balances
    
    @staticmethod
    def calculate_daily_cashflow(df_transactions: pd.DataFrame) -> pd.DataFrame:
        """日次キャッシュフローを計算"""
        if df_transactions.empty or '日付' not in df_transactions.columns:
            return pd.DataFrame()
        
        daily_flow = df_transactions.groupby('日付')['金額'].sum().reset_index()
        daily_flow = daily_flow.sort_values('日付')
        
        # 累積キャッシュフローも計算
        daily_flow['累積金額'] = daily_flow['金額'].cumsum()
        
        return daily_flow
    
    @staticmethod
    def get_country_summary(df_balances: pd.DataFrame) -> pd.DataFrame:
        """国別の残高サマリーを取得"""
        if df_balances.empty or '国コード' not in df_balances.columns:
            return pd.DataFrame()
        
        # 最新残高を取得
        latest_balances = DataProcessor.get_latest_balances(df_balances)
        
        # 国別に集計
        country_summary = latest_balances.groupby(['国コード', '国名']).agg({
            '残高': 'sum',
            '口座ID': 'count'
        }).reset_index()
        
        country_summary.rename(columns={'口座ID': '口座数'}, inplace=True)
        country_summary = country_summary.sort_values('残高', ascending=False)
        
        return country_summary

    @staticmethod
    def convert_balances_to_jpy(df_balances: pd.DataFrame, df_fx_rates: pd.DataFrame) -> pd.DataFrame:
        """残高を日本円に換算した列 `残高JPY` を追加した DataFrame を返す。
        ・JPY の場合はレート1固定
        ・それ以外は 同日付かつ <通貨>/JPY のレートを掛け合わせる
        """
        if df_balances.empty:
            return df_balances.copy()

        df = df_balances.copy()
        # 既に JPY 換算済みならスキップ
        if '残高JPY' in df.columns:
            return df

        # レート情報がない場合はそのまま返却
        if df_fx_rates is None or df_fx_rates.empty:
            df['残高JPY'] = df['残高']
            return df

        # 日付を datetime に統一
        if '日付' in df_fx_rates.columns and df_fx_rates['日付'].dtype != 'datetime64[ns]':
            df_fx_rates = df_fx_rates.copy()
            df_fx_rates['日付'] = pd.to_datetime(df_fx_rates['日付'])

        # 通貨ペアから base 通貨抽出 (XXX/JPY)
        fx_jpy = df_fx_rates[df_fx_rates['通貨ペア'].str.endswith('/JPY')].copy()
        fx_jpy['通貨'] = fx_jpy['通貨ペア'].str.split('/').str[0]

        # マージキー準備
        df_merge = df.merge(
            fx_jpy[['日付', '通貨', 'レート']],
            how='left',
            left_on=['日付', '通貨'],
            right_on=['日付', '通貨']
        )

        # JPY はレート1
        df_merge['レート'].fillna(1.0, inplace=True)

        df_merge['残高JPY'] = df_merge['残高'] * df_merge['レート']
        return df_merge

    @staticmethod
    def add_region_column(df_balances: pd.DataFrame) -> pd.DataFrame:
        """国コードを大陸 / 地域名にマッピングした `地域` 列を追加"""
        if df_balances.empty:
            return df_balances

        # 既に地域列があればスキップ
        if '地域' in df_balances.columns:
            return df_balances

        region_map = {
            'JPN': 'Asia',
            'CHN': 'Asia',
            'KOR': 'Asia',
            'USA': 'North America',
            'CAN': 'North America',
            'MEX': 'North America',
            'DEU': 'Europe',
            'FRA': 'Europe',
            'GBR': 'Europe',
            'ESP': 'Europe',
            'ITA': 'Europe',
            'AUS': 'Oceania',
            'NZL': 'Oceania',
            'BRA': 'South America',
            'ARG': 'South America',
            'ZAF': 'Africa',
        }
        df = df_balances.copy()
        if '国コード' in df.columns:
            df['地域'] = df['国コード'].map(region_map).fillna('Other')
        else:
            df['地域'] = 'Other'
        return df

class Visualizer:
    """ビジュアライゼーション用のクラス"""
    
    @staticmethod
    def create_cashflow_chart(daily_flow: pd.DataFrame) -> go.Figure:
        """キャッシュフロー推移チャートを作成"""
        if daily_flow.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('日次キャッシュフロー', '累積キャッシュフロー'),
            vertical_spacing=0.1
        )
        
        # 日次キャッシュフロー
        fig.add_trace(
            go.Scatter(
                x=daily_flow['日付'],
                y=daily_flow['金額'],
                mode='lines+markers',
                name='日次金額',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # 累積キャッシュフロー
        fig.add_trace(
            go.Scatter(
                x=daily_flow['日付'],
                y=daily_flow['累積金額'],
                mode='lines',
                name='累積金額',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            title_text="キャッシュフロー分析",
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_balance_pie_chart(latest_balances: pd.DataFrame) -> go.Figure:
        """口座別残高の円グラフを作成"""
        if latest_balances.empty:
            return go.Figure()
        
        # 正の残高のみを表示
        positive_balances = latest_balances[latest_balances['残高'] > 0]
        
        fig = go.Figure(data=[go.Pie(
            labels=positive_balances['口座ID'],
            values=positive_balances['残高'],
            hovertemplate='<b>%{label}</b><br>残高: %{value:,.0f}<br>割合: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="口座別残高分布",
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_world_map(country_summary: pd.DataFrame) -> go.Figure:
        """世界地図での国別残高表示"""
        if country_summary.empty:
            return go.Figure()
        
        fig = px.choropleth(
            country_summary,
            locations="国コード",
            color="残高",
            hover_name="国名",
            hover_data={"残高": ":,.0f", "口座数": True},
            color_continuous_scale="Viridis",
            projection="natural earth",
            title="世界各国の口座残高分布"
        )
        
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth'
            ),
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_transaction_analysis(df_transactions: pd.DataFrame) -> go.Figure:
        """取引タイプ別の分析チャート"""
        if df_transactions.empty:
            return go.Figure()
        
        # 取引タイプ別の集計
        transaction_summary = df_transactions.groupby('取引タイプ').agg({
            '金額': ['sum', 'count', 'mean']
        }).round(2)
        
        transaction_summary.columns = ['合計金額', '取引件数', '平均金額']
        transaction_summary = transaction_summary.reset_index()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('取引タイプ別合計金額', '取引タイプ別件数'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 合計金額
        fig.add_trace(
            go.Bar(
                x=transaction_summary['取引タイプ'],
                y=transaction_summary['合計金額'],
                name='合計金額',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # 取引件数
        fig.add_trace(
            go.Bar(
                x=transaction_summary['取引タイプ'],
                y=transaction_summary['取引件数'],
                name='取引件数',
                marker_color='lightcoral'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            title_text="取引分析",
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_balance_trend_chart(daily_balance: pd.DataFrame, slope: float, intercept: float) -> go.Figure:
        """総残高の時系列推移と回帰直線を描画"""
        if daily_balance.empty:
            return go.Figure()

        # 散布図（実測値）
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily_balance['日付'],
                y=daily_balance['総残高'] if '総残高' in daily_balance.columns else daily_balance['残高'],
                mode='lines+markers',
                name='実測値',
                line=dict(color='#1f77b4')
            )
        )

        # 回帰直線
        # x 軸は日付、日数換算した方が分かりやすいが、Plotly では日付で描画しても問題ないため
        x_numeric = (daily_balance['日付'] - daily_balance['日付'].min()).dt.days
        y_pred = intercept + slope * x_numeric
        fig.add_trace(
            go.Scatter(
                x=daily_balance['日付'],
                y=y_pred,
                mode='lines',
                name='回帰直線',
                line=dict(color='red', dash='dash')
            )
        )

        fig.update_layout(
            title="総残高推移と回帰トレンド",
            xaxis_title="日付",
            yaxis_title="総残高",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    @staticmethod
    def create_country_flow_sankey(flow_df: pd.DataFrame, max_links: int = 50) -> go.Figure:
        """国間資金フローを Sankey ダイアグラムで可視化
        flow_df: columns ['source', 'target', 'value']
        max_links: 表示する最大リンク数（多すぎると可読性低下）
        改良ポイント：
        • 上位リンクのみ表示
        • ソース国 / シンク国 / 中立国 で左右にグループ化しノード座標を固定
        • ソースを左 (x=0.0)、シンクを右 (x=1.0)、中立を中央 (x=0.5)
        • arrangement='fixed' で交差を最小化
        """
        if flow_df.empty:
            return go.Figure()

        # 上位リンクのみ表示して可読性を向上
        flow_df = flow_df.sort_values('value', ascending=False).head(max_links)

        # ノードリスト作成
        countries = pd.unique(flow_df[['source', 'target']].values.ravel())
        country_to_index = {c: i for i, c in enumerate(countries)}

        # Sankey 用データ
        sources = flow_df['source'].map(country_to_index)
        targets = flow_df['target'].map(country_to_index)
        values = flow_df['value']

        # カラーマッピング（ソース国ごとに色分け）
        palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
        color_map = {c: palette[i % len(palette)] for i, c in enumerate(countries)}

        link_colors = [color_map[src] for src in flow_df['source']]
        node_colors = [color_map[c] for c in countries]

        # ---------------- ノード位置をグループ化して設定 -----------------
        # 各国の総IN/OUT 金額を算出
        total_out = flow_df.groupby('source')['value'].sum()
        total_in = flow_df.groupby('target')['value'].sum()
        net_flow = {c: total_out.get(c, 0) - total_in.get(c, 0) for c in countries}

        sources_group = [c for c in countries if net_flow[c] > 0]
        sinks_group   = [c for c in countries if net_flow[c] < 0]
        neutral_group = [c for c in countries if net_flow[c] == 0]

        def _assign_y(group):
            n = len(group)
            if n == 0:
                return {}
            return {c: (i+1)/(n+1) for i, c in enumerate(sorted(group))}

        y_positions = {}
        y_positions.update(_assign_y(sources_group))
        y_positions.update(_assign_y(neutral_group))
        y_positions.update(_assign_y(sinks_group))

        x_positions = {c: 0.0 for c in sources_group}
        x_positions.update({c: 0.5 for c in neutral_group})
        x_positions.update({c: 1.0 for c in sinks_group})

        node_x = [x_positions[c] for c in countries]
        node_y = [y_positions[c] for c in countries]

        fig = go.Figure(data=[go.Sankey(
            arrangement="fixed",
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=countries,
                color=node_colors,
                x=node_x,
                y=node_y
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                hovertemplate="%{source.label} → %{target.label}<br>金額: %{value:,.0f}<extra></extra>"
            )
        )])

        fig.update_layout(title_text="国間資金フロー", font_size=10, height=600)
        return fig

    @staticmethod
    def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, orientation: str = 'v') -> go.Figure:
        """汎用バーグラフ作成関数"""
        if df.empty:
            return go.Figure()

        fig = px.bar(
            df,
            x=x_col if orientation == 'v' else y_col,
            y=y_col if orientation == 'v' else x_col,
            orientation=orientation,
            text_auto='.2s',
            title=title,
        )
        fig.update_layout(height=400, showlegend=False)
        return fig

