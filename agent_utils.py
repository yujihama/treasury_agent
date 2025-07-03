import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import io
import base64
import re
import contextlib
from matplotlib import font_manager as fm  # 追加: フォント管理
import numpy as np  # 追加: 回帰分析用

# --- 日本語フォント設定 --------------------------------------------------
# DejaVu Sans では一部の日本語グリフが欠落しているため、システムに存在する
# 日本語フォントへ自動的に切り替える。最初に見つかったフォントを採用する。

# 候補フォント（優先順）
_JP_FONT_CANDIDATES = [
    "Yu Gothic", "YuGothic", "MS Gothic", "MS UI Gothic",
    "Meiryo", "Noto Sans CJK JP", "Noto Sans JP"
]

def _set_default_japanese_font():
    """利用可能な日本語フォントを Matplotlib のデフォルトに設定する"""
    try:
        available_fonts = {f.name for f in fm.fontManager.ttflist}
        for font in _JP_FONT_CANDIDATES:
            if font in available_fonts:
                plt.rcParams["font.family"] = font
                # マイナス記号を正しく表示
                plt.rcParams["axes.unicode_minus"] = False
                break
    except Exception:
        # フォント設定失敗時は既定フォントのまま進める
        pass

# モジュール読み込み時に一度だけ設定
_set_default_japanese_font()
# ----------------------------------------------------------------------

class TreasuryAgent:
    """トレジャリーマネジメント用のLangChainエージェント"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        self.llm = None
        self.agent_executor = None
        self.dataframes = {}
        
        if openai_api_key:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """LLMを初期化"""
        try:
            self.llm = ChatOpenAI(
                model="gpt-4.1-mini",
                temperature=0,
                openai_api_key=self.openai_api_key
            )
        except Exception as e:
            st.error(f"LLMの初期化に失敗しました: {e}")
            self.llm = None
    
    def set_dataframes(self, 
                      df_balances: pd.DataFrame = None,
                      df_transactions: pd.DataFrame = None,
                      df_budgets: pd.DataFrame = None,
                      df_fx_rates: pd.DataFrame = None):
        """データフレームを設定"""
        self.dataframes = {}
        
        if df_balances is not None and not df_balances.empty:
            self.dataframes['balances'] = df_balances
        if df_transactions is not None and not df_transactions.empty:
            self.dataframes['transactions'] = df_transactions
        if df_budgets is not None and not df_budgets.empty:
            self.dataframes['budgets'] = df_budgets
        if df_fx_rates is not None and not df_fx_rates.empty:
            self.dataframes['fx_rates'] = df_fx_rates
        
        # エージェントを再初期化
        if self.llm:
            self._initialize_agent()
    
    def _initialize_agent(self):
        """エージェントを初期化"""
        if not self.llm or not self.dataframes:
            return
        
        try:
            # ------------------------------------------------------------
            # すべての DataFrame をリストにまとめ、Pandas エージェントへ渡す
            # ------------------------------------------------------------
            dfs: list[pd.DataFrame] = []  # create_pandas_dataframe_agent へ渡す DF のリスト

            main_df: Optional[pd.DataFrame] = None  # 日付ソートなどの代表 DF
            df_info: list[str] = []  # プレフィックスに挿入するデータ概要

            # balances
            if 'balances' in self.dataframes:
                df_balances = self.dataframes['balances'].copy()
                dfs.append(df_balances)
                if main_df is None:
                    main_df = df_balances

                df_info.append("Balance Data(df): Date, Account ID, Bank Name, Currency, Balance, Country, Country Code")

                # データの詳細情報を追加
                countries = df_balances['国名'].unique() if '国名' in df_balances.columns else []
                df_info.append(f"Available Countries: {', '.join(countries)}")
                df_info.append(f"Total Data Count: {len(df_balances)}")

                # 国別データ数を追加
                if '国名' in df_balances.columns:
                    country_counts = df_balances['国名'].value_counts()
                    country_info = [f"{country}: {count}件" for country, count in country_counts.items()]
                    df_info.append(f"Country-wise Data Count: {', '.join(country_info)}")

            # transactions
            if 'transactions' in self.dataframes:
                df_transactions = self.dataframes['transactions'].copy()
                dfs.append(df_transactions)
                if main_df is None:
                    main_df = df_transactions

                df_info.append("Transaction Data(df1): Date, Account ID, Transaction ID, Transaction Type, Amount, Summary, Counterpart Country, Counterpart Country Code")

            # budgets
            if 'budgets' in self.dataframes:
                df_budgets = self.dataframes['budgets'].copy()
                dfs.append(df_budgets)
                df_info.append("Budget Data(df2): Date, Account ID, Budget Amount, Currency, Country, Country Code")

            # fx_rates
            if 'fx_rates' in self.dataframes:
                df_fx = self.dataframes['fx_rates'].copy()
                dfs.append(df_fx)
                df_info.append("FX Rate Data(df3): Date, Base Currency, Quote Currency, Rate")

            # データが 1 件も無い場合はエラー
            if not dfs:
                st.error("No data available for analysis")
                return

            # 代表 DF を用いて必要な前処理（例: 日付ソート）
            if main_df is not None and '日付' in main_df.columns:
                main_df = main_df.sort_values('日付').reset_index(drop=True)
            
            # 拡張されたプレフィックスを作成
            prefix = self._create_enhanced_prefix(df_info)
            
            # Pandasエージェントを作成（複数 DataFrame をリストで渡す）
            # df, df1, df2... のように自動で変数名が割り当てられる
            self.agent_executor = create_pandas_dataframe_agent(
                self.llm,
                dfs,  # すべての DF をリストで渡す
                verbose=True,
                allow_dangerous_code=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                prefix=prefix
            )
            
        except Exception as e:
            st.error(f"Failed to initialize the agent: {e}")
            self.agent_executor = None
    
    def query(self, user_input: str) -> str:
        """ユーザーの質問に回答"""
        if not self.agent_executor:
            return "The agent is not initialized. Please set the OpenAI API key and load the data."
        
        try:
            # セッション状態でグラフを管理
            if 'generated_plots' not in st.session_state:
                st.session_state.generated_plots = []

            # --- 追加: コンソール出力をキャプチャ ---
            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer):
                response = self.agent_executor.invoke({"input": user_input})
            console_output = stdout_buffer.getvalue()
            # ----------------------------------------

            # --- ANSIエスケープシーケンスを除去して見やすいログに変換 ---
            def _strip_ansi_codes(text: str) -> str:
                """ANSIカラーコードを除去したプレーンテキストを返す"""
                ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
                return ansi_escape.sub("", text)

            cleaned_output = _strip_ansi_codes(console_output)
            # -------------------------------------------------------------

            output = response.get("output", "Failed to generate a response.")

            # matplotlibのグラフをキャプチャして表示
            self._capture_and_display_plots()

            # キャプチャしたログをStreamlitで表示
            if cleaned_output:
                with st.expander("詳細ログ", expanded=False):
                    st.code(cleaned_output, language="bash")

            return output
        except Exception as e:
            error_msg = f"Agent execution error: {str(e)}"
            st.error(error_msg)
            return error_msg
    
    def _capture_and_display_plots(self):
        """生成されたmatplotlibプロットをキャプチャしてStreamlitで表示"""
        try:
            # 現在のすべてのfigureを取得
            figures = plt.get_fignums()
            
            for fig_num in figures:
                fig = plt.figure(fig_num)
                
                # figureが空でないかチェック
                if fig.axes:
                    # StreamlitでMatplotlibグラフを表示
                    st.pyplot(fig, clear_figure=True)
                    
        except Exception as e:
            st.error(f"Graph display error: {str(e)}")
    
    def _create_enhanced_prefix(self, df_info: list) -> str:
        """拡張されたプレフィックスを作成（グラフ生成指示を含む）"""
        return f"""
        あなたは企業のトレジャリーマネジメントを支援するAIアシスタントです。
        
        利用可能なデータ:
        {chr(10).join(df_info)}
        
        これらのデータを使って、ユーザーの質問に答えてください。
        
        グラフや可視化が必要な場合は、以下のガイドラインに従ってください：
        1. matplotlib.pyplotを使用してグラフを作成してください
        2. plt.figure(figsize=(10, 6))でサイズを指定してください
        3. 日本語のタイトルと軸ラベルを設定してください
        4. plt.show()を呼び出してグラフを表示してください
        5. グラフの説明も含めて回答してください
        
        例：
        ```python
        import matplotlib.pyplot as plt
        
        # まず利用可能なデータの構造を確認
        print("利用可能なデータの構造:", df.columns)

        # まず利用可能な国を確認
        print("利用可能な国:", df['国名'].unique())
        
        # 日本以外のデータをフィルタリング
        non_japan_data = df[df['国名'] != '日本']
        
        # グラフ作成
        plt.figure(figsize=(10, 6))
        plt.bar(data.index, data.values)
        plt.title('データの可視化')
        plt.xlabel('項目')
        plt.ylabel('値')
        plt.show()
        ```
        
        最終的な回答は日本語で分かりやすく説明してください。
        """
    
    def is_ready(self) -> bool:
        """エージェントが使用可能かチェック"""
        return self.agent_executor is not None and bool(self.dataframes)

# 非LLM版の分析機能（APIキーが無い場合のフォールバック）
class BasicAnalyzer:
    """基本的な分析機能（LLMを使わない）"""
    
    @staticmethod
    def analyze_balances(df_balances: pd.DataFrame) -> str:
        """口座残高の基本分析"""
        if df_balances.empty:
            return "口座残高データがありません。"
        
        from data_utils import DataProcessor
        latest_balances = DataProcessor.get_latest_balances(df_balances)
        
        total_balance = latest_balances['残高'].sum()
        account_count = len(latest_balances)
        avg_balance = latest_balances['残高'].mean()
        
        # 国別サマリー
        if '国名' in latest_balances.columns:
            country_summary = latest_balances.groupby('国名')['残高'].sum().sort_values(ascending=False)
            top_country = country_summary.index[0] if len(country_summary) > 0 else "不明"
            top_country_balance = country_summary.iloc[0] if len(country_summary) > 0 else 0
        else:
            top_country = "不明"
            top_country_balance = 0
        
        analysis = f"""
        **口座残高分析結果**
        
        - 総残高: {total_balance:,.0f}
        - 口座数: {account_count}件
        - 平均残高: {avg_balance:,.0f}
        - 最大残高国: {top_country} ({top_country_balance:,.0f})
        """
        
        return analysis
    
    @staticmethod
    def analyze_transactions(df_transactions: pd.DataFrame) -> str:
        """取引履歴の基本分析"""
        if df_transactions.empty:
            return "取引履歴データがありません。"
        
        total_transactions = len(df_transactions)
        total_amount = df_transactions['金額'].sum()
        avg_amount = df_transactions['金額'].mean()
        
        # 取引タイプ別分析
        type_summary = df_transactions.groupby('取引タイプ').agg({
            '金額': ['sum', 'count']
        }).round(2)
        
        # 最も多い取引タイプ
        most_frequent_type = df_transactions['取引タイプ'].value_counts().index[0]
        most_frequent_count = df_transactions['取引タイプ'].value_counts().iloc[0]
        
        analysis = f"""
        **取引履歴分析結果**
        
        - 総取引件数: {total_transactions}件
        - 総取引金額: {total_amount:,.0f}
        - 平均取引金額: {avg_amount:,.0f}
        - 最頻取引タイプ: {most_frequent_type} ({most_frequent_count}件)
        """
        
        return analysis

    @staticmethod
    def analyze_balance_trend(df_balances: pd.DataFrame):
        """残高の時系列トレンドを単回帰で分析し、結果テキストと図を返す"""
        from data_utils import Visualizer

        if df_balances.empty or '日付' not in df_balances.columns:
            return "残高トレンドを計算するためのデータが不足しています。", None

        # 日付ごとの総残高を集計
        daily_total = (
            df_balances.groupby('日付')['残高'].sum()
            .reset_index()
            .sort_values('日付')
            .rename(columns={'残高': '総残高'})
        )

        if len(daily_total) < 2:
            return "残高トレンドを計算するための十分なデータポイントがありません。", None

        # 回帰分析
        x = (daily_total['日付'] - daily_total['日付'].min()).dt.days.astype(float)
        y = daily_total['総残高']

        slope, intercept = np.polyfit(x, y, 1)

        # R² を計算
        y_pred = intercept + slope * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        analysis_text = f"""
        **残高トレンド回帰分析**
        
        - 回帰式: 残高 = {intercept:,.0f} + {slope:,.0f} × 経過日数
        - 日次平均増減: {slope:,.0f}
        - 月次（30日換算）平均増減: {slope * 30:,.0f}
        - 決定係数 (R²): {r2:.3f}
        """

        # グラフ生成
        fig = Visualizer.create_balance_trend_chart(daily_total, slope, intercept)

        return analysis_text, fig

    @staticmethod
    def analyze_country_flows(df_balances: pd.DataFrame, df_transactions: pd.DataFrame, 
                             source_filter: list[str] | None = None,
                             target_filter: list[str] | None = None):
        """国間資金フローを集計し、Sankey 図とサマリーを返す"""
        from data_utils import DataProcessor, Visualizer

        if df_transactions.empty or '相手国名' not in df_transactions.columns:
            return "国間フローを分析するためのデータが不足しています。", None

        # 口座ID=>国名 のマッピング（最新残高データを利用）
        if df_balances.empty or '国名' not in df_balances.columns:
            return "口座の国情報が不足しているためフロー分析を行えません。", None

        latest_balances = DataProcessor.get_latest_balances(df_balances)
        account_to_country = latest_balances.set_index('口座ID')['国名'].to_dict()

        # トランザクションに送金元・送金先を付与
        tx = df_transactions.copy()
        tx['送金元国'] = tx['口座ID'].map(account_to_country).fillna('Unknown')
        tx['送金先国'] = tx['相手国名'].fillna('Unknown')

        # 送金方向判定
        # 金額<0: 送金元 -> 送金先 (資金流出)
        # 金額>0: 送金先 -> 送金元 (資金流入)
        outflow = tx[tx['金額'] < 0].copy()
        outflow['flow_source'] = outflow['送金元国']
        outflow['flow_target'] = outflow['送金先国']
        outflow['flow_value'] = outflow['金額'].abs()

        inflow = tx[tx['金額'] > 0].copy()
        inflow['flow_source'] = inflow['送金先国']
        inflow['flow_target'] = inflow['送金元国']
        inflow['flow_value'] = inflow['金額']

        flow_df = (
            pd.concat([outflow[['flow_source', 'flow_target', 'flow_value']],
                       inflow[['flow_source', 'flow_target', 'flow_value']]])
            .groupby(['flow_source', 'flow_target'], as_index=False)['flow_value'].sum()
            .rename(columns={'flow_source': 'source', 'flow_target': 'target', 'flow_value': 'value'})
        )

        # ---------------- フィルタリング -----------------
        if source_filter:
            flow_df = flow_df[flow_df['source'].isin(source_filter)]
        if target_filter:
            flow_df = flow_df[flow_df['target'].isin(target_filter)]

        # Sankey 図
        sankey_fig = Visualizer.create_country_flow_sankey(flow_df, max_links=50)

        # サマリー
        country_totals = flow_df.groupby('source')['value'].sum().sort_values(ascending=False).head(10)
        summary_lines = [
            f"{country}: {val:,.0f}" for country, val in country_totals.items()
        ]
        summary_text = "\n".join(summary_lines)

        analysis_text = f""

        return analysis_text, sankey_fig

