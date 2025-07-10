"""
LLMクライアント機能
"""
import time
import logging
from typing import Optional, Dict, Any
import streamlit as st

class LLMClient:
    """
    LLMとの通信を行うクライアントクラス
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1"):
        """
        初期化
        
        Args:
            api_key: OpenAI APIキー
            model: 使用するモデル名
        """
        self.api_key = api_key
        self.model = model
        # モックモードを廃止し、APIキーが必須となるように変更
        if api_key is None:
            st.error("OpenAI APIキーが設定されていません。サイドバーから設定してください。")
            raise ValueError("OpenAI APIキーが必要です。")

        try:
            # LangChain ChatOpenAI を使用
            from langchain_openai import ChatOpenAI

            # ChatOpenAI インスタンスを作成
            self.llm = ChatOpenAI(
                api_key=api_key,
                model_name=model,
                temperature=0,
            )

        except ImportError as e:
            st.error("LangChain (langchain_openai) がインストールされていません。")
            raise e
    
    def generate_code(self, user_prompt: str, df_info: str) -> str:
        """
        ユーザープロンプトからStreamlitコードを生成
        
        Args:
            user_prompt: ユーザーの自然言語プロンプト
            df_info: データフレームの情報
            
        Returns:
            生成されたPythonコード
        """
        raw_code = self._generate_real_code(user_prompt, df_info)
        
        # マークダウンコードブロック記号を除去
        cleaned_code = self.clean_generated_code(raw_code)
        return cleaned_code
    
    def clean_generated_code(self, raw_code: str) -> str:
        """
        生成されたコードからマークダウンコードブロック記号を除去
        
        Args:
            raw_code: 生成された生のコード文字列
            
        Returns:
            クリーニングされたコード文字列
        """
        # マークダウンコードブロックの開始・終了記号を除去
        code = raw_code.strip()
        
        # 開始記号の除去（```python, ```py, ``` など）
        if code.startswith('```'):
            lines = code.split('\n')
            # 最初の行がコードブロック開始記号の場合は除去
            if lines[0].strip().startswith('```'):
                lines = lines[1:]
            code = '\n'.join(lines)
        
        # 終了記号の除去
        if code.endswith('```'):
            lines = code.split('\n')
            # 最後の行がコードブロック終了記号の場合は除去
            if lines[-1].strip() == '```':
                lines = lines[:-1]
            code = '\n'.join(lines)
        
        # 先頭と末尾の空白を除去
        code = code.strip()
        
        return code
    
    def _generate_real_code(self, user_prompt: str, df_info: str) -> str:
        """
        実際のLLMを使ったコード生成
        
        Args:
            user_prompt: ユーザープロンプト
            df_info: データフレーム情報
            
        Returns:
            生成されたコード
        """
        system_prompt = f"""
あなたはStreamlitアプリ用のPythonコードを生成するアシスタントです。

以下の制約に従ってコードを生成してください：

1. 使用可能なライブラリ：
   - pandas (pd)
   - numpy (np)
   - streamlit (st, st.markdown, st.dataframe, st.bar_chart, st.scatter_chart, st.plotly_chart, st.altair_chart, st.bokeh_chart, st.holoviews_chart, st.hvplot_chart, st.pydeck_chart, st.pyecharts_chart, st.folium_chart, st.geopandas_chart, st.sklearn_chart, st.PIL_chart)
   - plotly (plotly.express, plotly.graph_objects)
   - datetime
   - math
   - statistics
   - collections
   - matplotlib (plt, matplotlib.pyplot)
   - seaborn (sns)
   - altair
   - vega_datasets
   - bokeh
   - holoviews
   - hvplot
   - pydeck
   - pyecharts
   - folium
   - geopandas
   - sklearn (scikit_learn)
   - PIL (Pillow)

2. データフレーム：
   - 変数名はデータフレーム情報をもとに正確に使用してください。

3. 出力形式：
   - streamlitで描画可能なコード(st.から始まるメソッド)を使用
   - データフレームにあるカラムで可能な限りフィルタなどインタラクティブな操作ができるようにする
   - 適切なタイトルを設定
   - 表示のタイトル等はmarkdownで表示する。トップレベルはst.markdown("##")、サブレベルはst.markdown("###")
   - st.sidebarは禁止
   
4. 禁止事項：
   - ファイル操作
   - ネットワーク通信
   - システム操作
   - 外部ライブラリのインポート

5. コード形式：
   - インポート文から開始
   - コメントを適切に追加
   - デバッグのため、小まめにprint文で処理内容と処理結果を出力すること

ユーザーの要求に応じて、適切な可視化コードを生成してください。
回答はコードのみで、コメントは不要です。
"""
        
        try:
            # --- LangChain Structured Output Parser 設定 ---
            from langchain.output_parsers import StructuredOutputParser, ResponseSchema

            # 出力スキーマ定義
            response_schemas = [
                ResponseSchema(name="code", description="Streamlit 用 Python コード (コメント不要)。"),
                ResponseSchema(name="commentary", description="コードの簡潔な日本語説明。")
            ]

            parser = StructuredOutputParser.from_response_schemas(response_schemas)

            format_instructions = parser.get_format_instructions()

            # プロンプト組み立て
            full_prompt = f"""{system_prompt}

# 出力フォーマット
{format_instructions}

ユーザーの依頼: {user_prompt['task']}

入力データフレーム: {df_info}

※注意
- 依頼の意図をよく理解し、専門家に向けたより包括的なビジュアライズを心がけてください。
"""

            start_time = time.time()

            response = self.llm.invoke(full_prompt)

            end_time = time.time()

            # ログ出力
            logging.info(f"LLM call completed in {end_time - start_time:.2f}s")

            # 構造化出力のパース
            try:
                parsed = parser.parse(response.content)
                generated_code = parsed.get("code", "")
                if not generated_code:
                    raise ValueError("'code' フィールドが空です")
            except Exception as parse_err:
                logging.warning(f"構造化出力の解析に失敗しました: {parse_err}. 生の出力を使用します。")
                generated_code = response.content

            logging.info(f"Generated code length: {len(generated_code)}")

            return generated_code.strip()

        except Exception as e:
            logging.error(f"LLM call failed: {str(e)}")
            raise

    # ----------------------------------------------------
    # コード修正用メソッド
    # ----------------------------------------------------
    def fix_code(self, original_code: str, error_message: str, user_prompt: Any, df_info: str) -> str:
        """
        既存のコードとエラーメッセージをもとに、LLM に修正版コードを生成してもらう。

        Args:
            original_code: 元の Python コード
            error_message: 構文エラーや安全性チェックの詳細
            user_prompt: ユーザー依頼内容（dict もしくは str）
            df_info: データフレーム情報文字列

        Returns:
            修正後のコード文字列（マークダウン除去済み）
        """
        raw_code = self._generate_fixed_code(original_code, error_message, user_prompt, df_info)
        cleaned_code = self.clean_generated_code(raw_code)
        return cleaned_code

    def _generate_fixed_code(self, original_code: str, error_message: str, user_prompt: Any, df_info: str) -> str:
        """
        実際に LLM へ修正依頼を投げる内部メソッド。
        """
        # user_prompt が dict の場合は task フィールドを利用する
        try:
            prompt_text = user_prompt["task"] if isinstance(user_prompt, dict) else str(user_prompt)
        except Exception:
            prompt_text = str(user_prompt)

        system_prompt = """
あなたは熟練した Python/Streamlit エンジニアです。以下の元コードは自動生成時に安全性チェックもしくは構文チェックで失敗しました。エラーメッセージを参考に、安全で実行可能かつ構文エラーのないコードに修正してください。

# 制約条件
1. ファイル操作・ネットワーク操作・システム操作は禁止。
2. import 可能なライブラリは pandas/numpy/streamlit/plotly 等、generate_code と同様。
3. 構文エラーがないこと。
4. 安全性チェック（Forbidden Patterns, AST Import 制限など）を通過すること。
5. 出力はコードのみ（コメント不要）。マークダウンの ``` は含めないでください。
"""

        full_prompt = f"""{system_prompt}

## ユーザーの依頼内容
{prompt_text}

## データフレーム情報
{df_info}

## 元コード
```python
{original_code}
```

## エラーメッセージ / 安全性レポート
{error_message}

修正済みのコードを提示してください。コメントは不要です。"""

        # LLM 呼び出し
        start_time = time.time()
        response = self.llm.invoke(full_prompt)
        end_time = time.time()
        logging.info(f"LLM fix_code call completed in {end_time - start_time:.2f}s")

        return response.content.strip()
    