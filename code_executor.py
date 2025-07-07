"""
安全な環境でコードを実行するモジュール
"""
import io
import sys
import traceback
import contextlib
import ast  # 構文チェック用
from typing import Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

class CodeExecutor:
    """
    安全な環境でPythonコードを実行するクラス
    """
    
    def __init__(self, dataframes: Union[pd.DataFrame, Dict[str, pd.DataFrame]]):
        """
        初期化
        
        Args:
            dataframes: 単一の ``pandas.DataFrame`` もしくは
                ``{name: DataFrame}`` 形式で渡される複数 DataFrame
        """
        # 単一 DataFrame が渡された場合は ``df`` という名前でラップして扱う
        if isinstance(dataframes, pd.DataFrame):
            self.initial_df_dict: Dict[str, pd.DataFrame] = {"df": dataframes}
        else:
            self.initial_df_dict = dataframes

        self.setup_safe_globals()
    
    def setup_safe_globals(self) -> Dict[str, Any]:
        """
        安全なグローバル環境を設定
        
        Returns:
            安全なグローバル辞書
        """
        # 制限された組み込み関数
        safe_builtins = {
            'len': len,
            'range': range,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'print': print,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'enumerate': enumerate,
            'zip': zip,
            'type': type,
            'isinstance': isinstance,
            '__import__': __import__,  # インポートを許可
        }
        
        self.safe_globals = {
            '__builtins__': safe_builtins,
            'pd': pd,
            'np': np,
            'px': px,
            'go': go,
            'st': st,
            # モジュールを直接利用可能にする
            'pandas': pd,
            'numpy': np,
            'plotly': {'express': px, 'graph_objects': go},
            'streamlit': st,
        }
        
        # 受け取った各 DataFrame をグローバル変数として登録
        for name, df in self.initial_df_dict.items():
            self.safe_globals[name] = df

        # 後方互換のため ``df`` という名前を必ず提供
        if 'df' not in self.safe_globals and self.initial_df_dict:
            first_key = next(iter(self.initial_df_dict))
            self.safe_globals['df'] = self.initial_df_dict[first_key]
        
        return self.safe_globals
    
    @contextlib.contextmanager
    def capture_stdout(self):
        """
        標準出力をキャプチャするコンテキストマネージャー
        """
        old_stdout = sys.stdout
        stdout_capture = io.StringIO()
        try:
            sys.stdout = stdout_capture
            yield stdout_capture
        finally:
            sys.stdout = old_stdout
    
    def execute_code(self, code: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        コードを安全な環境で実行
        
        Args:
            code: 実行するPythonコード
            
        Returns:
            (success, stdout_output, error_message): 実行結果
        """
        # まず構文チェックを実施
        syntax_ok, syntax_error = self._check_syntax(code)
        if not syntax_ok:
            # 構文エラーの場合は実行せずにエラーを返す
            return False, None, syntax_error

        try:
            with self.capture_stdout() as stdout_capture:
                # コードを実行
                exec(code, self.safe_globals)
                
                # 標準出力を取得
                stdout_output = stdout_capture.getvalue()
                
            return True, stdout_output if stdout_output else None, None
            
        except Exception as e:
            error_message = f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"
            return False, None, error_message
    
    def _check_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        ``ast`` を用いてコードの構文を事前に検証する。

        Args:
            code: 検証対象の Python コード文字列

        Returns:
            (is_valid, error_message): 構文が正しいかどうかとエラーメッセージ
        """
        try:
            # ``compile`` でコード全体を構文チェックする
            compile(code, "<generated_code>", "exec")
            return True, None
        except SyntaxError as e:
            # ``traceback`` を付与して詳細を返す
            error_message = f"SyntaxError: {str(e)}\n\n{traceback.format_exc()}"
            return False, error_message
    
    def test_execution(self) -> None:
        """
        実行機能のテスト
        """
        # テストコード1: 基本的なプロット
        test_code1 = """
import plotly.express as px
fig = px.scatter(df, x='x', y='y', color='category')
st.plotly_chart(fig)
print("Scatter plot created successfully")
"""
        
        # テストコード2: データフレーム表示
        test_code2 = """
st.dataframe(df.head())
print(f"DataFrame shape: {df.shape}")
"""
        
        # テストコード3: エラーを含むコード
        test_code3 = """
undefined_variable + 1
"""
        
        print("Testing code execution:")
        
        for i, code in enumerate([test_code1, test_code2, test_code3], 1):
            print(f"\n--- Test {i} ---")
            success, stdout, error = self.execute_code(code)
            print(f"Success: {success}")
            if stdout:
                print(f"Stdout: {stdout}")
            if error:
                print(f"Error: {error}")

def test_code_executor():
    """
    CodeExecutorのテスト
    """
    # サンプルデータを作成
    from sample_data import create_simple_data
    df = create_simple_data()
    
    # エグゼキューターを作成してテスト
    executor = CodeExecutor(df)
    executor.test_execution()

if __name__ == "__main__":
    test_code_executor()

