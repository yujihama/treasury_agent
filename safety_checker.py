"""
生成されたコードの安全性をチェックするモジュール
"""
import re
import ast
import traceback
from typing import List, Tuple, Dict, Any

class SafetyChecker:
    """
    生成されたPythonコードの安全性をチェックするクラス
    """
    
    # 禁止されたパターンの正規表現
    FORBIDDEN_PATTERNS = [
        r'\bos\b',
        r'\bsubprocess\b',
        r'\bopen\s*\(',
        r'\bshutil\b',
        r'\beval\s*\(',
        r'\bexec\s*\(',
        r'\bsocket\b',
        r'\brequests\b',
        r'\burllib\b',
        r'\bhttp\b',
        r'\b__import__\b',
        r'\bgetattr\b',
        r'\bsetattr\b',
        r'\bdelattr\b',
        r'\bhasattr\b',
        r'\bglobals\b',
        r'\blocals\b',
        r'\bvars\b',
        r'\bdir\b',
        r'\bfile\b',
        r'\binput\b',
        r'\braw_input\b',
    ]
    
    # 許可されたインポート
    ALLOWED_IMPORTS = {
        'pandas', 'pd',
        'numpy', 'np', 
        'plotly', 'plotly.express', 'plotly.graph_objects',
        'streamlit', 'st',
        'datetime',
        'math',
        'statistics',
        'collections',
        'matplotlib', 'plt', 'matplotlib.pyplot',
        'seaborn', 'sns', 'altair', 'vega_datasets', 'bokeh', 'holoviews', 'hvplot', 'pydeck', 'pyecharts',
        'folium', 'geopandas', 'sklearn', 'scikit_learn', 
        'PIL', 'Pillow', 'Image', 'ImageDraw', 'ImageFont',
    }
    
    def __init__(self):
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.FORBIDDEN_PATTERNS]
    
    def check_forbidden_patterns(self, code: str) -> Tuple[bool, List[str]]:
        """
        禁止されたパターンをチェック
        
        Args:
            code: チェック対象のコード
            
        Returns:
            (is_safe, violations): 安全かどうかと違反リスト
        """
        violations = []
        
        for pattern in self.compiled_patterns:
            matches = pattern.findall(code)
            if matches:
                violations.extend([f"Forbidden pattern found: {match}" for match in matches])
        
        return len(violations) == 0, violations
    
    def check_ast_imports(self, code: str) -> Tuple[bool, List[str]]:
        """
        ASTを使ってインポートをチェック
        
        Args:
            code: チェック対象のコード
            
        Returns:
            (is_safe, violations): 安全かどうかと違反リスト
        """
        violations = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, [f"Syntax error: {str(e)}"]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.ALLOWED_IMPORTS:
                        violations.append(f"Forbidden import: {alias.name}")
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module not in self.ALLOWED_IMPORTS:
                    # モジュール名の部分チェック
                    module_parts = node.module.split('.')
                    allowed = False
                    for allowed_module in self.ALLOWED_IMPORTS:
                        if node.module.startswith(allowed_module):
                            allowed = True
                            break
                    
                    if not allowed:
                        violations.append(f"Forbidden import from: {node.module}")
        
        return len(violations) == 0, violations
    
    def check_dangerous_functions(self, code: str) -> Tuple[bool, List[str]]:
        """
        危険な関数呼び出しをチェック
        
        Args:
            code: チェック対象のコード
            
        Returns:
            (is_safe, violations): 安全かどうかと違反リスト
        """
        violations = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return True, []  # 構文エラーは別でチェック済み
        
        dangerous_functions = {
            'eval', 'exec', 'compile', '__import__',
            'getattr', 'setattr', 'delattr', 'hasattr',
            'globals', 'locals', 'vars', 'dir'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in dangerous_functions:
                        violations.append(f"Dangerous function call: {node.func.id}")
        
        return len(violations) == 0, violations
    
    def is_safe(self, code: str) -> Tuple[bool, Dict[str, Any]]:
        """
        コードの総合的な安全性チェック
        
        Args:
            code: チェック対象のコード
            
        Returns:
            (is_safe, report): 安全かどうかと詳細レポート
        """
        report = {
            'forbidden_patterns': {'safe': True, 'violations': []},
            'ast_imports': {'safe': True, 'violations': []},
            'dangerous_functions': {'safe': True, 'violations': []},
            'overall_safe': True
        }
        
        # 禁止パターンチェック
        safe1, violations1 = self.check_forbidden_patterns(code)
        report['forbidden_patterns'] = {'safe': safe1, 'violations': violations1}
        
        # ASTインポートチェック
        safe2, violations2 = self.check_ast_imports(code)
        report['ast_imports'] = {'safe': safe2, 'violations': violations2}
        
        # 危険な関数チェック
        safe3, violations3 = self.check_dangerous_functions(code)
        report['dangerous_functions'] = {'safe': safe3, 'violations': violations3}
        
        # 総合判定
        overall_safe = safe1 and safe2 and safe3
        report['overall_safe'] = overall_safe
        
        return overall_safe, report

def test_safety_checker():
    """
    SafetyCheckerのテスト
    """
    checker = SafetyChecker()
    
    # 安全なコード
    safe_code = """
import pandas as pd
import plotly.express as px
import streamlit as st

df_filtered = df[df['category'] == 'A']
fig = px.scatter(df_filtered, x='x', y='y')
st.plotly_chart(fig)
"""
    
    # 危険なコード
    dangerous_code = """
import os
import subprocess
os.system('rm -rf /')
"""
    
    print("Testing safe code:")
    is_safe, report = checker.is_safe(safe_code)
    print(f"Safe: {is_safe}")
    print(f"Report: {report}")
    
    print("\nTesting dangerous code:")
    is_safe, report = checker.is_safe(dangerous_code)
    print(f"Safe: {is_safe}")
    print(f"Report: {report}")

if __name__ == "__main__":
    test_safety_checker()

