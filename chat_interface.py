# """
# LLM × Streamlit インタラクティブ可視化アプリ
# """
# import streamlit as st
# import pandas as pd
# import logging
# import traceback
# import json
# from datetime import datetime
# from typing import Optional
# from langchain_openai import ChatOpenAI
# # from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
# from langchain_core.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field
# import ast
# # 文字列解析用
# import io
# import tokenize
# from tokenize import TokenInfo
# from langchain_core.prompts import PromptTemplate

# # 自作モジュールのインポート
# from safety_checker import SafetyChecker
# from code_executor import CodeExecutor
# from llm_client import LLMClient
# from pandas_dataframe_agent import create_pandas_dataframe_agent

# # ログ設定
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class Plan(BaseModel):
#     """各ステップの情報"""

#     category: str = Field(..., description="カテゴリ (prepare / visualize / report 等)")
#     task: str = Field(..., description="タスクの詳細")
#     input: list[str] = Field(..., description="入力データフレーム名のリスト")
#     output: str | None = Field(..., description="出力データフレーム名 (空文字列可)")
#     output_columns: list[str] | None = Field(..., description="出力データフレームのカラム名のリスト(出力データフレームがある場合のみ)")


# # ルートモデル (トップレベルがリスト)
# class ResponseFormatter(BaseModel):
#     """Plan の配列を格納するトップレベルオブジェクト"""
#     plans: list[Plan] = Field(..., description="Plan のリスト")

# def initialize_session_state(initial_df_dict: dict[str, pd.DataFrame]):
#     """
#     セッション状態の初期化
#     """
    
#     # 初期データフレームを dict 形式 {name: DataFrame} で保持
#     if 'initial_df_dict' not in st.session_state:
#         st.session_state.initial_df_dict = {
#             name: df
#             for name, df in initial_df_dict.items()
#         }
    
#     if 'safety_checker' not in st.session_state:
#         st.session_state.safety_checker = SafetyChecker()
    
#     if 'code_executor' not in st.session_state:
#         st.session_state.code_executor = CodeExecutor(st.session_state.initial_df_dict)
    
#     if 'api_key' not in st.session_state:
#         st.session_state.api_key = None
    
#     if 'llm_client' not in st.session_state:
#         st.session_state.llm_client = None
    
#     if 'execution_history' not in st.session_state:
#         st.session_state.execution_history = []
#     # visualize_plan で生成した複数コードを保持するリスト
#     if 'generated_codes' not in st.session_state:
#         st.session_state.generated_codes = []

# def setup_llm_client():
#     """
#     LLMクライアントの設定
#     """
#     # APIキーの取得を試行（優先順位：セッション状態 > シークレット）
#     api_key = st.session_state.api_key
#     if not api_key:
#         try:
#             api_key = st.secrets.get("OPENAI_API_KEY")
#         except Exception:
#             pass

#     # APIキーがある場合のみ LLM クライアントを初期化
#     if api_key:
#         if (
#             st.session_state.llm_client is None
#             or st.session_state.llm_client.api_key != api_key
#         ):
#             try:
#                 st.session_state.llm_client = LLMClient(api_key)
#             except Exception as e:
#                 st.error(f"LLMクライアントの初期化に失敗しました: {e}")
#                 st.session_state.llm_client = None
#     else:
#         st.session_state.llm_client = None
        
# def get_dataframe_info(df_list: list[dict[str, pd.DataFrame]]) -> str:
#     """
#     データフレームの情報を文字列で取得
    
#     Args:
#         df_list: データフレームのリスト
        
#     Returns:
#         データフレーム情報の文字列
#     """
#     info = ""
#     for df_info in df_list:
#         info += f"input_name: {df_info['input_name']}\n"
#         if isinstance(df_info['input_df'], pd.DataFrame):
#             info += f"Shape: {df_info['input_df'].shape}\n"
#             info += f"Columns: {', '.join(df_info['input_df'].columns)}\n"
#             info += f"Data types:\n"
#             for col, dtype in df_info['input_df'].dtypes.items():
#                 info += f"  {col}: {dtype}\n"
#         else:
#             info += f"len(input_df): {len(df_info['input_df'])}\n"
    
#     return info

# def log_execution(prompt: str, code: str, success: bool, error: Optional[str] = None):
#     """
#     実行ログを記録
    
#     Args:
#         prompt: ユーザープロンプト
#         code: 生成されたコード
#         success: 実行成功フラグ
#         error: エラーメッセージ（あれば）
#     """
#     log_entry = {
#         'timestamp': datetime.now(),
#         'prompt': prompt,
#         'code': code,
#         'success': success,
#         'error': error
#     }
    
#     st.session_state.execution_history.append(log_entry)
    
#     # ログ出力
#     logger.info(f"Execution logged: success={success}, prompt_length={len(prompt)}, code_length={len(code)}")
#     if error:
#         logger.error(f"Execution error: {error}")

# def replace_df_references(code: str, df_names: list[str]) -> str:
#     """
#     コメントおよびクオーテーションで囲まれた文字列内部を変更せず、
#     コード中の DataFrame 変数名を st.session_state 参照へ置換する。
#     """
#     df_set = set(df_names)
#     replace_map = {name: f"st.session_state.work_df_dict['{name}']" for name in df_set if name in st.session_state.work_df_dict}

#     result_tokens: list[TokenInfo] = []
#     sio = io.StringIO(code)

#     for tok in tokenize.generate_tokens(sio.readline):
#         if tok.type == tokenize.NAME and tok.string in replace_map.keys():
#             # 置換した文字列で新しい TokenInfo を生成
#             tok = TokenInfo(tok.type, replace_map[tok.string], tok.start, tok.end, tok.line)
#         result_tokens.append(tok)

#     return tokenize.untokenize(result_tokens)



# def main(initial_df_dict: dict[str, pd.DataFrame]):
#     """
#     メイン関数
#     """
#     st.set_page_config(
#         page_title="Streamlit Agent",
#         page_icon="",
#         layout="wide"
#     )

#     # --- 追加: ボタンのスタイルをダークブルーにカスタマイズ ---
#     st.markdown(
#         """
#         <style>
#             div.stButton > button[kind='primary'] {
#                 background-color: #004080;
#                 color: #ffffff;
#                 border: none;
#                 border-color: #004080;
#             }
#             div.stButton > button[kind='primary']:hover {
#                 background-color: #002d66;
#                 color: #ffffff;
#                 border: none;
#                 border-color: #002d66;
#             }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#     # --- 追加: タブ内コンテンツの背景色を変更 ---
#     st.markdown(
#         """
#         <style>
#             /* st.tabs() のタブパネル本体 (.block-container) にも色を付与 */
#             div[data-testid="stTabContent"] div[data-testid="stVerticalBlock"] {
#                 background-color: #f7fbff;  /* 任意の色に変更可能 */
#                 padding: 1rem;
#                 border-radius: 6px;
#             }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
    
#     # セッション状態の初期化
#     initialize_session_state(initial_df_dict=initial_df_dict)
    
#     # LLMクライアントの設定
#     setup_llm_client()
  
#     # プロンプト入力エリア
#     user_prompt = st.text_area(
#         "依頼内容を入力してください",
#         height=100,
#         placeholder="例: 売上と利益率の散布図を作成してください。",
#         key="user_prompt"
#     )

#     # 実行ボタン
#     run_button = st.button("実行", type="primary", use_container_width=True)
    
#     if run_button and user_prompt.strip():
#         if st.session_state.llm_client is None:
#             st.error("OpenAI APIキーを設定してください。サイドバーからAPIキーを入力すると実行できます。")
#         else:
#             try:
#                 st_callback = StreamlitCallbackHandler(st.container())
#                 # プログレスバー
#                 progress_bar = st.progress(0)
#                 status_text = st.empty()

#                 status_text.text("タスク生成中…")
#                 progress_bar.progress(10)

#                 # 複数 DF 情報をプロンプト用に作成
#                 df_overview_for_prompt = get_dataframe_info([
#                     {"input_name": name, "input_df": df}
#                     for name, df in st.session_state.initial_df_dict.items()
#                 ])

#                 prompt_for_plan = f"""
#                 あなたは経営者の意思決定をサポートする優秀なデータアナリストです。           
#                 以下の依頼内容の意図をよく考え、依頼内容を実現するための具体的なステップに細分化してください。
#                 # 依頼内容
#                 {user_prompt}

#                 # データの概要
#                 {df_overview_for_prompt}

#                 # 手順
#                 1. 依頼内容を理解する。背景にあるニーズも含めて理解する。
#                 2. データの概要を確認する
#                 3. どのようなデータ可視化を行うか検討する。多くのビジュアルを作成するのではなく、機能が凝縮された必要最小限のダッシュボードが望ましく、フィルタやしきい値設定ができるようにインタラクティブな操作ができるようにする。
#                 4. 依頼内容を実現するためにprepare、visualize、reportの3ステップのタスクに細分化する
#                 5. 細分化された各タスクについて以下の内容を回答する
#                     - category(prepare/visualize/report)
#                     - task(与えられたdfをインプットにどのような加工をするか明記)
#                     - input(複数ある場合はリスト)
#                     - output(単一のdataframe名※visualizeとreportの場合はスペース)
#                     - output_columns(出力するカラムのリスト※可能な限りinputのカラムを残す※出力データフレームがある場合のみ)
#                 6. prepareのタスクは、visualizeのタスクに必要な各dataframeごとに細分化してください。(1タスク1つのdataframeをoutputとして持つ)
#                 """

#                 # PromptTemplate を使って文字列から Runnable へ変換
#                 plan_prompt = PromptTemplate.from_template("{input}")
#                 plan_llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=st.session_state.api_key, callbacks=[st_callback], verbose=True).with_structured_output(ResponseFormatter)
#                 # Runnable でチェーンを組み立て
#                 plan_chain = plan_prompt | plan_llm
#                 response_plan = plan_chain.invoke(
#                     {"input": prompt_for_plan}, {"callbacks": [st_callback]}
#                 )
                
#                 if isinstance(response_plan, ResponseFormatter):
#                     plan = response_plan.plans
#                 else:
#                     plan = response_plan
                
#                 df_plan = pd.DataFrame([
#                     p.model_dump() if isinstance(p, Plan) else p for p in plan
#                 ])
#                 df_plan['status'] = '⬜'
#                 df_plan = df_plan.reindex(columns=['category', 'status', 'task', 'input', 'output', 'output_columns'])
#                 st.session_state.plan = df_plan
#                 st.session_state.generated_codes = []
#                 st.divider()
#                 st.markdown('<h3 class="section-header">タスク一覧</h3>', unsafe_allow_html=True)
#                 plan_placeholder = st.empty()
#                 plan_placeholder.dataframe(st.session_state.plan)

#                 prepare_plan = st.session_state.plan[st.session_state.plan['category'] == 'prepare'].to_dict(orient='records')
#                 visualize_plan = st.session_state.plan[st.session_state.plan['category'] == 'visualize'].to_dict(orient='records')
#                 report_plan = st.session_state.plan[st.session_state.plan['category'] == 'report'].to_dict(orient='records')

#                 status_text.text("データ準備中...")
#                 progress_bar.progress(20)

#                 st.session_state.work_df_dict = {}

#                 st.markdown('<h3 class="section-header">生成されたデータ</h3>', unsafe_allow_html=True)
#                 tabs = st.tabs([task['output'] for task in prepare_plan])
#                 for idx, task in enumerate(prepare_plan):
#                     st.session_state.plan.loc[st.session_state.plan['task'] == task['task'], 'status'] = '🔄'
#                     plan_placeholder.dataframe(st.session_state.plan)

#                     input_df_names = task['input']
#                     output_df_name = task['output']

#                     # 入力 DataFrame を dict 形式 {name: df} に変換
#                     input_df_dict = {name: st.session_state.work_df_dict[name] for name in input_df_names if name in st.session_state.work_df_dict}
#                     for name in input_df_names:
#                         if name in st.session_state.initial_df_dict:
#                             input_df_dict[name] = st.session_state.initial_df_dict[name]

#                     prepare_agent = create_pandas_dataframe_agent(
#                         llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=st.session_state.api_key),
#                         df=input_df_dict,
#                         agent_type="tool-calling",
#                         verbose=True,
#                         allow_dangerous_code=True,
#                         return_intermediate_steps=True,
#                         handle_parsing_errors=True,
#                         include_df_in_result=True,
#                         df_exec_instruction=True
#                     )
                
#                     prompt_for_data = f"""
#                     あなたは優秀なデータサイエンティストです。           
#                     以下のtaskに従ってdataframeを作成してください。
#                     # task
#                     {task['task']}
#                     # input
#                     {input_df_names}
#                     # output
#                     {output_df_name}
#                     # output_columns
#                     {input_df_names}のカラム(可能な限り) + {task['output_columns']}
                    
#                     # 注意点
#                     - 中間変数、中間データフレームは小まめにhead()を実行して想定通り作成されているか確認してください。
#                     - 日付型のカラムはdatetime型に変換してください。
#                     - outputを生成したら、以下のように{output_df_name}.jsonという名前で保存してください。
#                     {output_df_name}.to_json(
#                         f"tmp/{output_df_name}.json",
#                         orient="records",
#                         date_format="iso",
#                         date_unit="s",
#                         index=False,
#                         force_ascii=False,
#                     )
#                     """
#                     response_data = prepare_agent.invoke(
#                         {"input": prompt_for_data}, {"callbacks": [st_callback]}
#                         )
                    
#                     # 生成物(JSON文字列 または DataFrame)を取得
#                     df_output = pd.read_json(f"tmp/{output_df_name}.json", orient="records")
#                     st.session_state.work_df_dict[output_df_name] = df_output

#                     with tabs[idx]:
#                         st.markdown(f"### {output_df_name}")
#                         try:
#                             st.dataframe(st.session_state.work_df_dict[output_df_name])
#                         except Exception:
#                             st.write(st.session_state.work_df_dict[output_df_name])
                    
#                     st.session_state.plan.loc[st.session_state.plan['output'] == output_df_name, 'status'] = '✅'
#                     plan_placeholder.dataframe(st.session_state.plan)

#                 # ステップ1: コード生成
#                 status_text.text("可視化のためのコードを生成中...")
#                 progress_bar.progress(35)
                
#                 st.markdown('<h3 class="section-header">生成されたビジュアル</h3>', unsafe_allow_html=True)
#                 visualize_tabs = st.tabs([f"visual_{idx + 1}" for idx in range(len(visualize_plan))])
#                 for idx, task in enumerate(visualize_plan):
#                     with visualize_tabs[idx]:
#                         st.session_state.plan.loc[st.session_state.plan['task'] == task['task'], 'status'] = '🔄'
#                         plan_placeholder.dataframe(st.session_state.plan)

#                         input_df_names = task['input']

#                         input_df_list = [{"input_name": input_df, "input_df": st.session_state.work_df_dict[input_df]} for input_df in input_df_names if input_df in st.session_state.work_df_dict]
#                         for name in input_df_names:
#                             if name in st.session_state.initial_df_dict:
#                                 input_df_list.append({"input_name": name, "input_df": st.session_state.initial_df_dict[name]})
#                         df_info = get_dataframe_info(input_df_list)
#                         generated_code = st.session_state.llm_client.generate_code(task, df_info)
                    
#                         # ステップ2: 安全性チェック
#                         status_text.text("安全性をチェック中...")
#                         progress_bar.progress(50)
                        
#                         is_safe, safety_report = st.session_state.safety_checker.is_safe(generated_code)
                        
#                         if not is_safe:
#                             st.error("生成されたコードが安全性チェックに失敗しました")
                            
#                             with st.expander("安全性チェック詳細"):
#                                 st.json(safety_report)
                            
#                             log_execution(user_prompt, generated_code, False, "Safety check failed")

#                             # --------------------
#                             # LLM に修正依頼
#                             # --------------------
#                             st.warning("LLM によるコード修正を試みます…")
#                             try:
#                                 fixed_code = st.session_state.llm_client.fix_code(
#                                     generated_code,
#                                     json.dumps(safety_report, ensure_ascii=False, indent=2),
#                                     task,
#                                     df_info,
#                                 )

#                                 # 再度安全性チェック
#                                 is_safe_fixed, safety_report_fixed = st.session_state.safety_checker.is_safe(fixed_code)

#                                 if not is_safe_fixed:
#                                     st.error("修正後のコードも安全性チェックに失敗しました")
#                                     with st.expander("安全性チェック詳細(修正後)"):
#                                         st.json(safety_report_fixed)
#                                     log_execution(user_prompt, fixed_code, False, "Safety check failed after fix")
#                                 else:
#                                     # --- 修正済みコードを実行フェーズへ ---
#                                     fixed_code_replaced = replace_df_references(fixed_code, input_df_names)
#                                     st.session_state.generated_codes.append(fixed_code_replaced)

#                                     status_text.text("修正後のコードを実行中…")
#                                     progress_bar.progress(75)

#                                     success, stdout_output, error_message = st.session_state.code_executor.execute_code(fixed_code_replaced)

#                                     status_text.text("結果を表示中…")
#                                     progress_bar.progress(100)

#                                     if success:
#                                         log_execution(user_prompt, fixed_code_replaced, True)
#                                     else:
#                                         st.error("実行エラーが発生しました (修正後)")
#                                         st.code(error_message, language="text")
#                                         log_execution(user_prompt, fixed_code_replaced, False, error_message)

#                                     # print 出力表示
#                                     if stdout_output:
#                                         with st.expander("補足内容(修正後)"):
#                                             st.text(stdout_output)

#                             except Exception as fix_exc:
#                                 st.error(f"コード修正の試行に失敗しました: {str(fix_exc)}")

#                             # タスク完了ステータスを更新（修正後コードの実行が終わった場合も含む）
#                             st.session_state.plan.loc[st.session_state.plan['task'] == task['task'], 'status'] = '✅'
#                             plan_placeholder.dataframe(st.session_state.plan)

#                         else:
#                             # --------------------------------------------------
#                             # 安全性チェックを通過したコードをそのまま実行
#                             # --------------------------------------------------
#                             generated_code_replaced = replace_df_references(generated_code, input_df_names)

#                             # セッション保存
#                             st.session_state.last_generated_code = generated_code_replaced  # 後方互換
#                             st.session_state.last_prompt = user_prompt
#                             st.session_state.generated_codes.append(generated_code_replaced)

#                             status_text.text("コードを実行中…")
#                             progress_bar.progress(75)

#                             success, stdout_output, error_message = st.session_state.code_executor.execute_code(generated_code_replaced)

#                             status_text.text("結果を表示中…")
#                             progress_bar.progress(100)

#                             if success:
#                                 log_execution(user_prompt, generated_code_replaced, True)
#                             else:
#                                 # 構文エラー時は自動修正を試みる
#                                 if error_message and error_message.startswith("SyntaxError"):
#                                     st.warning("構文エラーの自動修正を試みます…")
#                                     try:
#                                         fixed_code = st.session_state.llm_client.fix_code(
#                                             generated_code_replaced,
#                                             error_message,
#                                             task,
#                                             df_info,
#                                         )

#                                         # 再度安全性チェック
#                                         is_safe_fixed, safety_report_fixed = st.session_state.safety_checker.is_safe(fixed_code)
#                                         if not is_safe_fixed:
#                                             st.error("修正後のコードが安全性チェックで失敗しました")
#                                             with st.expander("安全性チェック詳細(修正後)"):
#                                                 st.json(safety_report_fixed)
#                                             log_execution(user_prompt, fixed_code, False, "Safety check failed after syntax fix")
#                                         else:
#                                             fixed_code_replaced = replace_df_references(fixed_code, input_df_names)
#                                             st.session_state.generated_codes.append(fixed_code_replaced)

#                                             status_text.text("修正後のコードを再実行中…")
#                                             progress_bar.progress(80)

#                                             success2, stdout_output2, error_message2 = st.session_state.code_executor.execute_code(fixed_code_replaced)

#                                             status_text.text("結果を表示中…")
#                                             progress_bar.progress(100)

#                                             if success2:
#                                                 log_execution(user_prompt, fixed_code_replaced, True)
#                                             else:
#                                                 st.error("自動修正後も実行に失敗しました")
#                                                 st.code(error_message2, language="text")
#                                                 log_execution(user_prompt, fixed_code_replaced, False, error_message2)

#                                             if stdout_output2:
#                                                 with st.expander("補足内容(修正後)"):
#                                                     st.text(stdout_output2)

#                                     except Exception as fix_exc:
#                                         st.error(f"構文エラー修正に失敗しました: {str(fix_exc)}")
#                                 else:
#                                     st.error("実行エラーが発生しました")
#                                     st.code(error_message, language="text")
#                                     log_execution(user_prompt, generated_code_replaced, False, error_message)

#                             # 標準出力の表示
#                             if stdout_output:
#                                 with st.expander("補足内容"):
#                                     st.text(stdout_output)

#                             # タスク完了ステータスを更新
#                             st.session_state.plan.loc[st.session_state.plan['task'] == task['task'], 'status'] = '✅'
#                             plan_placeholder.dataframe(st.session_state.plan)

#                 # ステップ3: レポート生成
#                 status_text.text("レポート生成中...")
#                 progress_bar.progress(85)
                
#                 for task in report_plan:
#                     st.session_state.plan.loc[st.session_state.plan['task'] == task['task'], 'status'] = '🔄'
#                     plan_placeholder.dataframe(st.session_state.plan)

#                     input_df_names = task['input']
#                     # 入力 DataFrame を dict 形式 {name: df} に変換
#                     input_df_dict = {name: st.session_state.work_df_dict[name] for name in input_df_names if name in st.session_state.work_df_dict}
#                     for name in input_df_names:
#                         if name in st.session_state.initial_df_dict:
#                             input_df_dict[name] = st.session_state.initial_df_dict[name]

#                     prompt_for_report = f"""
#                     あなたは財務分析の経験豊富なデータアナリストです。
#                     以下のinputのデータをもとに、taskに従って分析レポートを作成してください。
#                     # task
#                     {task['task']}
                    
#                     # taskの背景
#                     {st.session_state.user_prompt}

#                     # input
#                     {input_df_names}

#                     # ここまでのデータ準備の経緯
#                     {prepare_plan}

#                     taskの背景を踏まえた上で包括的な分析レポートを作成してください。
#                     具体的なデータの解釈やデータから示唆されるリスクや課題も含めてレポートを作成してください。
#                     レポートはmarkdown形式で作成してください。

#                     エグゼクティブサマリー、分析方法、分析結果、発見事項、分析で使用したデータを含めてください。

#                     """

#                     report_agent = create_pandas_dataframe_agent(
#                         llm=ChatOpenAI(model="o3-mini", api_key=st.session_state.api_key),
#                         df=input_df_dict,
#                         agent_type="tool-calling",
#                         verbose=True,
#                         allow_dangerous_code=True,
#                         return_intermediate_steps=True,
#                         df_exec_instruction=False
#                     )

#                     response_report = report_agent.invoke(
#                         {"input": prompt_for_report}, {"callbacks": [st_callback]}
#                         )

#                     st.session_state.generated_report = response_report["output"]

#                     st.markdown('<h3 class="section-header">レポート</h3>', unsafe_allow_html=True)
#                     st.code(st.session_state.generated_report, language="markdown")
#                     st.divider()

#                     st.session_state.plan.loc[st.session_state.plan['task'] == task['task'], 'status'] = '✅'
#                     plan_placeholder.dataframe(st.session_state.plan)  

#                 # プログレスバーとステータステキストをクリア
#                 progress_bar.empty()
#                 status_text.empty()
                
#             except Exception as e:
#                 st.error(f"予期しないエラーが発生しました: {str(e)}")
#                 st.code(traceback.format_exc(), language="text")
#                 log_execution(user_prompt, "", False, str(e))
    
#     elif run_button:
#         st.warning("プロンプトを入力してください")
    
#     # ボタンが押されていない場合でも、最後に生成されたコードがあれば再実行して表示
#     if not run_button:

#         if 'generated_codes' in st.session_state and st.session_state.generated_codes:
#             st.divider()
#             st.markdown('<h3 class="section-header">タスク一覧</h3>', unsafe_allow_html=True)
#             st.dataframe(st.session_state.plan)

#             st.divider()
#             st.markdown('<h3 class="section-header">生成されたデータ</h3>', unsafe_allow_html=True)
#             # 生成されたデータの分だけtabを生成
#             tabs = st.tabs(list(st.session_state.work_df_dict.keys()))

#             for idx, (df_name, df) in enumerate(st.session_state.work_df_dict.items()):
#                 with tabs[idx]:
#                     st.markdown(f"#### {df_name}")
#                     st.dataframe(df, use_container_width=True, height=200)

#             st.divider()
#             st.markdown('<h3 class="section-header">生成されたビジュアル</h3>', unsafe_allow_html=True)
#             visualize_tabs = st.tabs([f"visual_{idx + 1}" for idx in range(len(st.session_state.generated_codes))])

#             for idx, gen_code in enumerate(st.session_state.generated_codes):
#                 with visualize_tabs[idx]:
#                     try:
#                         success, stdout_output, error_message = st.session_state.code_executor.execute_code(gen_code)

#                         if success:
#                             pass
#                         else:
#                             st.error(f"実行エラーが発生しました（再描画 {idx + 1}）")
#                             st.code(error_message, language="text")

#                         # print出力の表示
#                         if stdout_output:
#                             with st.expander(f"補足内容"):
#                                 st.text(stdout_output)

#                     except Exception as e:
#                         st.error(f"再描画中にエラーが発生しました ({idx + 1}): {str(e)}")
#                         st.code(traceback.format_exc(), language="text")
            
#             st.divider()

#         if 'generated_report' in st.session_state and st.session_state.generated_report:
#             st.markdown('<h3 class="section-header">レポート</h3>', unsafe_allow_html=True)
#             st.code(st.session_state.generated_report, language="markdown")
#             st.divider()

#         st.divider()
#     # 生成されたコードの表示
#     with st.expander("生成されたコード", expanded=False):
#         for idx, gen_code in enumerate(st.session_state.generated_codes):
#             st.code(gen_code, language="python")
#             st.divider()

#     # 実行履歴の表示
#     if st.session_state.execution_history:
#         with st.expander("実行履歴"):
#             for i, entry in enumerate(reversed(st.session_state.execution_history[-5:])):  # 最新5件
#                 st.write(f"**{entry['timestamp'].strftime('%H:%M:%S')}** - {'成功' if entry['success'] else '失敗'}")
#                 st.write(f"プロンプト: {entry['prompt'][:100]}...")
#                 if entry['error']:
#                     st.write(f"エラー: {entry['error'][:200]}...")
#                 st.divider()

# if __name__ == "__main__":
#     main()

