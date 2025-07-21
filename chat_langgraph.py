import os
import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any
import traceback 

# 追加: CSS ファイルを読み込んで注入するユーティリティ関数
CSS_PATH = "style_report.css"

def _inject_report_css():
    """style_report.css が存在すれば読み込み、<style> タグとして挿入する"""
    if os.path.exists(CSS_PATH):
        try:
            with open(CSS_PATH, "r", encoding="utf-8") as f:
                css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        except Exception as css_err:
            logging.warning(f"CSS 読み込みに失敗: {css_err}")
    else:
        logging.info(f"{CSS_PATH} が見つかりませんでした。CSS は読み込まれません。")

from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

# 自作モジュール
import code_executor
from safety_checker import SafetyChecker
from code_executor import CodeExecutor
from llm_client import LLMClient
from pandas_dataframe_agent import create_pandas_dataframe_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
# Context and Utility Dataclasses
###############################################################################

class ConversationContext:
    """
    一連の分析とチャットのコンテキストを管理するクラス。
    """
    def __init__(self):
        self.user_prompt: str | None = None
        self.refined_prompt: str | None = None
        self.plan: pd.DataFrame | None = None
        self.prepare_results: Dict[str, pd.DataFrame] | None = None
        self.visualize_results: list[Dict[str, str]] | None = None
        self.report: str | None = None
        self.chat_history: list[Dict[str, str]] = []

    def get_full_context_as_string(self) -> str:
        """
        保持しているすべてのコンテキストを単一の文字列にフォーマットして返す。
        """
        context_parts = []
        if self.user_prompt:
            context_parts.append(f"## ユーザーからの元の依頼内容\n{self.user_prompt}")
        if self.refined_prompt:
            context_parts.append(f"## 分析方針\n{self.refined_prompt}")
        if self.plan is not None and not self.plan.empty:
            context_parts.append(f"## 実行されたタスクプラン\n{self.plan.to_markdown()}")
        if self.prepare_results:
            prepared_dfs = ", ".join(self.prepare_results.keys())
            context_parts.append(f"## タスク(prepare)で生成されたデータ一覧\n{prepared_dfs}")
        if self.visualize_results:
            viz_tasks = "\n".join([f"- {res.get('task', 'N/A')}" for res in self.visualize_results])
            context_parts.append(f"## タスク(visualize)の概要\n{viz_tasks}")
        if self.report:
            context_parts.append(f"## 生成されたレポート全文\n{self.report}")

        # チャット履歴を時系列で追加 (最新のユーザー質問は含めない)
        history_str = "\n".join([f"### {msg['role']}\n{msg['content']}" for msg in self.chat_history[:-1]])
        if history_str:
            context_parts.append(f"## これまでのチャット履歴\n{history_str}")
            
        return "\n\n---\n\n".join(context_parts)

class Plan(BaseModel):
    category: str
    task: str
    input: list[str]
    output: str | None
    output_columns: list[str] | None

class ResponseFormatter(BaseModel):
    plans: list[Plan]

class RefinePromptFormat(BaseModel):
    refined_prompt: str

###############################################################################
# Session Helpers                                                              
###############################################################################

def initialize_session_state(initial_df_dict: Dict[str, pd.DataFrame]):
    # 一度だけ実行される初期化処理
    if "initialized" not in st.session_state:
        st.session_state.initial_df_dict = initial_df_dict
        st.session_state.safety_checker = SafetyChecker()
        st.session_state.code_executor = CodeExecutor(initial_df_dict)
        st.session_state.execution_history = []
        st.session_state.generated_codes = []
        st.session_state.generated_report = ""
        st.session_state.work_df_dict: Dict[str, pd.DataFrame] = {}
        st.session_state.initialized = True

    # チャット機能追加に伴う state。古いセッションでもエラーにならないように、存在をチェックして初期化する。
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = ConversationContext()
    if "messages" not in st.session_state:
        st.session_state.messages = []


def log_execution(prompt: str, code: str, success: bool, error: str | None = None):
    st.session_state.execution_history.append(
        {
            "timestamp": datetime.now(),
            "prompt": prompt,
            "code": code,
            "success": success,
            "error": error,
        }
    )


def get_dataframe_info(df_list: list[dict[str, pd.DataFrame]]) -> str:
    info = ""
    for df_info in df_list:
        info += f"input_name: {df_info['input_name']}\n"
        if isinstance(df_info["input_df"], pd.DataFrame):
            info += f"Shape: {df_info['input_df'].shape}\n"
            info += f"Columns: {', '.join(df_info['input_df'].columns)}\n"
            info += "Data types:\n"
            for col, dtype in df_info["input_df"].dtypes.items():
                info += f"  {col}: {dtype}\n"
        else:
            info += f"len(input_df): {len(df_info['input_df'])}\n"
    return info


def replace_df_references(code: str, df_names: list[str]) -> str:
    """旧コードから流用。st.session_state.work_df_dict 参照へ書き換える"""
    import io, tokenize
    from tokenize import TokenInfo

    df_set = set(df_names)
    replace_map = {
        name: f"st.session_state.work_df_dict['{name}']"
        for name in df_set
        if name in st.session_state.work_df_dict
    }

    result_tokens: list[TokenInfo] = []
    sio = io.StringIO(code)

    for tok in tokenize.generate_tokens(sio.readline):
        if tok.type == tokenize.NAME and tok.string in replace_map.keys():
            tok = TokenInfo(tok.type, replace_map[tok.string], tok.start, tok.end, tok.line)
        result_tokens.append(tok)

    return tokenize.untokenize(result_tokens)

###############################################################################
# LLM 関連                                                                    
###############################################################################

def get_llm_client() -> LLMClient | None:
    api_key = st.session_state.get("api_key") 

    if not api_key:
        return None

    if (
        "llm_client" not in st.session_state
        or st.session_state.llm_client is None
        or st.session_state.llm_client.api_key != api_key
    ):
        st.session_state.llm_client = LLMClient(api_key)

    return st.session_state.llm_client

###############################################################################
# Prompt Refinement ノード                                                      
###############################################################################

def refine_prompt_node(state: Dict[str, Any]):
    """
    ユーザーの入力(依頼内容)をLLMで詳細化し、後続ノードが利用できるよう
    state["refined_prompt"] として格納する。
    API Key が未設定の場合は詳細化をスキップし、元の入力をそのまま渡す。
    """
    user_prompt: str = state["user_prompt"]

    llm_client = get_llm_client()
    # API Key が無い場合は詳細化せずにスキップ
    if llm_client is None:
        state["refined_prompt"] = user_prompt
        return state

    with st.spinner("依頼内容を詳細化しています"):
        refine_prompt_text = f"""
あなたは優秀なトレジャリーマネジメントの専門家です。
以下の依頼内容をよく理解し、背景や意図を踏まえて回答のための分析方針を詳細化してマークダウン形式で回答してください。
# 依頼内容
{user_prompt}

# 前提
- あなたは口座残高、取引履歴、予算、為替レート、買掛金、売掛金、借入金、投資、デリバティブのデータを保持しています。
- これらのデータを加工してデータの可視化や分析レポートの作成ができます。
- 依頼内容をただ対応するだけでなく、背景のニーズに答えられるように回答方針を立ててください。

# 回答フォーマット(前後の説明などは不要です。以下のフォーマットのみを回答してください。)
- 詳細化した依頼内容
- 分析の具体的手法
- 可視化のアイデア
- 分析結果で記述するポイント
- 分析にあたり考慮すべき注意点
"""
        refine_llm = ChatOpenAI(
            model="gpt-4.1",
            api_key=llm_client.api_key,
            verbose=True,
        ).with_structured_output(RefinePromptFormat)
        response = refine_llm.invoke(refine_prompt_text)
        refined_prompt = response.refined_prompt

    # 詳細化結果を画面に表示
    st.markdown("<h3 class='section-header'>分析方針</h3>", unsafe_allow_html=True)
    state["refined_prompt"] = refined_prompt
    st.session_state.refined_prompt = refined_prompt
    
    with st.expander("分析方針"):
        st.markdown(st.session_state.refined_prompt, unsafe_allow_html=True)

    return state

###############################################################################
# Plan 生成ノード                                                              
###############################################################################

def generate_plan_node(state: Dict[str, Any]):
    st.markdown('<h3 class="section-header">タスク一覧</h3>', unsafe_allow_html=True)
    # refine ノードで詳細化されたプロンプトがあればそちらを優先
    if "refined_prompt" in st.session_state:
        user_prompt: str = st.session_state.refined_prompt
    else:
        st.warning("詳細化された依頼内容がありません。")
        user_prompt: str = st.session_state.user_prompt
    df_overview: str = state["df_overview"]

    with st.spinner("タスクを生成中"):
        plan_prompt_text = f"""
    あなたは経営者の意思決定をサポートする優秀なデータアナリストです。           
    以下の依頼内容の意図をよく考え、依頼内容を実現するための具体的なステップに細分化してください。
    # 依頼内容
    {user_prompt}

    # データの概要
    {df_overview}

    # 手順
    1. 依頼内容を理解する。背景にあるニーズも含めて理解する。
    2. データの概要を確認する
    3. 依頼内容に回答するためにどのような分析やデータ可視化を行うか検討する。データ可視化は必要最小限のダッシュボードが望ましく、フィルタやしきい値設定ができるようにインタラクティブな操作ができるようにする。
    4. 依頼内容を実現するためにprepare(複数可)、visualize(複数可)、report(1つのみ)の3ステップのタスクに細分化する
    5. 細分化された各タスクについて以下の内容を回答する
        - category(prepare/visualize/report)
        - task(与えられたdfをインプットにどのような加工または分析をするか具体的かつ詳細に明記)
        - input(複数ある場合はリスト)
        - output(df_から始まる単一のdataframe名※visualizeとreportの場合はスペース)
        - output_columns(出力するカラムのリスト※可能な限りinputのカラムを残す※出力データフレームがある場合のみ)
    6. prepareのタスクは、visualizeとreportのタスクに必要な各dataframeごとに細分化してください。(1タスク1つのdataframeをoutputとして持つ)
    7. prepareのタスクは、各dataframeをインプットにpythonで実施できる分析内容にしてください。
        """

        plan_prompt = PromptTemplate.from_template("{input}")
        plan_llm = ChatOpenAI(
            model="gpt-4.1",
            temperature=0,
            api_key=get_llm_client().api_key,
            verbose=True,
        ).with_structured_output(ResponseFormatter)

        plan_chain = plan_prompt | plan_llm
        response_plan = plan_chain.invoke({"input": plan_prompt_text})
        if isinstance(response_plan, ResponseFormatter):
            plans = response_plan.plans
        else:
            plans = response_plan

        plan_df = pd.DataFrame([p.model_dump() if isinstance(p, Plan) else p for p in plans])
        plan_df["status"] = "⬜"
        plan_df = plan_df.reindex(
            columns=["category", "status", "task", "input", "output", "output_columns"]
        )

        st.session_state.plan = plan_df

    # ---------------- 表示プレースホルダ ----------------
    st.session_state.plan_placeholder = st.empty()
    st.session_state.plan_placeholder.dataframe(plan_df, use_container_width=True)
    
    # state 更新
    state["plan_df"] = plan_df
    return state

###############################################################################
# Prepare ノード                                                               
###############################################################################

def prepare_node(state: Dict[str, Any]):
    plan_df: pd.DataFrame = state["plan_df"]
    prepare_tasks = plan_df[plan_df["category"] == "prepare"].to_dict(orient="records")

    st.markdown('<h3 class="section-header">データ</h3>', unsafe_allow_html=True)
    with st.expander("生成されたデータ", expanded=True):
        # まずタブをすべて描画し、各タブ内にプレースホルダーを用意しておく
        output_names = [task["output"] for task in prepare_tasks]
        tabs = st.tabs(output_names)

        # タブ名 → プレースホルダー の対応表
        tab_placeholders: Dict[str, Any] = {}
        for idx, name in enumerate(output_names):
            with tabs[idx]:
                # ここでは空のプレースホルダーを作成しておき、後で中身を埋め込む
                tab_placeholders[name] = st.empty()

        work_df_dict: Dict[str, pd.DataFrame] = {}
         
        try:
            for task in prepare_tasks:
                output_df_name = task.get("output")
                # 現在処理中のタブのプレースホルダーを取得
                placeholder = tab_placeholders.get(output_df_name)
                with st.spinner(f"データを生成中[{task['task']}]"):
                    st.session_state.plan.loc[
                        st.session_state.plan["task"] == task["task"], "status"
                    ] = "🔄"
                    st.session_state.plan_placeholder.dataframe(st.session_state.plan, use_container_width=True)

                    input_df_names = task["input"]

                    # 入力 df を dict 化（存在するもののみ）
                    input_df_dict: Dict[str, pd.DataFrame] = {}
                    for name in input_df_names:
                        if name in work_df_dict:
                            input_df_dict[name] = work_df_dict[name].copy()
                        elif name in st.session_state.initial_df_dict:
                            input_df_dict[name] = st.session_state.initial_df_dict[name].copy()
                        else:
                            st.warning(f"DataFrame '{name}' が見つかりません。スキップします。")

                    if not input_df_dict:
                        st.error("有効な入力 DataFrame が無いため prepare タスクをスキップします。")
                        continue

                    st_callback = StreamlitCallbackHandler(parent_container = st.container(), max_thought_containers=2, expand_new_thoughts=False)
                    prepare_agent = create_pandas_dataframe_agent(
                        llm=ChatOpenAI(model="gpt-4.1", temperature=0, api_key=get_llm_client().api_key),
                        df=input_df_dict,
                        agent_type="zero-shot-react-description",
                        verbose=True,
                        allow_dangerous_code=True,
                        return_intermediate_steps=True,
                        include_df_in_result=True,
                        df_exec_instruction=True,
                        agent_executor_kwargs={"handle_parsing_errors": True},
                    )

                    prompt_for_data = f"""
                                    あなたは優秀なデータサイエンティストです。           
                                    以下のtaskに従ってdataframeを作成してください。
                                    # task
                                    {task['task']}
                                    # input
                                    {input_df_names}
                                    # output
                                    {output_df_name}
                                    # output_columns
                                    {input_df_names}のカラム(可能な限り) + {task['output_columns']}
                                    
                                    # 注意点
                                    - まず初めにinputのデータにアクセス可能かhead()で確認してください。アクセス不可の場合のその旨回答して処理を終了してください。
                                    - 変数、中間データフレームは小まめにhead()を実行して想定通り作成されているか確認してください。
                                    - outputを生成したら、以下のように{output_df_name}.jsonという名前で保存してください。
                                    {output_df_name}.to_json(
                                        f"tmp/{output_df_name}.json",
                                        orient="records",
                                        date_format="iso",
                                        date_unit="s",
                                        index=False,
                                        force_ascii=False,
                                    )
                                    """

                    prepare_agent.invoke({"input": prompt_for_data}, {"callbacks": [st_callback]})

                    # 生成された json を読み込み
                    if os.path.exists(f"tmp/{output_df_name}.json"):
                        df_output = pd.read_json(f"tmp/{output_df_name}.json", orient="records")
                        work_df_dict[output_df_name] = df_output
                        st.session_state.plan.loc[
                            st.session_state.plan["output"] == output_df_name, "status"
                        ] = "✅"
                    else:
                        prepare_agent.invoke({"input": prompt_for_data}, {"callbacks": [st_callback]})

                        if os.path.exists(f"tmp/{output_df_name}.json"):
                            df_output = pd.read_json(f"tmp/{output_df_name}.json", orient="records")
                            work_df_dict[output_df_name] = df_output
                            st.session_state.plan.loc[
                                st.session_state.plan["output"] == output_df_name, "status"
                            ] = "✅"
                        else:
                            st.error(f"DataFrame '{output_df_name}' の生成に失敗しました。")
                            st.session_state.plan.loc[
                                st.session_state.plan["output"] == output_df_name, "status"
                            ] = "❌"
                            continue

                    # 生成結果を該当タブに描画
                    if placeholder is not None:
                        with placeholder.container():
                            st.info(f"タスク: {task['task']}")
                            st.dataframe(df_output[:100], use_container_width=True)

        except Exception as e:
            # 例外発生時にトレースバックを取得して画面とログに出力
            st.exception(e)
            logger.exception("prepare タスク失敗")
            # task または output 名で安全にステータス更新 (例外時のみ)
            if output_df_name:
                st.session_state.plan.loc[
                    st.session_state.plan["output"] == output_df_name, "status"
                ] = "❌"
            else:
                st.session_state.plan.loc[
                    st.session_state.plan["task"] == task["task"], "status"
                ] = "❌"

    st.session_state.plan_placeholder.dataframe(st.session_state.plan, use_container_width=True)

    st.session_state.work_df_dict = work_df_dict
    state["work_df_dict"] = work_df_dict
    return state

###############################################################################
# Visualize ノード                                                             
###############################################################################

def visualize_node(state: Dict[str, Any]):
    plan_df: pd.DataFrame = state["plan_df"]
    visualize_tasks = plan_df[plan_df["category"] == "visualize"].to_dict(orient="records")

    def fix_code(code: str, error_message: str, task: dict[str, Any], df_info: str):
        """
        LLM にコード修正を依頼し、安全性チェックを通過した修正コード
        を返します。安全でない、もしくは修正に失敗した場合は ``None`` を返却します。
        """
        input_df_names = task["input"]

        try:
            # --- LLM へ修正依頼 ---
            fixed_code = st.session_state.llm_client.fix_code(
                code,
                error_message,
                task,
                df_info,
            )

            # --- 安全性チェック ---
            is_safe_fixed, safety_report_fixed = st.session_state.safety_checker.is_safe(fixed_code)
            if not is_safe_fixed:
                st.error("修正後のコードが安全性チェックで失敗しました")
                with st.expander("安全性チェック詳細(修正後)"):
                    st.json(safety_report_fixed)
                return None

            # DataFrame 参照を書き換え
            fixed_code_replaced = replace_df_references(fixed_code, input_df_names)
            return fixed_code_replaced

        except Exception as e:
            st.exception(e)
            logger.exception("fix_code 失敗")
            return None

    st.markdown('<h3 class="section-header">ビジュアル</h3>', unsafe_allow_html=True)
    with st.expander("生成されたビジュアル", expanded=True):
        vis_tabs = st.tabs([f"visual_{i+1}" for i in range(len(visualize_tasks))])

        for idx, task in enumerate(visualize_tasks):
            try:
                with vis_tabs[idx]:
                    with st.spinner(f"ビジュアルを生成中[{task['task']}]"):
                        st.session_state.plan.loc[
                            st.session_state.plan["task"] == task["task"], "status"
                        ] = "🔄"
                        st.session_state.plan_placeholder.dataframe(st.session_state.plan, use_container_width=True)

                        input_df_names = task["input"]
                        input_df_list = []
                        for name in input_df_names:
                            df_val = st.session_state.work_df_dict.get(name, st.session_state.initial_df_dict.get(name))
                            if df_val is None:
                                st.warning(f"DataFrame '{name}' が見つかりません。ビジュアル生成をスキップします。")
                                continue
                            input_df_list.append({"input_name": name, "input_df": df_val})
                        df_info = get_dataframe_info(input_df_list)

                        if not input_df_list:
                            continue

                        generated_code = get_llm_client().generate_code(task, df_info)

                        # ---------------- 安全性チェック ----------------
                        replaced_generated_code = replace_df_references(generated_code, input_df_names)
                        is_safe, _ = st.session_state.safety_checker.is_safe(replaced_generated_code)

                        code_to_run = replaced_generated_code
                        if not is_safe:
                            fixed = fix_code(replaced_generated_code, "安全性チェックに失敗しました", task, df_info)
                            if fixed is None:
                                # 修正出来なかった場合はタスク失敗扱い
                                st.session_state.plan.loc[
                                    st.session_state.plan["task"] == task["task"], "status"
                                ] = "❌"
                                st.session_state.plan_placeholder.dataframe(st.session_state.plan, use_container_width=True)
                                continue
                            code_to_run = fixed

                    # ---------------- コード実行 ----------------
                    with vis_tabs[idx]:
                        st.info(f"タスク: {task['task']}")
                        # 1回目と2回目の描画が重複しないようにプレースホルダーを用意
                        output_placeholder = st.empty()

                        def run_and_render(code: str, suffix: str = "") -> tuple[bool, str | None, str | None]:
                            """コードを実行し、プレースホルダーに描画する共通関数"""
                            # 既存の描画をクリア
                            output_placeholder.empty()
                            with output_placeholder.container():
                                success_inner, stdout_inner, err_inner = (
                                    st.session_state.code_executor.execute_code(code)
                                )

                                # ログ出力
                                if stdout_inner:
                                    with st.expander(f"log{suffix}"):
                                        st.text(stdout_inner)

                                # 実行したコード表示
                                with st.expander(f"code{suffix}"):
                                    st.code(code, language="python")

                            return success_inner, stdout_inner, err_inner

                        # 1 回目の実行
                        success, stdout, err = run_and_render(code_to_run)

                        # 実行失敗時の自動修正
                        if not success and err:
                            fixed = fix_code(code_to_run, err, task, df_info)
                            if fixed:
                                # 2 回目の実行（修正後）
                                success, stdout, err = run_and_render(fixed, "(修正後)")
                                code_to_run = fixed  # 成功すれば code を更新

                        # 2 回目でも失敗した場合はエラーメッセージ表示
                        if not success and err:
                            st.error(err)

                    # 成否に応じて生成コードを保存
                    st.session_state.generated_codes.append(
                        {"task": task["task"], "code": code_to_run}
                    )

                    # ステータス更新
                    st.session_state.plan.loc[
                        st.session_state.plan["task"] == task["task"], "status"
                    ] = "✅" if success else "❌"
            except Exception as e:
                st.error(f"visualize 実行失敗: {e}")
                st.session_state.plan.loc[
                    st.session_state.plan["task"] == task["task"], "status"
                ] = "❌"
    
            st.session_state.plan_placeholder.dataframe(st.session_state.plan, use_container_width=True)

    state["generated_codes"] = st.session_state.generated_codes
    return state

###############################################################################
# Report ノード                                                                
###############################################################################

def report_node(state: Dict[str, Any]):
    st.session_state.generated_report = []
    plan_df: pd.DataFrame = state["plan_df"]
    report_tasks = plan_df[plan_df["category"] == "report"].to_dict(orient="records")

    llm = get_llm_client()
    if not llm:
        st.error("API Key 未設定のためレポート生成をスキップ")
        return state

    for task in report_tasks:
        # prepare タスクの概要を文字列化
        prepare_plan = plan_df[plan_df["category"] == "prepare"][["task", "input", "output"]].to_dict(orient="records")

        input_df_names = task["input"]
        # 入力 DataFrame を dict 形式 {name: df} に変換
        input_df_dict = {name: st.session_state.work_df_dict[name] for name in input_df_names if name in st.session_state.work_df_dict}
        for name in input_df_names:
            if name in st.session_state.initial_df_dict:
                input_df_dict[name] = st.session_state.initial_df_dict[name]

        prompt_for_report = f"""
あなたは財務分析の経験豊富なデータアナリストです。
以下のinputのデータをもとに、taskに従って分析レポートを作成してください。

# ユーザーからの依頼内容
{st.session_state.user_prompt}

# あなたのタスク
{task['task']}

# あなたのタスクの背景
{st.session_state.refined_prompt}

# あなたのタスクのinput
{input_df_names}

# ここまでのデータ準備の経緯
{prepare_plan}

# あなたのタスク
inputのデータをよく参照し、taskの背景を踏まえた上で具体的な分析レポートを作成してください。

# 注意点
- まず初めにinputのデータにアクセス可能かhead()などで確認してください。アクセス不可の場合のその旨回答して処理を終了してください。
- 推測で回答せず、PythonAstREPLToolを使用してinputのデータに対して分析を行い、確認して示唆に富んだ観点を示してください。
- 具体的な数値など定量的なデータも根拠に結論を示してください。
- レポートはmarkdown形式で作成してください。
- セクションは必ず `<div data-card>` と `</div>` で囲み、カード形式で視覚的に区切ってください。
- 強調または警告メッセージには `<p data-alert="ok|warn|info"> ... </p>` を使用してください。
- エグゼクティブサマリー、分析方法、分析結果、発見事項、分析で使用したデータを含めてください。
"""
        st.markdown('<h3 class="section-header">レポート</h3>', unsafe_allow_html=True)
        
        report_agent = create_pandas_dataframe_agent(
            llm=ChatOpenAI(model="gpt-4.1", api_key=llm.api_key),
            df=input_df_dict,
            agent_type="tool-calling",
            verbose=True,
            allow_dangerous_code=True,
            return_intermediate_steps=True,
            df_exec_instruction=False,
            agent_executor_kwargs={"handle_parsing_errors": True},
            )
        with st.spinner(f"レポートを生成中[{task['task']}]"):
            st_callback = StreamlitCallbackHandler(parent_container = st.container(), max_thought_containers=2, expand_new_thoughts=False)
            res = report_agent.invoke({"input": prompt_for_report}, {"callbacks": [st_callback]})
            st.session_state.generated_report.append(res["output"])
            #st.markdown(res["output"]) # codeブロックではなくMarkdownで表示
            with st.expander("レポート"):
                st.markdown(res["output"], unsafe_allow_html=True)

        st.session_state.plan.loc[
            st.session_state.plan["task"] == task["task"], "status"
        ] = "✅"
        st.session_state.plan_placeholder.dataframe(st.session_state.plan, use_container_width=True)

    state["generated_report"] = st.session_state.generated_report
    return state

###############################################################################
# Flow Builder                                                                 
###############################################################################

def build_flow():
    g = StateGraph(dict)

    # 新規ノード
    g.add_node("refine", refine_prompt_node)
    g.add_node("plan", generate_plan_node)
    g.add_node("prepare", prepare_node)
    g.add_node("visualize", visualize_node)
    g.add_node("report", report_node)

    # フローの接続
    g.add_edge("refine", "plan")
    g.add_edge("plan", "prepare")
    g.add_edge("prepare", "visualize")
    g.add_edge("visualize", "report")
    
    # エントリーポイントと終了ポイント
    g.set_entry_point("refine")
    g.set_finish_point("report")
    compiled = g.compile()
    return compiled

###############################################################################
# Streamlit Main                                                               
###############################################################################

def main(initial_df_dict: Dict[str, pd.DataFrame]):
    st.set_page_config(page_title="Streamlit Agent (LangGraph)", layout="wide")
    # --- 追加: CSS を注入 ---
    _inject_report_css()
    initialize_session_state(initial_df_dict)

    user_prompt = st.text_area("依頼内容を入力してください", height=100)
    run_button = st.button("実行", type="primary")

    if run_button:
        if not user_prompt.strip():
            st.warning("プロンプトを入力してください")
            st.stop()

        # 実行のたびに、前の実行結果とチャット履歴をクリアする
        st.session_state.messages = []
        st.session_state.plan = pd.DataFrame()
        st.session_state.work_df_dict = {}
        st.session_state.generated_codes = []
        st.session_state.generated_report = ""
        st.session_state.refined_prompt = ""
        st.session_state.conversation_context = ConversationContext()
        # ユーザープロンプトは上書き
        st.session_state.user_prompt = user_prompt

        llm_client = get_llm_client()
        if llm_client is None:
            st.error("API Key を入力してください")
            st.stop()

        df_overview = get_dataframe_info(
            [
                {"input_name": n, "input_df": df}
                for n, df in st.session_state.initial_df_dict.items()
            ]
        )

        # グラフの実行を安全にラップ
        try:
            flow = build_flow()
            state = {
                "user_prompt": user_prompt,
                "df_overview": df_overview,
            }
            flow.invoke(state)

            # --- 全フロー完了後、コンテキストを更新 ---
            context = st.session_state.conversation_context
            context.user_prompt = st.session_state.user_prompt
            context.refined_prompt = st.session_state.get("refined_prompt")
            context.plan = st.session_state.get("plan")
            context.prepare_results = st.session_state.get("work_df_dict")
            context.visualize_results = st.session_state.get("generated_codes")
            context.report = "\n".join(st.session_state.get("generated_report", []))

        except Exception as e:
            # フロー全体で予期せぬ例外が発生した場合でも画面が白くならないようにする
            st.exception(e)
            logger.exception("flow.invoke 失敗")

        st.success("分析が完了しました")

    # ------------------------------------------------------------------
    # ページ表示/再描画 (セッションステートに基づいてUIを構築)
    # ------------------------------------------------------------------

    # --- 分析結果の表示 ---
    if not run_button and "plan" in st.session_state and isinstance(st.session_state.plan, pd.DataFrame) and not st.session_state.plan.empty:
        if st.session_state.get("refined_prompt"):
            st.markdown("<h3 class='section-header'>分析方針</h3>", unsafe_allow_html=True)
            with st.expander("分析方針", expanded=False):
                st.markdown(st.session_state.refined_prompt, unsafe_allow_html=True)

        st.markdown('<h3 class="section-header">タスク一覧</h3>', unsafe_allow_html=True)
        st.dataframe(st.session_state.plan, use_container_width=True)

        # 生成データ
        if st.session_state.get("work_df_dict"):
            st.markdown('<h3 class="section-header">データ</h3>', unsafe_allow_html=True)
            with st.expander("生成されたデータ"):
                tabs = st.tabs(list(st.session_state.work_df_dict.keys()))
                for idx, (df_name, df_val) in enumerate(st.session_state.work_df_dict.items()):
                    with tabs[idx]:
                        # plan から該当タスクを検索
                        task_info = st.session_state.plan[st.session_state.plan["output"] == df_name]
                        if not task_info.empty:
                            st.info(f"タスク: {task_info.iloc[0]['task']}")
                        st.dataframe(df_val[:100], use_container_width=True, )

        # ビジュアル再描画
        if st.session_state.get("generated_codes"):
            st.markdown('<h3 class="section-header">ビジュアル</h3>', unsafe_allow_html=True)
            with st.expander("生成されたビジュアル"):
                vis_tabs = st.tabs([f"visual_{i+1}" for i in range(len(st.session_state.generated_codes))])
                for idx, gen_code_info in enumerate(st.session_state.generated_codes):
                    with vis_tabs[idx]:
                        try:
                            task_description = gen_code_info.get("task", "タスクの説明がありません。")
                            gen_code = gen_code_info.get("code", "")
                            st.info(f"タスク: {task_description}")

                            success, stdout, err = st.session_state.code_executor.execute_code(gen_code)
                            if not success and err:
                                st.error(err)
                            if stdout:
                                with st.expander("log"):
                                    st.text(stdout)
                            with st.expander("code"):
                                st.code(gen_code, language="python")

                        except Exception as e:
                            st.error(f"再描画失敗: {e}")

        # レポート
        if st.session_state.get("generated_report"):
            st.markdown('<h3 class="section-header">レポート</h3>', unsafe_allow_html=True)
            for report in st.session_state.generated_report:
                #st.markdown(report)
                with st.expander("レポート"):
                    st.markdown(report, unsafe_allow_html=True)

    # --- チャット入力欄の表示と処理 ---
    # レポートが生成された後のみチャット機能を有効化
    is_report_ready = bool(
        st.session_state.get("conversation_context") and st.session_state.conversation_context.report
    )

    # --- チャット履歴の表示 ---
    if is_report_ready:
        st.divider()
        st.markdown('<h3 class="section-header">チャット</h3>', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    # アシスタントの応答を生成・表示
    # 最新のメッセージがユーザーからのものであれば、アシスタントが応答する
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                st.session_state.conversation_context.chat_history = st.session_state.messages
                full_context_str = st.session_state.conversation_context.get_full_context_as_string()
                
                last_user_prompt = st.session_state.messages[-1]["content"]

                prompt_for_chat = f"""
あなたは、すでに行われた一連の分析結果を完全に理解した上で、追加の質問に答えるデータアナリストです。
以下のコンテキスト情報を踏まえて、ユーザーからの最後の質問に、詳細かつ的確に回答してください。
必要であれば、利用可能なDataFrameを分析して回答を生成することもできます。

# コンテキスト
{full_context_str}

# ユーザーからの最後の質問
{last_user_prompt}

**ユーザーの質問の意図を推測して、分析の過程で生成されたDataFrameをしっかりと確認したうえで論理的に回答してください。**

"""
                all_dfs = {**st.session_state.initial_df_dict, **st.session_state.work_df_dict}
                chat_agent = create_pandas_dataframe_agent(
                    llm=ChatOpenAI(model="gpt-4.1", temperature=0, api_key=get_llm_client().api_key),
                    df=all_dfs,
                    agent_type="zero-shot-react-description",
                    verbose=True,
                    allow_dangerous_code=True,
                    return_intermediate_steps=True,
                    agent_executor_kwargs={"handle_parsing_errors": True},
                    df_exec_instruction=True,
                )

                # StreamlitCallbackHandler用のコンテナを用意
                st_callback_container = st.container()
                st_callback = StreamlitCallbackHandler(parent_container = st_callback_container, max_thought_containers=2, expand_new_thoughts=False)
                
                try:
                    # stream() は generator を返すため、そのままでは dict アクセスできず例外となる。
                    # エンドツーエンドで一括応答を得る場合は invoke() を使用する。
                    response_dict = chat_agent.invoke(
                        {"input": prompt_for_chat},
                        {"callbacks": [st_callback]}
                    )
                    response_text = response_dict.get("output", "")
                except Exception as e:
                    response_text = f"申し訳ありません、エラーが発生しました: {e}"
                    logger.error(f"Chat agent invocation failed: {e}")
                
                # 最終的な回答を表示
                st.markdown(response_text, unsafe_allow_html=True)
                # アシスタントの応答を履歴に追加
                st.session_state.messages.append({"role": "assistant", "content": response_text})

    # ユーザーからの入力を受け付ける
    if is_report_ready:
        if chat_prompt := st.chat_input(
            "レポートや分析について追加で質問してください", disabled=not is_report_ready
        ):
            # ユーザーのメッセージを履歴に追加
            st.session_state.messages.append({"role": "user", "content": chat_prompt})
            # 画面を再実行して、ユーザーのメッセージを即時表示
            st.rerun()
