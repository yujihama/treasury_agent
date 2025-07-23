import os
import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any
import traceback 
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.callbacks.base import BaseCallbackHandler
from time import sleep

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
    need_clarification: bool
    questions: list[str] | None = None
    refined_prompt: str | None = None

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
    if "app_status" not in st.session_state:
        st.session_state.app_status = "initial"
    if "clarification_history" not in st.session_state:
        st.session_state.clarification_history = []


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


# DF名と日本語解説のマッピング
DF_DESCRIPTIONS = {
    'balances': '口座残高データ',
    'transactions': '取引履歴データ',
    'fx_rates': '為替レート',
    'payables': '買掛金データ',
    'receivables': '売掛金データ',
    'loans': '借入金データ',
    'investments': '投資データ',
    'derivatives': 'デリバティブデータ',
}


def get_dataframe_info(df_list: list[dict[str, pd.DataFrame]]) -> str:
    info = ""
    for df_info in df_list:
        df_name = df_info["input_name"]
        description = DF_DESCRIPTIONS.get(df_name, "説明なし")
        info += f"input_name: {df_name} ({description})\n"
        if isinstance(df_info["input_df"], pd.DataFrame):
            info += f"Shape: {df_info['input_df'].shape}\n"
            info += f"Columns: {', '.join(df_info['input_df'].columns)}\n"
            info += "Data types:\n"
            for col, dtype in df_info["input_df"].dtypes.items():
                info += f"  {col}: {dtype}\n"
        else:
            info += f"len(input_df): {len(df_info['input_df'])}\n"
        info += "\n"
    return info


def replace_df_references(code: str, df_names: list[str], df_dict: Dict[str, pd.DataFrame] | None = None) -> str:
    """コード中の DataFrame 変数名を st.session_state.work_df_dict 参照に置き換える。

    Parameters
    ----------
    code: str
        変換対象のコード。
    df_names: list[str]
        置き換え対象となる DataFrame 名リスト。
    df_dict: dict[str, pd.DataFrame] | None
        参照可能な DataFrame 辞書。サブスレッドでは session_state を使わずに
        スナップショットを渡すため、この引数を使用する。
    """
    import io, tokenize
    from tokenize import TokenInfo

    df_set = set(df_names)

    # サブスレッドでは session_state を触らない
    if df_dict is None:
        df_dict = st.session_state.get("work_df_dict", {})

    replace_map = {
        name: f"st.session_state.work_df_dict['{name}']" for name in df_set if name in df_dict
    }

    result_tokens: list[TokenInfo] = []
    sio = io.StringIO(code)

    for tok in tokenize.generate_tokens(sio.readline):
        if tok.type == tokenize.NAME and tok.string in replace_map:
            tok = TokenInfo(tok.type, replace_map[tok.string], tok.start, tok.end, tok.line)
        result_tokens.append(tok)

    return tokenize.untokenize(result_tokens)

# ---------------------------------------------------------------------------
# Callback handler for background LLM thinking log
# ---------------------------------------------------------------------------


class ListCallbackHandler(BaseCallbackHandler):
    """Collects generated tokens into a list for later display."""

    def __init__(self):
        super().__init__()
        self.logs: list[str] = []

    def on_llm_new_token(self, token: str, **kwargs):
        self.logs.append(token)

    # Tool end (captures the output from the tool)
    def on_tool_end(self, output, **kwargs):
        self.logs.append(f"Action Output: {output}\n")

    # Agent observation hook (LangChain 0.1 compatibility)
    def on_agent_finish(self, output, **kwargs):
        # Some agents send their observation here
        if output:
            self.logs.append(f"Agent Observation: {output}\n")

# ---------------------------------------------------------------------------
# Log formatting helper
# ---------------------------------------------------------------------------

import re


def _format_llm_logs(tokens: list[str]) -> str:
    """
    LLMのログトークンリストを、セクションごとにマークダウンで見やすく整形します。
    **Action Input**の次の行から**Action Output**の前の行まではcodeブロックで表示します。
    """
    text = "".join(tokens)

    # セクションごとに分割
    # まず各セクションの見出しを挿入
    text = re.sub(r"Thought:\s*", "\n\n**Thought**\n\n", text)
    text = re.sub(r"Action:\s*", "\n\n**Action**\n\n", text)
    text = re.sub(r"Action Input:\s*", "\n\n**Action Input**\n\n", text)
    text = re.sub(r"Action Output:\s*", "\n\n**Action Output**\n\n", text)
    text = re.sub(r"Agent Observation:\s*", "\n\n**Agent Observation**\n\n", text)

    # Action Input から Action Output まで、Action Output から次のセクションまたは文末までをcodeブロックにする
    # (?s)で改行も含めてマッチ

    def code_block_replacer_input(match):
        code_content = match.group(1).strip()
        return f"\n\n**Action Input**\n\n```python\n{code_content}\n```\n\n"

    def code_block_replacer_output(match):
        code_content = match.group(1).strip()
        return f"\n\n**Action Output**\n\n```python\n{code_content}\n```\n\n"

    # **Action Input** から **Action Output** までを抽出し、codeブロックで囲む
    text = re.sub(
        r"\*\*Action Input\*\*\n(.*?)(?=\n\*\*Action Output\*\*)",
        code_block_replacer_input,
        text,
        flags=re.DOTALL,
    )

    # **Action Output** から次のセクションまたは文末までを抽出し、codeブロックで囲む
    text = re.sub(
        r"\*\*Action Output\*\*\n(.*?)(?=\n\*\*|\Z)",
        code_block_replacer_output,
        text,
        flags=re.DOTALL,
    )

    return text.strip()

# ---------------------------------------------------------------------------
# Prepare Task Runner (並列実行用ヘルパー)
# ---------------------------------------------------------------------------

def _run_prepare_task(
    task: dict[str, Any],
    base_df_dict: Dict[str, pd.DataFrame],
    api_key: str,
) -> tuple[str, pd.DataFrame | None, bool, str, list[str]]:
    """
    1 つの prepare タスクを実行して結果を返す。

    Returns
    -------
    (output_df_name, df_output_or_None, success, error_message)
    Streamlit API は使用しないため並列スレッドから安全に呼び出せる。
    """
    import os

    output_df_name: str = task.get("output")
    handler = ListCallbackHandler()
    try:
        # --- 入力 DataFrame 準備 ---
        input_df_dict: Dict[str, pd.DataFrame] = {
            name: base_df_dict[name].copy() for name in task.get("input", []) if name in base_df_dict
        }

        if not input_df_dict:
            return output_df_name, None, False, "入力 DataFrame が見つかりません"

        # --- LLM Agent 実行 ---
        prepare_agent = create_pandas_dataframe_agent(
            llm=ChatOpenAI(model="gpt-4.1", temperature=0, api_key=api_key),
            df=input_df_dict,
            agent_type="zero-shot-react-description",
            verbose=True,
            allow_dangerous_code=True,
            return_intermediate_steps=True,
            include_df_in_result=True,
            df_exec_instruction=True,
            agent_executor_kwargs={"handle_parsing_errors": True},
        )

        prompt_for_data = f"""\
あなたは優秀なデータサイエンティストです。
# task
{task['task']}
# input
{task['input']}
# output
{output_df_name}
# output_columns
{task['output_columns']}

# 注意点
- まず初めにinputの各データにアクセス可能かhead()で確認してください。アクセス不可の場合のその旨回答して処理を終了してください。
- 変数、中間データフレームは小まめにhead()を実行して想定通り作成されているか確認してください。
- outputを生成したら、PythonAstREPLToolで以下のコードを実行して{output_df_name}.jsonを生成してください。
{output_df_name}.to_json(
    f"tmp/{output_df_name}.json",
    orient="records",
    date_format="iso",
    date_unit="s",
    index=False,
    force_ascii=False,
)
- 最終ステップにて、{output_df_name}.jsonの実行ステップが正常に終了したことを確認して回答してください。
"""

        # Streamlit Callback は使用しない
        prepare_agent.invoke({"input": prompt_for_data}, {"callbacks": [handler]})

        # --- 生成 json 読み込み ---
        json_path = f"tmp/{output_df_name}.json"
        if os.path.exists(json_path):
            df_output = pd.read_json(json_path, orient="records")
            return output_df_name, df_output, True, "", handler.logs

        return output_df_name, None, False, "JSON が生成されませんでした", handler.logs

    except Exception as exc:
        return output_df_name, None, False, str(exc), []

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
    ユーザーの入力(依頼内容)と対話履歴を元にLLMで詳細化し、
    必要に応じて追加の質問を生成するか、分析方針を確定する。
    """
    user_prompt: str = state["user_prompt"]
    clarification_history: list[Dict[str, str]] = state.get("clarification_history", [])

    llm_client = get_llm_client()
    if llm_client is None:
        state["refined_prompt"] = user_prompt
        state["need_clarification"] = False
        state["questions"] = None
        return state

    history_str = "\n".join(
        [f"Assistant: {msg['content']}" if msg['role'] == 'assistant' 
         else f"User: {msg['content']}" for msg in clarification_history]
    )

    input_df_list = [{"input_name": n, "input_df": df} for n, df in st.session_state.initial_df_dict.items()]
    df_info = get_dataframe_info(input_df_list)

    refine_prompt_text = f"""
あなたは優秀なトレジャリーマネジメントの専門家です。
以下の依頼内容とこれまでの質疑応答を踏まえ、分析方針を立ててください。

# 依頼内容
{user_prompt}

# インプットデータの概要
{df_info}

# これまでの質疑応答
{history_str if history_str else "なし"}

# あなたのタスク
1. 依頼内容と質疑応答を理解し、インプットデータを使った分析方針を明確にします。
2. 不明な点があれば具体的な質問を生成してください。ある程度推測できる内容は質問しないか、クローズドクエスチョンにしてください。
3. 質問は丁寧でプロフェッショナルな口調で行ってください。
4. すべての情報が揃っている場合は、背景や意図を踏まえた詳細な分析方針をマークダウン形式で作成してください。

# 重要ルール
- **`# これまでの質疑応答` で解決済みの内容は、再度同じ内容の質問をしないでください。**
- あなたの最終目的は、ユーザーとの対話を通じてあいまいさをなくし、インプットデータを使った実行可能な分析方針（`refined_prompt`）を完成させることです。
- 一度に全て質問はせず、1,2個程度ずつ質問してください。

# 考慮事項
- あなたは口座残高、取引履歴、為替レート、買掛金、売掛金、借入金、投資、デリバティブのデータを保持しています。
- これらのデータを加工してデータの可視化や分析レポートの作成ができます。
- 依頼内容をただ対応するだけでなく、背景のニーズに答えられるように回答方針を立ててください。

# 出力フォーマット
必ず以下のJSON形式で回答してください。説明は不要です。
- 依頼内容を分析するのに情報が不足している場合:
  `{{"need_clarification": true, "questions": ["具体的な質問1", "具体的な質問2", ...]}}`
- 分析方針を立てるのに十分な情報がある場合:
  `{{"need_clarification": false, "refined_prompt": "## 分析方針\\n..."}}`
"""
    refine_llm = ChatOpenAI(
        model="gpt-4.1",
        api_key=llm_client.api_key,
        verbose=True,
    ).with_structured_output(RefinePromptFormat)
    
    response = refine_llm.invoke(refine_prompt_text)
    
    state["need_clarification"] = response.need_clarification
    state["questions"] = response.questions
    state["refined_prompt"] = response.refined_prompt
    
    # このノードはStreamlitのUI要素を直接操作しない
    return state


###############################################################################
# Plan 生成ノード                                                              
###############################################################################

def generate_plan_node(state: Dict[str, Any]):
    # st.markdown('<h3 class="section-header">タスク一覧</h3>', unsafe_allow_html=True)
    # refine ノードで詳細化されたプロンプトがあればそちらを優先
    if "refined_prompt" in st.session_state:
        user_prompt: str = st.session_state.refined_prompt
    else:
        st.warning("詳細化された依頼内容がありません。")
        user_prompt: str = st.session_state.user_prompt
    df_overview: str = state["df_overview"]

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
    # state 更新
    state["plan_df"] = plan_df
    return state

###############################################################################
# Prepare ノード                                                               
###############################################################################

def prepare_node(state: Dict[str, Any]):
    import os

    plan_df: pd.DataFrame = state["plan_df"]
    prepare_tasks = plan_df[plan_df["category"] == "prepare"].to_dict(orient="records")

    st.markdown('<h3 class="section-header">データ</h3>', unsafe_allow_html=True)
    with st.spinner("データを生成中"):
        with st.expander("生成されたデータ", expanded=True):
            # --- タブ & プレースホルダー生成 ---
            output_names = [task["output"] for task in prepare_tasks]
            tabs = st.tabs(output_names)
            tab_placeholders: Dict[str, Any] = {}
            for idx, name in enumerate(output_names):
                with tabs[idx]:
                    tab_placeholders[name] = st.empty()

            work_df_dict: Dict[str, pd.DataFrame] = {}
            pending_tasks = prepare_tasks.copy()
            api_key = get_llm_client().api_key

            while pending_tasks:
                # 依存が解決済みのタスクを抽出
                ready_tasks = [t for t in pending_tasks if all(
                    n in work_df_dict or n in st.session_state.initial_df_dict for n in t["input"]
                )]

                if not ready_tasks:
                    st.error("依存関係を解決できない prepare タスクがあります。循環参照の可能性があります。")
                    break

                # "🔄" にステータス更新
                for t in ready_tasks:
                    st.session_state.plan.loc[
                        st.session_state.plan["task"] == t["task"], "status"
                    ] = "🔄"
                st.session_state.plan_placeholder.dataframe(st.session_state.plan, use_container_width=True)

                # 並列実行用にスナップショットを渡す
                base_df_dict_snapshot = {**st.session_state.initial_df_dict, **work_df_dict}

                with ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_task = {
                        executor.submit(_run_prepare_task, t, base_df_dict_snapshot, api_key): t for t in ready_tasks
                    }

                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        output_name, df_output, success, err_msg, logs = future.result()

                        # ステータス更新
                        st.session_state.plan.loc[
                            st.session_state.plan["task"] == task["task"], "status"
                        ] = "✅" if success else "❌"

                        placeholder = tab_placeholders.get(output_name)

                        if success and df_output is not None:
                            work_df_dict[output_name] = df_output
                            if placeholder is not None:
                                with placeholder.container():
                                    st.info(f"タスク: {task['task']}")
                                    st.dataframe(df_output[:100], use_container_width=True)
                                    if logs:
                                        with st.expander("LLM Log"):
                                            st.markdown(_format_llm_logs(logs), unsafe_allow_html=True)
                        else: # if success false show error and logs
                            # 失敗した場合でもタブを残すため、ダミーの空 DataFrame を登録
                            work_df_dict[output_name] = pd.DataFrame()
                            if placeholder is not None:
                                with placeholder.container():
                                    st.error(f"{task['task']} 失敗: {err_msg}")
                                    if logs:
                                        with st.expander("LLM Log"):
                                            st.markdown(_format_llm_logs(logs), unsafe_allow_html=True)

                        # プラン表を再描画
                        st.session_state.plan_placeholder.dataframe(st.session_state.plan, use_container_width=True)

                # 完了したタスクを pending から除去
                pending_tasks = [t for t in pending_tasks if t not in ready_tasks]

        st.session_state.plan_placeholder.dataframe(st.session_state.plan, use_container_width=True)

        st.session_state.work_df_dict = work_df_dict
        state["work_df_dict"] = work_df_dict
    return state

###############################################################################
# Visualize Task Runner (並列実行用ヘルパー)
###############################################################################

def _run_visualize_task(
    task: dict[str, Any],
    base_df_dict: Dict[str, pd.DataFrame],
    api_key: str,
    safety_checker: SafetyChecker,
) -> tuple[str, str | None, bool, str]:
    """Heavy part of visualize task (LLM code generation & safety check).

    Returns
    -------
    (task_id, code_to_run_or_None, safe_ok, error_msg)
    """
    try:
        # DataFrame 情報を文字列化
        input_df_names = task["input"]
        input_df_list = [
            {"input_name": n, "input_df": base_df_dict.get(n)} for n in input_df_names if n in base_df_dict
        ]

        df_info = get_dataframe_info(input_df_list)

        from llm_client import LLMClient
        llm_client = LLMClient(api_key)
        generated_code = llm_client.generate_code(task, df_info)

        # DataFrame 参照書き換え
        replaced_code = replace_df_references(generated_code, input_df_names, base_df_dict)

        # 安全性チェック
        is_safe, _ = safety_checker.is_safe(replaced_code)
        code_to_use = replaced_code

        if not is_safe:
            fixed_code = llm_client.fix_code(replaced_code, "安全性チェックに失敗しました", task, df_info)
            fixed_code = replace_df_references(fixed_code, input_df_names, base_df_dict)
            is_safe, _ = safety_checker.is_safe(fixed_code)
            code_to_use = fixed_code if is_safe else None

        if code_to_use is None:
            return task["task"], None, False, "コードの安全性を確保できませんでした"

        return task["task"], code_to_use, True, ""

    except Exception as e:
        return task["task"], None, False, str(e)

###############################################################################
# Visualize ノード                                                             
###############################################################################

def visualize_node(state: Dict[str, Any]):
    import os

    # work_df_dict が存在しないケースに備えて初期化
    if "work_df_dict" not in st.session_state:
        st.session_state.work_df_dict = {}

    plan_df: pd.DataFrame = state["plan_df"]
    visualize_tasks = plan_df[plan_df["category"] == "visualize"].to_dict(orient="records")

    def fix_code(code: str, error_message: str, task: dict[str, Any], df_info: str):
        """メインスレッドで呼び出すコード修正ユーティリティ"""
        input_df_names = task["input"]

        try:
            fixed_code = st.session_state.llm_client.fix_code(code, error_message, task, df_info)
            is_safe_fixed, _ = st.session_state.safety_checker.is_safe(fixed_code)
            if not is_safe_fixed:
                return None
            return replace_df_references(fixed_code, input_df_names, st.session_state.initial_df_dict)
        except Exception as exc:
            logger.exception("fix_code 失敗", exc_info=exc)
            return None

    st.markdown('<h3 class="section-header">ビジュアル</h3>', unsafe_allow_html=True)
    with st.spinner("ビジュアルを生成中"):
        with st.expander("生成されたビジュアル", expanded=True):
            vis_tabs = st.tabs([f"visual_{i+1}" for i in range(len(visualize_tasks))])

            # タブごとにプレースホルダーを用意
            tab_placeholders: Dict[str, Any] = {}
            for idx, task in enumerate(visualize_tasks):
                with vis_tabs[idx]:
                    tab_placeholders[task["task"]] = st.container()

            # 事前にステータスを 🔄 に
            for t in visualize_tasks:
                st.session_state.plan.loc[
                    st.session_state.plan["task"] == t["task"], "status"
                ] = "🔄"
            st.session_state.plan_placeholder.dataframe(st.session_state.plan, use_container_width=True)

            # --- 入力 DataFrame のスナップショット作成 ---
            base_df_dict_snapshot = {**st.session_state.initial_df_dict, **st.session_state.get("work_df_dict", {})}
            api_key = get_llm_client().api_key
            safety_checker = st.session_state.safety_checker

            # --- 依存 DF が存在しないタスクを除外し、ステータスを❌に設定 ---
            runnable_tasks = []
            for t in visualize_tasks:
                missing_inputs = [n for n in t["input"] if n not in base_df_dict_snapshot]
                if missing_inputs:
                    tab_container = tab_placeholders[t["task"]]
                    with tab_container:
                        st.warning(f"入力 DataFrame が不足しているためスキップ: {', '.join(missing_inputs)}")
                    st.session_state.plan.loc[
                        st.session_state.plan["task"] == t["task"], "status"
                    ] = "❌"
                else:
                    runnable_tasks.append(t)

            st.session_state.plan_placeholder.dataframe(st.session_state.plan, use_container_width=True)

            if not runnable_tasks:
                state["generated_codes"] = st.session_state.generated_codes
                return state

            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_task = {
                    executor.submit(_run_visualize_task, t, base_df_dict_snapshot, api_key, safety_checker): t for t in runnable_tasks
                }

                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    tab_container = tab_placeholders[task["task"]]

                    with tab_container:
                        st.info(f"タスク: {task['task']}")
                        with st.spinner(f"生成中"):
                            try:
                                task_id, code_to_run, safe_ok, err_msg = future.result()
                            except Exception as e:
                                code_to_run, safe_ok, err_msg = None, False, str(e)

                            if not safe_ok or code_to_run is None:
                                st.error(f"ビジュアル生成失敗: {err_msg}")
                                st.session_state.plan.loc[
                                    st.session_state.plan["task"] == task["task"], "status"
                                ] = "❌"
                                st.session_state.plan_placeholder.dataframe(st.session_state.plan, use_container_width=True)
                                continue

                        # ---------------- コード実行 ----------------
                        output_placeholder = tab_container.empty()

                        def run_and_render(code: str, suffix: str = "") -> tuple[bool, str | None, str | None]:
                            output_placeholder.empty()
                            with output_placeholder.container():
                                success_inner, stdout_inner, err_inner = (
                                    st.session_state.code_executor.execute_code(code)
                                )

                                if stdout_inner:
                                    with st.expander(f"log{suffix}"):
                                        st.text(stdout_inner)
                                with st.expander(f"code{suffix}"):
                                    st.code(code, language="python")
                            return success_inner, stdout_inner, err_inner

                        # 1 回目
                        success, stdout, err = run_and_render(code_to_run)

                    if not success or err:
                        with tab_container.empty() as tab_container:
                            # 修正試行
                            input_df_list = [
                                {"input_name": n, "input_df": base_df_dict_snapshot.get(n)} for n in task["input"] if n in base_df_dict_snapshot
                            ]
                            df_info = get_dataframe_info(input_df_list)
                            fixed_code = fix_code(code_to_run, err, task, df_info)
                            if fixed_code:
                                st.warning(f"修正後のコードを実行")
                                success, stdout, err = run_and_render(fixed_code, "(修正後)")
                                if success:
                                    code_to_run = fixed_code
                                else:
                                    st.error(err)

                    # 生成コード保存
                    st.session_state.generated_codes.append({"task": task["task"], "code": code_to_run})

                    # ステータス更新
                    st.session_state.plan.loc[
                        st.session_state.plan["task"] == task["task"], "status"
                    ] = "✅" if success else "❌"

                    # プラン表再描画
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
            handler = ListCallbackHandler()
            res = report_agent.invoke({"input": prompt_for_report}, {"callbacks": [handler]})
            st.session_state.generated_report.append(res["output"])
            #st.markdown(res["output"]) # codeブロックではなくMarkdownで表示
            with st.expander("レポート"):
                # res["output"]の中に、</div> <div data-card> #のような並びがあれば改行を入れる
                res["output"] = re.sub(r"<div data-card>[^\n]*", "<div data-card>\n", res["output"])
                st.markdown(res["output"], unsafe_allow_html=True)
                with st.expander("LLM Log"):
                    logs = _format_llm_logs(handler.logs)
                    st.markdown(logs, unsafe_allow_html=True)

        st.session_state.plan.loc[
            st.session_state.plan["task"] == task["task"], "status"
        ] = "✅"
        st.session_state.plan_placeholder.dataframe(st.session_state.plan, use_container_width=True)

    state["generated_report"] = st.session_state.generated_report
    return state

###############################################################################
# Flow Builder                                                                 
###############################################################################

def build_plan_flow():
    g = StateGraph(dict)
    g.add_node("refine", refine_prompt_node)
    g.add_node("plan", generate_plan_node)
    g.add_edge("refine", "plan")
    g.set_entry_point("refine")
    g.set_finish_point("plan")
    return g.compile()

def build_execution_flow():
    g = StateGraph(dict)
    g.add_node("prepare", prepare_node)
    g.add_node("visualize", visualize_node)
    g.add_node("report", report_node)
    g.add_edge("prepare", "visualize")
    g.add_edge("visualize", "report")
    g.set_entry_point("prepare")
    g.set_finish_point("report")
    return g.compile()

###############################################################################
# Streamlit Main                                                               
###############################################################################

def main(initial_df_dict: Dict[str, pd.DataFrame]):
    st.set_page_config(page_title="Treasury Agent", layout="wide")
    _inject_report_css()
    initialize_session_state(initial_df_dict)

    # user_prompt = st.text_area("依頼内容を入力してください", height=100)
    # run_button = st.button("実行", type="primary")

    # ------------------------------------------------------------------
    # 1. 依頼入力フェーズ
    # ------------------------------------------------------------------
    with st.container():
        # 以前のプロンプトがセッションにあれば表示
        prompt_value = st.session_state.get("user_prompt", "")
        user_prompt = st.text_area("依頼内容を入力してください", value=prompt_value, height=100, key="user_prompt_input")
        
        # APIキーが設定されていない場合にメッセージを表示
        if not st.session_state.get("api_key"):
            st.warning("サイドバーからOpenAI API Keyを設定してください。")

        generate_plan_button = st.button(
            "タスク生成", 
            type="primary",
            disabled=not st.session_state.get("api_key") or st.session_state.app_status not in ["initial", "plan_generated", "completed"]
        )

    if generate_plan_button:
        if not user_prompt.strip():
            st.warning("依頼内容を入力してください")
            st.stop()
        
        # 実行のたびに状態をリセット
        st.session_state.messages = []
        st.session_state.plan = pd.DataFrame()
        st.session_state.work_df_dict = {}
        st.session_state.generated_codes = []
        st.session_state.generated_report = ""
        st.session_state.refined_prompt = ""
        st.session_state.clarification_history = []
        st.session_state.conversation_context = ConversationContext()
        st.session_state.user_prompt = user_prompt
        st.session_state.app_status = "planning" # ステータスを計画中に

        st.rerun()

    # ------------------------------------------------------------------
    # 2. 依頼詳細化フェーズ (対話ループ)
    # ------------------------------------------------------------------
    if st.session_state.app_status == "planning":
        st.markdown("---")
        st.markdown("<h4 class='section-header'>依頼内容の確認</h4>", unsafe_allow_html=True)

        if st.session_state.clarification_history:
            with st.expander("対話履歴", expanded=False):
                for message in st.session_state.clarification_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"], unsafe_allow_html=True)

        # 対話履歴の表示
        if len(st.session_state.clarification_history) == 0:
            pass
        elif st.session_state.clarification_history[-1]["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(st.session_state.clarification_history[-1]["content"], unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(st.session_state.clarification_history[-2]["content"], unsafe_allow_html=True)
            with st.chat_message("user"):
                st.markdown(st.session_state.clarification_history[-1]["content"], unsafe_allow_html=True)

        # LLMを呼び出すべきか判断 (初回またはユーザー返信後)
        with st.spinner("確認中..."):
            should_run_refine = not st.session_state.clarification_history or st.session_state.clarification_history[-1]["role"] == "user"

            if should_run_refine:
                # with st.spinner("依頼内容を分析・詳細化しています..."):
                state = {
                    "user_prompt": st.session_state.user_prompt,
                    "clarification_history": st.session_state.clarification_history
                }
                
                try:
                    result_state = refine_prompt_node(state)
                    
                    if result_state.get("need_clarification"):
                        questions = result_state.get("questions", [])
                        if questions:
                            st.session_state.clarification_history.append({"role": "assistant", "content": "\n".join(f"{q}" for q in questions)})
                        st.rerun()
                    else:
                        st.session_state.refined_prompt = result_state.get("refined_prompt")
                        st.session_state.app_status = "plan_generating" # 次のステップへ
                        st.rerun()

                except Exception as e:
                    st.exception(e)
                    logger.exception("refine_prompt_node 失敗")
                    st.session_state.app_status = "initial"
                    # st.rerun()

        # ユーザーからの回答を待つ
        if st.session_state.clarification_history and st.session_state.clarification_history[-1]["role"] == "assistant":
            if user_reply := st.chat_input("回答を入力してください"):
                st.session_state.clarification_history.append({"role": "user", "content": user_reply})
                st.rerun()

    # ------------------------------------------------------------------
    # 2.5 プラン生成フェーズ
    # ------------------------------------------------------------------
    if st.session_state.app_status == "plan_generating":
        with st.spinner("タスクを生成しています..."):
            try:
                sleep(.1) 
                df_overview = get_dataframe_info(
                    [{"input_name": n, "input_df": df} for n, df in st.session_state.initial_df_dict.items()]
                )
                state = {
                    "user_prompt": st.session_state.user_prompt,
                    "refined_prompt": st.session_state.refined_prompt,
                    "df_overview": df_overview
                }
                # generate_plan_nodeはUIを操作するため、メインスレッドで実行
                generate_plan_node(state)
                st.session_state.app_status = "plan_generated"
                st.rerun()
            except Exception as e:
                st.exception(e)
                logger.exception("generate_plan_node 失敗")
                st.session_state.app_status = "initial"
                # st.rerun()

    # ------------------------------------------------------------------
    # 3. プラン確認・編集フェーズ
    # ------------------------------------------------------------------
    if st.session_state.app_status == "plan_generated":
        # st.markdown("---")
        st.markdown("<h3 class='section-header'>タスク一覧</h3>", unsafe_allow_html=True)

        st.info("以下のタスクを確認し、必要に応じてタスクの追加・削除・編集を行ってください。問題がなければ「タスク実行」ボタンを押してください。")
        
        # 対話履歴をコンテキストとして表示
        if st.session_state.get("clarification_history"):
            with st.expander("依頼内容の確認履歴", expanded=False):
                for message in st.session_state.clarification_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"], unsafe_allow_html=True)

        if st.session_state.get("refined_prompt"):
            with st.expander("詳細化された依頼内容", expanded=False):
                st.markdown(st.session_state.refined_prompt, unsafe_allow_html=True)
        
        # data_editorでプランを編集可能にする
        edited_plan_df = st.data_editor(
            st.session_state.plan,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "category": st.column_config.SelectboxColumn(
                    "Category",
                    options=["prepare", "visualize", "report"],
                    required=True,
                ),
                "status": st.column_config.TextColumn("Status", disabled=True),
                "input": st.column_config.ListColumn("Input"),
                "output_columns": st.column_config.ListColumn("Output Columns"),
            },
            key="plan_editor"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("タスク実行", type="primary"):
                st.session_state.plan = edited_plan_df # 実行前に最新の編集内容を保存
                st.session_state.app_status = "executing"
                st.rerun()
        with col2:
            if st.button("最初からやり直す"):
                st.session_state.app_status = "initial"
                st.rerun()

    # ------------------------------------------------------------------
    # 4. 実行フェーズ & 5. 結果表示・チャットフェーズ
    # ------------------------------------------------------------------
    if st.session_state.app_status in ["executing", "completed"]:
        st.markdown("---")
        st.markdown("### タスク実行状況")
        st.session_state.plan_placeholder = st.empty()
        st.session_state.plan_placeholder.dataframe(st.session_state.plan, use_container_width=True)

    if st.session_state.app_status == "executing":
        try:
            # 実行フローを開始
            execution_flow = build_execution_flow()
            state = {
                "plan_df": st.session_state.plan,
                "user_prompt": st.session_state.user_prompt,
                "refined_prompt": st.session_state.get("refined_prompt", ""),
            }
            # 各ノードが session_state.initial_df_dict を直接参照するため、state に含める必要は必ずしもない
            execution_flow.invoke(state)

            # コンテキストを更新
            context = st.session_state.conversation_context
            context.user_prompt = st.session_state.user_prompt
            context.refined_prompt = st.session_state.get("refined_prompt")
            context.plan = st.session_state.get("plan")
            context.prepare_results = st.session_state.get("work_df_dict")
            context.visualize_results = st.session_state.get("generated_codes")
            context.report = "\n".join(st.session_state.get("generated_report", []))
            # 対話履歴もコンテキストに追加
            context.chat_history.extend(st.session_state.clarification_history)

            st.session_state.app_status = "completed"
            st.success("分析が完了しました。")
            st.rerun()
            
        except Exception as e:
            st.exception(e)
            logger.exception("execution_flow.invoke 失敗")
            st.session_state.app_status = "plan_generated" # 失敗したらプラン編集画面に戻す
            # st.rerun()

    if st.session_state.app_status == "completed":
        # 生成データ
        if st.session_state.get("work_df_dict"):
            st.markdown('<h3 class="section-header">データ</h3>', unsafe_allow_html=True)
            with st.expander("生成されたデータ", expanded=False):
                tabs = st.tabs(list(st.session_state.work_df_dict.keys()))
                for idx, (df_name, df_val) in enumerate(st.session_state.work_df_dict.items()):
                    with tabs[idx]:
                        task_info = st.session_state.plan[st.session_state.plan["output"] == df_name]
                        if not task_info.empty:
                            st.info(f"タスク: {task_info.iloc[0]['task']}")
                        st.dataframe(df_val[:100], use_container_width=True)

        # ビジュアル再描画
        if st.session_state.get("generated_codes"):
            st.markdown('<h3 class="section-header">ビジュアル</h3>', unsafe_allow_html=True)
            with st.expander("生成されたビジュアル", expanded=False):
                vis_tabs = st.tabs([f"visual_{i+1}" for i in range(len(st.session_state.generated_codes))])
                for idx, gen_code_info in enumerate(st.session_state.generated_codes):
                    with vis_tabs[idx]:
                        try:
                            task_description = gen_code_info.get("task", "タスクの説明がありません。")
                            gen_code = gen_code_info.get("code", "")
                            st.info(f"タスク: {task_description}")
                            success, stdout, err = st.session_state.code_executor.execute_code(gen_code)
                            if not success and err: st.error(err)
                            if stdout:
                                with st.expander("log"): st.text(stdout)
                            with st.expander("code"): st.code(gen_code, language="python")
                        except Exception as e:
                            st.error(f"再描画失敗: {e}")

        # レポート
        if st.session_state.get("generated_report"):
            st.markdown('<h3 class="section-header">レポート</h3>', unsafe_allow_html=True)
            for report in st.session_state.generated_report:
                with st.expander("レポート", expanded=True):
                    report = re.sub(r"<div data-card>[^\n]*", "<div data-card>\n", report)
                    st.markdown(report, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 5. チャットフェーズ
    # ------------------------------------------------------------------
    is_report_ready = st.session_state.app_status == "completed" and bool(st.session_state.conversation_context.report)

    if is_report_ready:
        st.divider()
        st.markdown('<h3 class="section-header">チャット</h3>', unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

    if is_report_ready and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
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

                st_callback_container = st.container()
                st_callback = StreamlitCallbackHandler(parent_container=st_callback_container, max_thought_containers=2, expand_new_thoughts=False)
                
                try:
                    response_dict = chat_agent.invoke({"input": prompt_for_chat}, {"callbacks": [st_callback]})
                    response_text = response_dict.get("output", "")
                except Exception as e:
                    response_text = f"申し訳ありません、エラーが発生しました: {e}"
                    logger.error(f"Chat agent invocation failed: {e}")
                
                st.markdown(response_text, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

    if is_report_ready:
        if chat_prompt := st.chat_input("レポートや分析について追加で質問してください"):
            st.session_state.messages.append({"role": "user", "content": chat_prompt})
            st.rerun()
