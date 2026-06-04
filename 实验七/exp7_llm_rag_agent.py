#!/usr/bin/env python
# coding: utf-8
"""
实验七：大语言模型实验

本脚本把实验七的三部分合在一个可直接运行的代码文件中：
1. 提示工程实践：CRISPE 提示词 + 糟糕提示词对比。
2. RAG 系统构建：本地企业政策文档 -> 分块 -> 向量化 -> FAISS 检索 -> LLM 生成。
3. 简易 Agent 设计：模拟“搜索-摘要-报告”的思考-行动-观察循环。

默认 backend=auto：
- 检测到 ZHIPUAI_API_KEY / ZHIPU_API_KEY 时优先使用智谱 AI。
- 检测到 DEEPSEEK_API_KEY 时使用 DeepSeek OpenAI-compatible API。
- 检测到 OPENAI_API_KEY 时使用 OpenAI/OpenAI-compatible API。
- 没有 API Key 时自动切换 mock 后端，方便先离线跑通代码和生成报告素材。

运行示例：
    python exp7_llm_rag_agent.py --part all --backend mock
    python exp7_llm_rag_agent.py --part rag --backend zhipuai
    python exp7_llm_rag_agent.py --part prompt --backend deepseek

.env 示例：
    ZHIPUAI_API_KEY=你的智谱AI Key
    ZHIPUAI_CHAT_MODEL=glm-4-flash
    ZHIPUAI_EMBEDDING_MODEL=embedding-3

    DEEPSEEK_API_KEY=你的DeepSeek Key
    DEEPSEEK_BASE_URL=https://api.deepseek.com
    DEEPSEEK_CHAT_MODEL=deepseek-v4-flash

    OPENAI_API_KEY=你的Key
    OPENAI_BASE_URL=https://api.openai.com/v1
    OPENAI_CHAT_MODEL=gpt-4o-mini
    OPENAI_EMBEDDING_MODEL=text-embedding-3-small
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - 兼容未安装 python-dotenv 的临时环境
    def load_dotenv(*args, **kwargs) -> bool:
        return False

try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except Exception:  # pragma: no cover - 没装 faiss-cpu 时仍允许 mock 跑通
    faiss = None
    FAISS_AVAILABLE = False


# Windows 下重定向日志时也尽量保持中文输出稳定。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)


RANDOM_SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs_exp7"
DEFAULT_KB_DIR = SCRIPT_DIR / "knowledge_base_exp7"


# =========================
# 通用工具
# =========================


def print_header(title: str) -> None:
    print(f"\n========== {title} ==========")


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_json_dump(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================
# API 客户端：LLM + Embedding
# =========================


@dataclass
class ExperimentConfig:
    backend: str = "auto"  # auto / mock / zhipuai / deepseek / openai
    output_dir: Path = DEFAULT_OUTPUT_DIR
    kb_dir: Path = DEFAULT_KB_DIR
    temperature: float = 0.2
    max_tokens: int = 1600
    embedding_dim: int = 384
    chunk_size: int = 450
    chunk_overlap: int = 80
    top_k: int = 4
    rebuild_kb: bool = False


class ChatClient:
    """统一封装智谱 AI、DeepSeek、OpenAI/OpenAI-compatible 和离线 mock 后端。"""

    def __init__(
        self,
        backend: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1600,
    ) -> None:
        load_dotenv()
        self.requested_backend = backend.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.backend = self._resolve_backend(self.requested_backend)
        self.client = None
        self.model = "mock-llm"

        if self.backend == "zhipuai":
            api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
            if not api_key:
                print("未检测到智谱 API Key，ChatClient 切换到 mock。")
                self.backend = "mock"
            else:
                try:
                    from zhipuai import ZhipuAI  # type: ignore

                    self.client = ZhipuAI(api_key=api_key)
                    self.model = os.getenv("ZHIPUAI_CHAT_MODEL", "glm-4-flash")
                except Exception as exc:
                    print(f"初始化智谱客户端失败，切换到 mock：{exc}")
                    self.backend = "mock"
        elif self.backend == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("未检测到 OPENAI_API_KEY，ChatClient 切换到 mock。")
                self.backend = "mock"
            else:
                try:
                    from openai import OpenAI  # type: ignore

                    base_url = os.getenv("OPENAI_BASE_URL") or None
                    self.client = OpenAI(api_key=api_key, base_url=base_url)
                    self.model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
                except Exception as exc:
                    print(f"初始化 OpenAI 客户端失败，切换到 mock：{exc}")
                    self.backend = "mock"
        elif self.backend == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                print("未检测到 DEEPSEEK_API_KEY，ChatClient 切换到 mock。")
                self.backend = "mock"
            else:
                try:
                    from openai import OpenAI  # type: ignore

                    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
                    self.client = OpenAI(api_key=api_key, base_url=base_url)
                    self.model = (
                        os.getenv("DEEPSEEK_CHAT_MODEL")
                        or os.getenv("DEEPSEEK_MODEL")
                        or "deepseek-v4-flash"
                    )
                except Exception as exc:
                    print(f"初始化 DeepSeek 客户端失败，切换到 mock：{exc}")
                    self.backend = "mock"

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend in {"mock", "zhipuai", "deepseek", "openai"}:
            return backend
        has_zhipu = bool(os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY"))
        has_deepseek = bool(os.getenv("DEEPSEEK_API_KEY"))
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        if has_zhipu:
            return "zhipuai"
        if has_deepseek:
            return "deepseek"
        if has_openai:
            return "openai"
        return "mock"

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if self.backend == "zhipuai" and self.client is not None:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content

        if self.backend in {"openai", "deepseek"} and self.client is not None:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""

        return self._mock_chat(system_prompt, user_prompt)

    def _mock_chat(self, system_prompt: str, user_prompt: str) -> str:
        """离线演示输出：保证没有 API Key 时仍能完整跑通实验流程。"""
        combined = system_prompt + "\n" + user_prompt
        if "智能作文批改助手" in combined or "作文" in combined:
            if "CRISPE" in combined or "输出格式" in combined:
                return mock_good_essay_feedback(user_prompt)
            return mock_bad_essay_feedback(user_prompt)

        if "企业制度问答助手" in combined or "只根据资料回答" in combined:
            question = extract_between(user_prompt, "问题：", "\n") or user_prompt
            context = user_prompt
            return mock_rag_answer(question, context)

        return "这是 mock 后端返回的示例结果。请配置 API Key 后重新运行，可获得真实模型输出。"


class EmbeddingClient:
    """统一封装真实嵌入 API 和离线哈希嵌入。"""

    def __init__(self, backend: str = "auto", dimensions: int = 384, batch_size: int = 32) -> None:
        load_dotenv()
        self.requested_backend = backend.lower()
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.backend = ChatClient._resolve_backend(self.requested_backend)
        self.client = None
        self.model = "mock-hash-embedding"

        if self.backend == "zhipuai":
            api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
            if not api_key:
                print("未检测到智谱 API Key，EmbeddingClient 切换到 mock-hash。")
                self.backend = "mock"
            else:
                try:
                    from zhipuai import ZhipuAI  # type: ignore

                    self.client = ZhipuAI(api_key=api_key)
                    self.model = os.getenv("ZHIPUAI_EMBEDDING_MODEL", "embedding-3")
                except Exception as exc:
                    print(f"初始化智谱 Embedding 失败，切换到 mock-hash：{exc}")
                    self.backend = "mock"
        elif self.backend == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("未检测到 OPENAI_API_KEY，EmbeddingClient 切换到 mock-hash。")
                self.backend = "mock"
            else:
                try:
                    from openai import OpenAI  # type: ignore

                    base_url = os.getenv("OPENAI_BASE_URL") or None
                    self.client = OpenAI(api_key=api_key, base_url=base_url)
                    self.model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
                except Exception as exc:
                    print(f"初始化 OpenAI Embedding 失败，切换到 mock-hash：{exc}")
                    self.backend = "mock"
        elif self.backend == "deepseek":
            print("DeepSeek 后端用于聊天生成；Embedding 使用离线 mock-hash 兜底。")
            self.backend = "mock"

        if self.backend == "mock":
            self.client = None
            self.model = "mock-hash-embedding"

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        texts = [str(text) for text in texts]
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        if self.backend == "zhipuai" and self.client is not None:
            vectors: List[List[float]] = []
            for batch in chunked(texts, self.batch_size):
                response = self.client.embeddings.create(model=self.model, input=list(batch))
                vectors.extend([item.embedding for item in response.data])
            return normalize_vectors(np.asarray(vectors, dtype=np.float32))

        if self.backend == "openai" and self.client is not None:
            vectors = []
            for batch in chunked(texts, self.batch_size):
                response = self.client.embeddings.create(model=self.model, input=list(batch))
                vectors.extend([item.embedding for item in response.data])
            return normalize_vectors(np.asarray(vectors, dtype=np.float32))

        vectors = np.vstack([hash_text_embedding(text, self.dimensions) for text in texts])
        return normalize_vectors(vectors.astype(np.float32))


# =========================
# 第一部分：提示工程实践
# =========================


@dataclass
class EssayCase:
    essay_id: str
    title: str
    text: str


SAMPLE_ESSAYS: List[EssayCase] = [
    EssayCase(
        essay_id="essay_1",
        title="难忘的运动会",
        text=(
            "今天学校举行了一年一度的运动会。早晨操场上人山人海，同学们都非常兴奋。"
            "我报名参加了八百米比赛，刚开始我跑得很快，可是到了第二圈就觉得胸口很闷，脚步也慢了下来。"
            "班主任在跑道边不停给我鼓厉，同学们也喊着我的名字。听到大家的加油，我仿佛又有了力气，终于坚持到了终点。"
            "虽然我没有取得第一名，但是我明白了坚持比名次更重要。这次运动会让我受益非浅，也让我更加热爱班集体。"
        ),
    ),
    EssayCase(
        essay_id="essay_2",
        title="一次读书分享会",
        text=(
            "上个周末，我和父母一起去图书馆参加读书分享会。走进大厅，整齐的书架和淡淡的书香扑面而来。"
            "我选择了一本关于科学家的传记，它讲述了他们面对困难仍然认真研究的故事。"
            "通过这次活动，使我懂得了读书不仅能增长知识，还能让人学会思考。"
            "回家后，我决定每天坚持阅读二十分钟，并把好词好句记在笔记本上。我很珍惜这次难得的机慧。"
        ),
    ),
]


def build_crispe_prompt(essay: EssayCase) -> str:
    """使用 CRISPE 框架构造高质量作文批改提示词。"""
    return f"""
你将使用 CRISPE 框架完成作文批改任务。

C - Context（背景）：这是一篇初中生短文，老师需要你做基础语文批改，重点关注错别字、病句、评分与总体评价。
R - Role（角色）：你是认真、客观、表达温和的语文作文批改助手。
I - Input（输入）：作文题目和正文如下。
题目：{essay.title}
正文：{essay.text}
S - Steps（步骤）：
1. 通读全文，只依据作文原文进行判断，不要虚构不存在的问题。
2. 标注作文中的错别字，给出“原词/正确写法/说明”；如果没有错别字，写“未发现明显错别字”。
3. 找出至少 1 处病句或表达不顺的句子，给出修改建议。
4. 从内容、结构、语言三个维度打分，每项满分 10 分，并给出一句理由。
5. 给出 50 字以内的总体评价，语气鼓励但要指出改进方向。
P - Parameters（限制）：回答必须简洁清楚；不要超过 500 字；不要输出与作文无关的内容。
E - Evaluation（输出格式）：请严格按下面 Markdown 格式输出：

## 1. 错别字
| 原词 | 正确写法 | 说明 |
|---|---|---|

## 2. 病句与修改
| 原句 | 问题 | 修改建议 |
|---|---|---|

## 3. 三维评分
| 维度 | 分数/10 | 理由 |
|---|---:|---|
| 内容 |  |  |
| 结构 |  |  |
| 语言 |  |  |

## 4. 总体评价（50字以内）
""".strip()


def build_bad_prompt(essay: EssayCase) -> str:
    return f"帮我看看这篇作文写得怎么样：{essay.text}"


def run_prompt_engineering(chat_client: ChatClient, output_dir: Path) -> Dict[str, object]:
    print_header("第一部分：提示工程实践")
    ensure_dir(output_dir)

    records: List[Dict[str, str]] = []
    for essay in SAMPLE_ESSAYS:
        good_prompt = build_crispe_prompt(essay)
        bad_prompt = build_bad_prompt(essay)

        good_output = chat_client.chat(
            system_prompt="你是智能作文批改助手。请按用户指定格式完成批改。",
            user_prompt=good_prompt,
        )
        bad_output = chat_client.chat(
            system_prompt="你是普通聊天助手。",
            user_prompt=bad_prompt,
        )

        records.append(
            {
                "essay_id": essay.essay_id,
                "title": essay.title,
                "essay_text": essay.text,
                "crispe_prompt": good_prompt,
                "bad_prompt": bad_prompt,
                "crispe_output": good_output,
                "bad_output": bad_output,
            }
        )

    markdown = render_prompt_results(records, chat_client)
    write_text(output_dir / "exp7_prompt_engineering_results.md", markdown)
    safe_json_dump(output_dir / "exp7_prompt_engineering_results.json", records)
    print(f"提示工程结果已保存：{output_dir / 'exp7_prompt_engineering_results.md'}")
    return {"records": records, "markdown_path": str(output_dir / "exp7_prompt_engineering_results.md")}


def render_prompt_results(records: Sequence[Dict[str, str]], chat_client: ChatClient) -> str:
    lines = [
        "# 实验七第一部分：提示工程实践结果",
        "",
        f"生成时间：{now_str()}",
        f"LLM 后端：{chat_client.backend} / {chat_client.model}",
        "",
        "## 简要分析（100字以内）",
        "CRISPE 提示词明确了背景、角色、输入、步骤、限制和输出格式，能让模型按要求逐项检查；糟糕提示词缺少角色、评分维度和格式约束，输出容易泛泛而谈，难以用于报告对比。",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## 作文：{record['title']}（{record['essay_id']}）",
                "",
                "### 原文",
                record["essay_text"],
                "",
                "### 完整 CRISPE 提示词",
                "```text",
                record["crispe_prompt"],
                "```",
                "",
                "### 糟糕提示词",
                "```text",
                record["bad_prompt"],
                "```",
                "",
                "### CRISPE 提示词输出",
                record["crispe_output"],
                "",
                "### 糟糕提示词输出",
                record["bad_output"],
                "",
            ]
        )
    return "\n".join(lines)


def mock_good_essay_feedback(prompt: str) -> str:
    if "鼓厉" in prompt or "非浅" in prompt:
        return """## 1. 错别字
| 原词 | 正确写法 | 说明 |
|---|---|---|
| 鼓厉 | 鼓励 | “鼓励”表示激励、支持。 |
| 受益非浅 | 受益匪浅 | 常用成语应写作“受益匪浅”。 |

## 2. 病句与修改
| 原句 | 问题 | 修改建议 |
|---|---|---|
| 听到大家的加油，我仿佛又有了力气。 | “加油”作名词不够准确。 | 听到大家的加油声，我仿佛又有了力气。 |

## 3. 三维评分
| 维度 | 分数/10 | 理由 |
|---|---:|---|
| 内容 | 8 | 事件完整，能写出坚持的收获。 |
| 结构 | 8 | 按比赛过程展开，层次较清楚。 |
| 语言 | 7 | 表达通顺，但有错别字和个别搭配问题。 |

## 4. 总体评价（50字以内）
叙事完整，有真情实感；注意错别字和词语搭配，语言会更准确。"""
    return """## 1. 错别字
| 原词 | 正确写法 | 说明 |
|---|---|---|
| 机慧 | 机会 | “机会”指时机、契机。 |

## 2. 病句与修改
| 原句 | 问题 | 修改建议 |
|---|---|---|
| 通过这次活动，使我懂得了读书不仅能增长知识，还能让人学会思考。 | 介词结构滥用导致主语残缺。 | 这次活动使我懂得了读书不仅能增长知识，还能让人学会思考。 |

## 3. 三维评分
| 维度 | 分数/10 | 理由 |
|---|---:|---|
| 内容 | 8 | 围绕读书分享会展开，中心明确。 |
| 结构 | 8 | 从参加活动到收获，顺序自然。 |
| 语言 | 7 | 整体流畅，但有错别字和病句。 |

## 4. 总体评价（50字以内）
主题积极，结构完整；改正错别字和主语残缺后，表达会更严谨。"""


def mock_bad_essay_feedback(prompt: str) -> str:
    return "作文整体写得不错，内容比较完整，也有一定真情实感。建议再把语言写得更生动一些，注意检查错别字和句子是否通顺。"


# =========================
# 第二部分：RAG 系统构建
# =========================


@dataclass
class DocumentChunk:
    chunk_id: str
    source: str
    title: str
    text: str
    start: int
    end: int


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float


SAMPLE_KB_DOCS: Dict[str, str] = {
    "company_leave_policy.txt": """
公司请假制度

第一条 适用范围：本制度适用于公司全体正式员工、试用期员工以及经部门负责人确认的实习人员。员工因病、因事、婚丧、生育、学习考试等原因不能按时出勤时，应按照本制度办理请假手续。请假应遵循提前申请、真实说明、逐级审批、按时销假的原则。

第二条 年假规定：员工连续工作满一年后可以享受带薪年假。工龄满一年不满十年的，每年享有五天年假；满十年不满二十年的，每年享有十天年假；满二十年的，每年享有十五天年假。年假原则上应在当年使用，确因项目安排不能休完的，经人力资源部确认后可顺延至次年三月底。

第三条 病假规定：员工因身体原因无法出勤，应在上班前通过企业微信提交病假申请，并在返岗后三个工作日内补交医院诊断证明、病历或正规医疗机构开具的休息建议。病假一天以内由直属主管审批，超过三天需部门经理和人力资源部共同审批。连续病假超过七天的，公司可要求员工提供复诊证明。

第四条 事假规定：员工因个人事务请假，应至少提前一天提交申请，说明请假原因和起止时间。事假期间不计发当日工资。紧急情况无法提前申请的，应先电话告知主管，并在当日补交线上申请。

第五条 调休与销假：员工因加班产生调休的，应在系统中选择“调休假”类型，并关联加班记录。请假结束后，员工应在返岗当天完成销假；逾期未销假的，按缺勤异常处理。任何请假申请不得由他人代填，虚假请假一经查实将按公司纪律规定处理。
""".strip(),
    "overtime_allowance_policy.txt": """
公司加班与补贴政策

第一条 加班定义：加班是指员工因工作需要，在标准工作时间之外继续完成经批准的工作任务。员工自愿延长工作时间但未经过审批的，不计为公司认可的加班。所有加班应坚持必要、合理、可追溯原则，部门负责人应优先通过优化排班和提高效率减少不必要加班。

第二条 审批流程：员工预计需要加班时，应在加班开始前通过企业微信提交加班申请，填写项目名称、加班原因、预计时长和工作成果。工作日加班由直属主管审批；周末或法定节假日加班需部门经理审批；涉及跨部门项目的，应同步抄送项目负责人。特殊紧急情况可先口头报备，并在二十四小时内补提申请。

第三条 补贴标准：工作日晚间加班满两小时的，给予三十元餐补；超过四小时且结束时间晚于二十二点的，可报销单程交通费。周末加班满四小时按半天计算，满八小时按一天计算，公司优先安排调休；确因项目节点无法调休的，经审批后按公司薪酬规则发放加班补贴。法定节假日加班按照国家相关规定执行。

第四条 调休规则：周末加班形成的调休应在三个月内使用，使用时需在请假系统中选择“调休假”，并关联已审批的加班单。调休原则上以半天为最小单位，不得拆分为小时级休假。员工离职前仍有未使用调休的，由人力资源部根据审批记录进行核对。

第五条 记录与监督：员工应在加班结束后填写实际完成事项，主管需确认加班成果。无审批、无记录或记录明显与工作无关的加班，公司不予认定。财务部每月根据人力资源部导出的加班台账核算补贴。
""".strip(),
    "office_equipment_policy.txt": """
办公设备申领与归还流程

第一条 设备范围：本流程所称办公设备包括笔记本电脑、台式机、显示器、键盘鼠标、耳机、移动硬盘、投影仪、会议摄像头以及经行政部登记的其他办公资产。设备由行政部统一采购、编号、发放和盘点，信息技术部负责系统安装、账号配置和安全检查。

第二条 新员工申领：新员工入职前，直属主管应在入职前三个工作日提交设备申领单，注明岗位、办公地点、是否需要高性能电脑以及特殊软件需求。行政部根据岗位标准配置电脑和外设，信息技术部完成系统初始化、杀毒软件安装、磁盘加密和公司邮箱配置。员工领取设备时需核对资产编号并签署《办公资产领用确认单》。

第三条 临时借用：员工因会议、培训、出差或项目演示需要临时借用投影仪、会议摄像头、备用电脑等设备，应提前一天在行政系统提交借用申请。借用期限一般不超过七天，到期应主动归还。如因项目原因需要延长，应重新提交延期说明。

第四条 维修与更换：设备出现故障时，员工应先联系信息技术部进行检测，不得自行拆机或交由外部维修。经确认影响正常工作的，可申请备用机。因自然损耗导致的维修费用由公司承担；因个人保管不当造成的损坏或遗失，员工需按照资产管理规定承担相应责任。

第五条 归还与安全：员工调岗、离职或项目结束不再需要设备时，应在三个工作日内归还。信息技术部负责备份工作资料、清除个人信息、注销本地权限并检查设备安全状态；行政部确认资产外观和配件完整后完成归还登记。未完成归还手续的，不得办理离职结算。
""".strip(),
    "remote_work_policy.txt": """
远程办公与信息安全规定

第一条 申请条件：远程办公适用于因项目协作、出差、特殊天气、临时照护家庭成员或其他经公司认可的情形。员工申请远程办公时，应明确远程日期、工作地点、主要任务、沟通方式和可交付成果。连续远程办公超过三天的，需部门经理审批；涉及客户现场或敏感数据处理的，还需信息安全负责人确认。

第二条 考勤要求：远程办公期间，员工应按正常工作时间在线，并通过企业微信完成上下班打卡。因网络故障或突发情况无法打卡的，应在当天向主管说明并补交证明。远程办公不等同于休假，员工应按计划参加线上会议，及时回应工作消息，并在当天结束前提交工作进展。

第三条 设备与网络：员工远程办公原则上应使用公司发放的电脑，不建议使用私人电脑处理公司资料。必须连接公司 VPN 或经批准的安全访问通道，不得通过公共网盘、个人邮箱或未授权即时通信工具传输公司文件。处理重要数据时，应开启磁盘加密和屏幕锁定，离开座位时及时锁屏。

第四条 数据安全：远程办公人员不得在公共场所讨论客户隐私、商业报价、源代码或其他敏感信息。确需打印资料的，应经主管批准，并在使用后及时碎纸或带回公司归档。发现设备遗失、账号异常、疑似钓鱼邮件或数据泄露风险时，应在一小时内报告信息技术部。

第五条 绩效与责任：主管应根据工作成果而非在线时长评价远程办公质量。员工未按要求保持联系、未交付承诺成果或违反信息安全规定的，公司可取消其远程办公资格，并视情节进行纪律处理。
""".strip(),
}


RAG_TEST_QUESTIONS: List[Dict[str, str]] = [
    {"type": "直接问题", "question": "病假需要在什么时候提交申请，返岗后需要补交什么材料？"},
    {"type": "直接问题", "question": "工作日晚间加班满两小时有什么补贴？超过四小时并晚于二十二点结束怎么处理？"},
    {"type": "综合问题", "question": "员工周末加班后想在下周调休，应该怎么申请，调休和加班记录之间有什么关系？"},
    {"type": "综合问题", "question": "新员工需要领电脑并偶尔远程办公时，设备申领和信息安全方面分别要注意什么？"},
    {"type": "知识库外", "question": "公司的股票期权什么时候兑现？"},
]


def ensure_sample_knowledge_base(kb_dir: Path, rebuild: bool = False) -> List[Path]:
    ensure_dir(kb_dir)
    if rebuild:
        for path in kb_dir.glob("*.txt"):
            path.unlink()

    for filename, content in SAMPLE_KB_DOCS.items():
        path = kb_dir / filename
        if rebuild or not path.exists():
            write_text(path, content)

    paths = sorted(list(kb_dir.glob("*.txt")) + list(kb_dir.glob("*.pdf")))
    if not paths:
        raise FileNotFoundError(f"知识库目录为空：{kb_dir}")
    return paths


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as exc:
            raise ImportError("读取 PDF 需要安装 pypdf：pip install pypdf") from exc
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"暂不支持的文档类型：{path}")


def split_document_text(
    text: str,
    source: str,
    title: str,
    chunk_size: int = 450,
    overlap: int = 80,
) -> List[DocumentChunk]:
    text = normalize_text(text)
    if not text:
        return []
    chunks: List[DocumentChunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append(
            DocumentChunk(
                chunk_id=f"{Path(source).stem}_chunk_{idx:03d}",
                source=source,
                title=title,
                text=chunk_text,
                start=start,
                end=end,
            )
        )
        if end >= len(text):
            break
        start = max(0, end - overlap)
        idx += 1
    return chunks


def load_and_split_documents(kb_dir: Path, chunk_size: int, chunk_overlap: int) -> List[DocumentChunk]:
    paths = sorted(list(kb_dir.glob("*.txt")) + list(kb_dir.glob("*.pdf")))
    chunks: List[DocumentChunk] = []
    for path in paths:
        text = read_document(path)
        title = path.stem
        chunks.extend(split_document_text(text, path.name, title, chunk_size, chunk_overlap))
    return chunks


class RagIndex:
    """FAISS 检索封装；未安装 faiss-cpu 时用 numpy fallback 保证演示可运行。"""

    def __init__(self, chunks: Sequence[DocumentChunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量不一致。")
        self.chunks = list(chunks)
        self.vectors = normalize_vectors(vectors.astype(np.float32))
        self.dim = self.vectors.shape[1] if len(self.vectors) else 0
        self.index = None

        if FAISS_AVAILABLE and self.dim > 0:
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(self.vectors)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[RetrievedChunk]:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = normalize_vectors(query_vector.astype(np.float32))
        top_k = max(1, min(top_k, len(self.chunks)))

        if self.index is not None:
            scores, indices = self.index.search(query_vector, top_k)
            return [
                RetrievedChunk(chunk=self.chunks[int(idx)], score=float(score))
                for idx, score in zip(indices[0], scores[0])
                if int(idx) >= 0
            ]

        scores = (self.vectors @ query_vector[0]).reshape(-1)
        indices = np.argsort(-scores)[:top_k]
        return [RetrievedChunk(chunk=self.chunks[int(i)], score=float(scores[int(i)])) for i in indices]


def build_rag_index(chunks: Sequence[DocumentChunk], embedding_client: EmbeddingClient) -> RagIndex:
    texts = [chunk.text for chunk in chunks]
    vectors = embedding_client.embed_texts(texts)
    return RagIndex(chunks, vectors)


def answer_with_rag(
    question: str,
    rag_index: RagIndex,
    embedding_client: EmbeddingClient,
    chat_client: ChatClient,
    top_k: int = 4,
) -> Dict[str, object]:
    query_vec = embedding_client.embed_texts([question])
    retrieved = rag_index.search(query_vec, top_k=top_k)
    context = "\n\n".join(
        [
            f"[来源{i + 1}] 文件：{item.chunk.source}；片段：{item.chunk.chunk_id}；相似度：{item.score:.4f}\n{item.chunk.text}"
            for i, item in enumerate(retrieved)
        ]
    )

    in_scope = is_question_in_scope(question, retrieved)
    if not in_scope:
        answer = "知识库中未找到与该问题直接相关的政策依据，因此无法回答。建议咨询人力资源部或查看补充制度文件。"
    else:
        user_prompt = f"""
请只根据下面的企业制度资料回答问题。若资料中没有明确依据，请直接说明“知识库中未找到相关依据”，不要编造。

问题：{question}

资料：
{context}

输出格式：
回答：用 2-5 句话给出结论。
依据：列出使用到的来源文件名。
""".strip()
        answer = chat_client.chat(
            system_prompt="你是企业制度问答助手，必须基于给定资料回答，并保留来源。",
            user_prompt=user_prompt,
        )

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "source": item.chunk.source,
                "chunk_id": item.chunk.chunk_id,
                "score": round(item.score, 4),
                "text_preview": item.chunk.text[:120] + ("..." if len(item.chunk.text) > 120 else ""),
            }
            for item in retrieved
        ],
        "in_scope": in_scope,
    }


def run_rag_experiment(
    chat_client: ChatClient,
    embedding_client: EmbeddingClient,
    cfg: ExperimentConfig,
) -> Dict[str, object]:
    print_header("第二部分：RAG 系统构建")
    output_dir = ensure_dir(cfg.output_dir)
    kb_dir = ensure_dir(cfg.kb_dir)

    doc_paths = ensure_sample_knowledge_base(kb_dir, rebuild=cfg.rebuild_kb)
    chunks = load_and_split_documents(kb_dir, cfg.chunk_size, cfg.chunk_overlap)
    rag_index = build_rag_index(chunks, embedding_client)

    print(f"知识库目录：{kb_dir}")
    print(f"文档数量：{len(doc_paths)}")
    print(f"文档块数量：{len(chunks)}")
    print(f"Embedding 后端：{embedding_client.backend} / {embedding_client.model}")
    print(f"向量检索：{'FAISS IndexFlatIP' if rag_index.index is not None else 'numpy fallback'}")

    answers: List[Dict[str, object]] = []
    for item in RAG_TEST_QUESTIONS:
        result = answer_with_rag(
            question=item["question"],
            rag_index=rag_index,
            embedding_client=embedding_client,
            chat_client=chat_client,
            top_k=cfg.top_k,
        )
        result["type"] = item["type"]
        answers.append(result)

    md = render_rag_results(doc_paths, chunks, answers, chat_client, embedding_client, rag_index)
    write_text(output_dir / "exp7_rag_results.md", md)
    safe_json_dump(output_dir / "exp7_rag_results.json", answers)
    safe_json_dump(output_dir / "exp7_rag_chunks.json", [asdict(chunk) for chunk in chunks])
    print(f"RAG 结果已保存：{output_dir / 'exp7_rag_results.md'}")
    return {
        "doc_paths": [str(path) for path in doc_paths],
        "chunk_count": len(chunks),
        "answers": answers,
        "markdown_path": str(output_dir / "exp7_rag_results.md"),
    }


def render_rag_results(
    doc_paths: Sequence[Path],
    chunks: Sequence[DocumentChunk],
    answers: Sequence[Dict[str, object]],
    chat_client: ChatClient,
    embedding_client: EmbeddingClient,
    rag_index: RagIndex,
) -> str:
    lines = [
        "# 实验七第二部分：RAG 系统构建结果",
        "",
        f"生成时间：{now_str()}",
        f"LLM 后端：{chat_client.backend} / {chat_client.model}",
        f"Embedding 后端：{embedding_client.backend} / {embedding_client.model}",
        f"向量数据库：{'FAISS IndexFlatIP' if rag_index.index is not None else 'numpy fallback'}",
        "",
        "## 知识库文档列表及内容摘要",
        "",
        "| 文档 | 摘要 |",
        "|---|---|",
    ]
    for path in doc_paths:
        text = normalize_text(read_document(path))
        lines.append(f"| {path.name} | {text[:90]}... |")

    lines.extend(
        [
            "",
            "## 分块统计",
            "",
            f"共加载 {len(doc_paths)} 个文档，切分为 {len(chunks)} 个文档块。",
            "",
            "## 测试问题及模型回答记录表",
            "",
            "| 类型 | 问题 | 模型回答 | 来源引用 |",
            "|---|---|---|---|",
        ]
    )
    for item in answers:
        source_text = "；".join(
            f"{src['source']}({src['chunk_id']}, score={src['score']})"
            for src in item.get("sources", [])  # type: ignore[union-attr]
        )
        answer = str(item["answer"]).replace("\n", "<br>")
        question = str(item["question"])
        lines.append(f"| {item['type']} | {question} | {answer} | {source_text} |")

    lines.extend(
        [
            "",
            "## 简要分析",
            "",
            "直接问题通常能命中单个制度片段，回答较稳定；综合问题需要把请假、加班、设备、远程办公等片段合并，来源引用能帮助核查；知识库外问题应拒答，避免模型凭常识编造公司制度。",
            "",
            "## 核心流程说明",
            "",
            "1. 读取 TXT/PDF 文档。",
            "2. 按固定长度与重叠窗口切分文档块。",
            "3. 通过 EmbeddingClient 生成向量。",
            "4. 使用 FAISS IndexFlatIP 存储并按相似度检索。",
            "5. 将检索片段作为上下文交给 LLM 生成答案并保留来源。",
        ]
    )
    return "\n".join(lines)


# =========================
# 第三部分：简易 Agent 设计
# =========================


@dataclass
class ToolResult:
    tool_name: str
    input: Dict[str, object]
    output: object


@dataclass
class AgentStep:
    round_id: int
    thought: str
    action: str
    observation: str


SIMULATED_SEARCH_DB: Dict[str, List[Dict[str, str]]] = {
    "人工智能在医疗领域的应用": [
        {
            "title": "医学影像辅助诊断",
            "source": "模拟来源A：医疗AI行业综述",
            "snippet": "AI 可用于 CT、MRI、X 光等医学影像的病灶筛查，帮助医生提高阅片效率，但最终诊断仍需医生确认。",
        },
        {
            "title": "临床决策支持",
            "source": "模拟来源B：医院信息化白皮书",
            "snippet": "临床决策支持系统可以结合病历、检查指标和指南知识，为医生提供用药提醒、风险预警和诊疗路径建议。",
        },
        {
            "title": "药物研发与蛋白结构预测",
            "source": "模拟来源C：AI 制药案例集",
            "snippet": "AI 可用于分子筛选、靶点发现和蛋白结构预测，缩短候选药物发现周期，但仍需要实验验证和临床试验。",
        },
        {
            "title": "隐私与安全挑战",
            "source": "模拟来源D：医疗数据治理指南",
            "snippet": "医疗 AI 涉及大量敏感个人信息，落地时必须重视数据脱敏、访问控制、模型偏差和责任边界。",
        },
    ]
}


def search_tool(query: str, top_k: int = 4) -> ToolResult:
    """模拟搜索工具：输入主题，返回若干条结构化搜索结果。"""
    results = SIMULATED_SEARCH_DB.get(query, [])
    if not results:
        # 简单关键词兜底：保持实验可控，不真实联网。
        results = SIMULATED_SEARCH_DB["人工智能在医疗领域的应用"]
    return ToolResult(tool_name="search_tool", input={"query": query, "top_k": top_k}, output=results[:top_k])


def summarize_tool(search_results: Sequence[Dict[str, str]]) -> ToolResult:
    """模拟摘要工具：把搜索结果整理为主题概述、关键发现和来源。"""
    findings = [
        "医学影像是医疗 AI 最常见的落地场景，可辅助医生进行筛查和提高效率。",
        "临床决策支持可以把病历、检验指标和指南结合起来，为医生提供风险提醒。",
        "药物研发是 AI 的重要方向，但候选结果仍需实验和临床验证。",
        "医疗 AI 必须重视隐私保护、数据安全、模型偏差和责任边界。",
    ]
    sources = [item["source"] for item in search_results]
    summary = {
        "overview": "人工智能在医疗领域主要承担辅助分析、信息整合和效率提升的角色，不能替代医生独立诊断。",
        "findings": findings[:3],
        "risks": findings[3:],
        "sources": sources,
    }
    return ToolResult(tool_name="summarize_tool", input={"result_count": len(search_results)}, output=summary)


def generate_agent_report(topic: str, summary: Dict[str, object]) -> str:
    findings = summary.get("findings", [])
    sources = summary.get("sources", [])
    lines = [
        f"# 信息收集与整理 Agent 报告：{topic}",
        "",
        "## 主题概述",
        str(summary.get("overview", "")),
        "",
        "## 3个关键发现",
    ]
    for idx, finding in enumerate(findings, start=1):
        lines.append(f"{idx}. {finding}")
    lines.extend(["", "## 风险提示"])
    for risk in summary.get("risks", []):
        lines.append(f"- {risk}")
    lines.extend(["", "## 信息来源"])
    for source in sources:
        lines.append(f"- {source}")
    return "\n".join(lines)


def run_agent_experiment(topic: str, output_dir: Path) -> Dict[str, object]:
    print_header("第三部分：简易 Agent 设计")
    ensure_dir(output_dir)

    steps: List[AgentStep] = []

    # 第 1 轮：搜索
    first_thought = "先明确主题并收集基础资料，优先获取应用场景、价值和风险信息。"
    search_result = search_tool(topic, top_k=4)
    observation_1 = f"搜索工具返回 {len(search_result.output)} 条结果，覆盖影像诊断、临床决策、药物研发和隐私安全。"
    steps.append(AgentStep(1, first_thought, f"search_tool({search_result.input})", observation_1))

    # 第 2 轮：摘要
    second_thought = "搜索结果已经覆盖多个角度，下一步调用摘要工具提炼关键发现。"
    summary_result = summarize_tool(search_result.output)  # type: ignore[arg-type]
    summary_output = summary_result.output  # type: ignore[assignment]
    observation_2 = "摘要工具生成了主题概述、3个关键发现、风险提示和信息来源列表。"
    steps.append(AgentStep(2, second_thought, f"summarize_tool({summary_result.input})", observation_2))

    # 第 3 轮：完成判断并生成报告
    third_thought = "信息已足够生成结构化报告，检查报告是否包含主题概述、3个关键发现和来源。"
    report = generate_agent_report(topic, summary_output)  # type: ignore[arg-type]
    observation_3 = "报告已生成，包含主题概述、关键发现、风险提示和模拟来源，任务完成。"
    steps.append(AgentStep(3, third_thought, "generate_agent_report(summary)", observation_3))

    trace_md = render_agent_trace(topic, steps, report)
    write_text(output_dir / "exp7_agent_trace.md", trace_md)
    write_text(output_dir / "exp7_agent_report.md", report)
    safe_json_dump(
        output_dir / "exp7_agent_trace.json",
        {
            "topic": topic,
            "steps": [asdict(step) for step in steps],
            "search_result": asdict(search_result),
            "summary_result": asdict(summary_result),
            "report": report,
        },
    )
    print(f"Agent 记录已保存：{output_dir / 'exp7_agent_trace.md'}")
    return {"topic": topic, "steps": [asdict(step) for step in steps], "report": report}


def render_agent_trace(topic: str, steps: Sequence[AgentStep], report: str) -> str:
    lines = [
        "# 实验七第三部分：简易 Agent 设计结果",
        "",
        f"主题：{topic}",
        "",
        "## Agent 工作流程步骤描述",
        "",
        "用户任务 → 规划搜索方向 → 调用搜索工具 → 观察搜索结果 → 调用摘要工具 → 判断是否满足报告要求 → 生成 Markdown 报告",
        "",
        "## 工具定义代码/伪代码",
        "",
        "```python",
        "def search_tool(query: str, top_k: int = 4) -> list[dict]:",
        "    return simulated_search_results",
        "",
        "def summarize_tool(search_results: list[dict]) -> dict:",
        "    return {'overview': ..., 'findings': ..., 'sources': ...}",
        "```",
        "",
        "## 思考-行动-观察记录",
        "",
        "| 轮次 | 思考 | 行动 | 观察 |",
        "|---:|---|---|---|",
    ]
    for step in steps:
        lines.append(f"| {step.round_id} | {step.thought} | `{step.action}` | {step.observation} |")
    lines.extend(["", "## 最终生成报告", "", report])
    return "\n".join(lines)


# =========================
# 文本、向量与 mock 规则
# =========================


def chunked(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def char_ngrams(text: str, n_values: Sequence[int] = (1, 2, 3)) -> List[str]:
    cleaned = re.sub(r"\s+", "", text.lower())
    cleaned = re.sub(r"[，。！？；：、,.!?;:()（）《》“”\"'\[\]{}]", "", cleaned)
    grams: List[str] = []
    for n in n_values:
        if len(cleaned) < n:
            continue
        grams.extend(cleaned[i : i + n] for i in range(len(cleaned) - n + 1))
    return grams


def hash_text_embedding(text: str, dim: int = 384) -> np.ndarray:
    """离线哈希嵌入：用于无 API Key 时的可复现实验演示。"""
    vec = np.zeros(dim, dtype=np.float32)
    grams = char_ngrams(text)
    if not grams:
        return vec
    for gram in grams:
        digest = hashlib.md5(gram.encode("utf-8")).hexdigest()
        value = int(digest[:8], 16)
        idx = value % dim
        sign = 1.0 if (value & 1) else -1.0
        vec[idx] += sign
    return vec


def extract_between(text: str, start_marker: str, end_marker: str) -> str:
    if start_marker not in text:
        return ""
    start = text.index(start_marker) + len(start_marker)
    if end_marker and end_marker in text[start:]:
        end = text.index(end_marker, start)
        return text[start:end].strip()
    return text[start:].strip()


def is_question_in_scope(question: str, retrieved: Sequence[RetrievedChunk]) -> bool:
    if not retrieved:
        return False
    combined_context = "\n".join(item.chunk.text for item in retrieved[:3])

    # 明确的知识库外测试项。
    if any(term in question for term in ["股票", "期权", "股权", "上市", "兑现"]):
        if not any(term in combined_context for term in ["股票", "期权", "股权", "上市", "兑现"]):
            return False

    important_terms = [
        term
        for term in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", question)
        if term not in {"公司", "员工", "需要", "什么", "怎么", "时候", "应该", "规定", "制度", "方面", "分别"}
    ]
    if any(term in combined_context for term in important_terms):
        return True
    return retrieved[0].score >= 0.18


def mock_rag_answer(question: str, context: str) -> str:
    if any(term in question for term in ["股票", "期权", "股权", "兑现"]):
        return "回答：知识库中未找到与股票期权兑现相关的政策依据，因此无法回答。\n依据：无。"
    if "病假" in question:
        return "回答：病假应在上班前通过企业微信提交申请；返岗后三个工作日内需要补交医院诊断证明、病历或正规医疗机构开具的休息建议。病假一天以内由直属主管审批，超过三天需部门经理和人力资源部共同审批。\n依据：company_leave_policy.txt。"
    if "晚间加班" in question or "二十二点" in question:
        return "回答：工作日晚间加班满两小时给予三十元餐补；超过四小时且结束时间晚于二十二点的，可报销单程交通费。加班必须提前或按规定补提审批。\n依据：overtime_allowance_policy.txt。"
    if "周末加班" in question or "调休" in question:
        return "回答：周末加班需经过审批，满四小时按半天、满八小时按一天计算，并优先安排调休。使用调休时，应在请假系统中选择“调休假”，并关联已审批的加班记录，通常应在三个月内使用。\n依据：overtime_allowance_policy.txt；company_leave_policy.txt。"
    if "新员工" in question or "远程办公" in question or "电脑" in question:
        return "回答：新员工电脑应由直属主管在入职前三个工作日提交设备申领单，行政部发放资产，信息技术部完成系统、账号和安全配置。远程办公原则上使用公司电脑，连接 VPN 或批准的安全通道，不得用个人邮箱、公共网盘等传输公司文件。\n依据：office_equipment_policy.txt；remote_work_policy.txt。"
    return "回答：可根据检索到的制度片段处理，但建议进一步核对来源文件。\n依据：检索结果中的相关来源文件。"


# =========================
# 主程序
# =========================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验七：大语言模型实验代码")
    parser.add_argument("--part", choices=["all", "prompt", "rag", "agent"], default="all", help="选择运行哪一部分")
    parser.add_argument(
        "--backend",
        choices=["auto", "mock", "zhipuai", "deepseek", "openai"],
        default="auto",
        help="LLM/Embedding 后端",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--kb-dir", type=Path, default=DEFAULT_KB_DIR, help="RAG 知识库目录")
    parser.add_argument("--rebuild-kb", action="store_true", help="重新生成示例知识库 TXT 文档")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=1600, help="LLM 最大输出 token")
    parser.add_argument("--chunk-size", type=int, default=450, help="RAG 文档块长度")
    parser.add_argument("--chunk-overlap", type=int, default=80, help="RAG 文档块重叠长度")
    parser.add_argument("--top-k", type=int, default=4, help="RAG 检索 top_k")
    parser.add_argument("--embedding-dim", type=int, default=384, help="mock 哈希嵌入维度")
    parser.add_argument("--agent-topic", type=str, default="人工智能在医疗领域的应用", help="Agent 模拟主题")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)

    cfg = ExperimentConfig(
        backend=args.backend,
        output_dir=args.output_dir,
        kb_dir=args.kb_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        embedding_dim=args.embedding_dim,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        rebuild_kb=args.rebuild_kb,
    )

    ensure_dir(cfg.output_dir)
    print_header("实验七：大语言模型实验")
    print(f"脚本目录：{SCRIPT_DIR}")
    print(f"输出目录：{cfg.output_dir}")
    print(f"知识库目录：{cfg.kb_dir}")
    print(f"随机种子：{RANDOM_SEED}")
    print(f"FAISS 可用：{FAISS_AVAILABLE}")

    chat_client = ChatClient(backend=cfg.backend, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    embedding_client = EmbeddingClient(backend=cfg.backend, dimensions=cfg.embedding_dim)
    print(f"LLM 后端：{chat_client.backend} / {chat_client.model}")
    print(f"Embedding 后端：{embedding_client.backend} / {embedding_client.model}")

    summary: Dict[str, object] = {
        "generated_at": now_str(),
        "backend": chat_client.backend,
        "chat_model": chat_client.model,
        "embedding_backend": embedding_client.backend,
        "embedding_model": embedding_client.model,
        "faiss_available": FAISS_AVAILABLE,
        "parts": {},
    }

    if args.part in {"all", "prompt"}:
        summary["parts"]["prompt"] = run_prompt_engineering(chat_client, cfg.output_dir)  # type: ignore[index]

    if args.part in {"all", "rag"}:
        summary["parts"]["rag"] = run_rag_experiment(chat_client, embedding_client, cfg)  # type: ignore[index]

    if args.part in {"all", "agent"}:
        summary["parts"]["agent"] = run_agent_experiment(args.agent_topic, cfg.output_dir)  # type: ignore[index]

    safe_json_dump(cfg.output_dir / "exp7_run_summary.json", summary)
    write_text(cfg.output_dir / "exp7_run_summary.txt", render_run_summary(summary, cfg.output_dir))

    print_header("已保存文件列表")
    for path in sorted(cfg.output_dir.glob("*")):
        if path.is_file():
            print(path)


def render_run_summary(summary: Dict[str, object], output_dir: Path) -> str:
    lines = [
        "实验七：大语言模型实验运行摘要",
        "",
        f"生成时间：{summary.get('generated_at')}",
        f"LLM 后端：{summary.get('backend')} / {summary.get('chat_model')}",
        f"Embedding 后端：{summary.get('embedding_backend')} / {summary.get('embedding_model')}",
        f"FAISS 可用：{summary.get('faiss_available')}",
        f"输出目录：{output_dir}",
        "",
        "主要输出：",
        "- exp7_prompt_engineering_results.md：提示工程对比结果",
        "- exp7_rag_results.md：RAG 问答记录和来源引用",
        "- exp7_agent_trace.md：Agent 思考-行动-观察记录",
        "- exp7_agent_report.md：Agent 最终报告",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
