"""启明 · LLM 客户端层。

**零重写 / 集大成**：
- ``OpenAILLM``：直接复用 OpenAI 官方 SDK，对接任何 OpenAI 兼容端点
  （llama.cpp / vLLM / Ollama / DeepSeek / 通义 / Moonshot / 本地网关）。
  这是 2024 年以来行业事实上标准接口，零重写。
- ``MockLLM``：确定性抽取式回答（取最相关上下文的首句 + 引用标注），
  用于离线测试与基准评测的**可复现基线**——不依赖任何网络与密钥，
  整个测试套件因此完全 hermetic。

生成提示词采用引用约束（citation-constrained prompting）：
要求模型逐句标注引用编号 [n]，资料不足时必须回答"无法回答"，
这是可验证性（faithfulness）的第一道防线。
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

from qiming.embeddings import content_tokens
from qiming.textutil import split_sentences

logger = logging.getLogger("qiming.llm")

SYSTEM_PROMPT = """你是启明（Qiming）系统的回答引擎。严格遵守：
1. 只能依据下面给定的资料片段回答，禁止使用片段之外的任何知识。
2. 每一句陈述后面必须标注引用编号，格式为 [n]（n 为资料片段序号）。
3. 资料不足以回答时，必须且只能回答：无法回答
4. 用简体中文，简洁、准确。"""


class LLM(Protocol):
    name: str

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        """基于给定资料片段生成回答。"""
        ...


class MockLLM:
    """确定性抽取式 LLM（测试 / 基准基线）。

    行为模拟一个**遵循指令**的 LLM：
    - 问题与最相关资料的**实义词**（滤除 的/是 等功能字）共享数 < 2
      → 回答"无法回答"（对应真实 LLM 在系统提示词约束下的拒答行为）
    - 否则取最相关资料的前两句作为回答并标注 [1]

    输出完全确定，保证测试套件与基准评测可复现。
    """

    name = "mock-extractive-v1"
    MIN_SHARED_TOKENS = 2

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        if not contexts:
            return "无法回答"
        q_tokens = content_tokens(question)
        c_tokens = content_tokens(contexts[0])
        if len(q_tokens & c_tokens) < self.MIN_SHARED_TOKENS:
            return "无法回答"
        text = contexts[0]
        sentences = [s.strip() for s in split_sentences(text) if s.strip()]
        answer = "".join(sentences[:2])
        return f"{answer} [1]" if answer else "无法回答"


class OpenAILLM:
    """OpenAI 兼容 LLM 客户端。"""

    name = "openai-compatible"

    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        api_key: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAILLM 需要 api_key（环境变量 QIMING_LLM_API_KEY）")
        from openai import OpenAI  # 延迟导入

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.name = f"openai({model}@{base_url})"

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        user_prompt = f"资料片段：\n{context_block}\n\n问题：{question}"
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


def build_llm(backend: str = "mock", **kwargs) -> LLM:
    """按配置构造 LLM。"""
    if backend == "mock":
        return MockLLM()
    if backend == "openai":
        return OpenAILLM(**kwargs)
    raise ValueError(f"未知 LLM backend: {backend}")
