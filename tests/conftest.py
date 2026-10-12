"""共享 fixtures：强制确定性后端，保证测试套件完全 hermetic（零网络、零模型）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qiming.config import QimingConfig  # noqa: E402
from qiming.pipeline import Document, QimingPipeline  # noqa: E402


def make_config() -> QimingConfig:
    cfg = QimingConfig.from_env()
    cfg.embed.backend = "hashing"      # 确定性嵌入，无网络
    cfg.index.backend = "numpy"        # 暴力精确索引
    cfg.llm.backend = "mock"           # 确定性生成
    cfg.retrieve.reranker_backend = "cosine"
    return cfg


@pytest.fixture(scope="session")
def cfg() -> QimingConfig:
    return make_config()


@pytest.fixture(scope="session")
def pipeline(cfg) -> QimingPipeline:
    pipe = QimingPipeline(cfg)
    pipe.ingest(
        [
            Document("faq-pw", "用户可以在设置页面重置登录密码，重置链接会发送到注册邮箱。系统每 90 天强制要求更换密码。"),
            Document("faq-export", "数据导出支持 CSV 与 XLSX 格式，导出任务完成后会收到站内通知，下载链接有效期 7 天。"),
            Document("faq-sso", "企业版支持 SAML 2.0 与 OIDC 两种单点登录协议，配置路径为管理后台安全设置。"),
            Document("release", "v3.2 版本完成组件库全面重构，渲染性能提升 40%，包体积减少 28%。"),
        ]
    )
    return pipe
