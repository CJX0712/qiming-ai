"""可验证性校验测试（I-X1 ~ I-X5）：系统创新内核的硬不变量。"""

from __future__ import annotations

import pytest

from qiming.embeddings import HashingEmbedder
from qiming.verify import ABSTAIN_TEXT, split_sentences, verify


@pytest.fixture(scope="module")
def emb():
    return HashingEmbedder(dim=256)


EVIDENCE = [
    "用户可以在设置页面重置登录密码，重置链接会发送到注册邮箱。系统每 90 天强制要求更换密码。",
    "数据导出支持 CSV 与 XLSX 格式，导出任务完成后会收到站内通知。",
]


class TestVerify:
    def test_i_x1_copied_answer_full_coverage(self, emb):
        """回答逐字摘自某证据（完整子集）→ coverage = 1.0。"""
        answer = EVIDENCE[0] + " [1]"
        rep = verify(answer, EVIDENCE, emb)
        assert rep.coverage == 1.0
        assert rep.passed

    def test_i_x1b_verbatim_sentence_perfect_faithfulness(self, emb):
        """回答每句均与某证据逐字相同 → faithfulness = 1.0。"""
        single = ["用户可以在设置页面重置登录密码，重置链接会发送到注册邮箱。", "数据导出支持 CSV 与 XLSX 格式。"]
        rep = verify(single[0] + " [1]", single, emb)
        assert rep.faithfulness == 1.0
        assert rep.coverage == 1.0
        assert rep.passed

    def test_i_x2_unrelated_answer_rejected(self, emb):
        answer = "火星探测器的推进器采用液态甲烷作为燃料，点火温度极低。 [1]"
        rep = verify(answer, EVIDENCE, emb)
        assert not rep.passed
        assert rep.coverage < 0.6

    def test_i_x3_out_of_range_citation_invalid(self, emb):
        answer = "用户可以在设置页面重置登录密码。 [9]"
        rep = verify(answer, EVIDENCE, emb)
        assert not rep.citations_valid
        assert not rep.passed

    def test_i_x4_empty_evidence_abstains(self, emb):
        rep = verify("任何回答内容。 [1]", [], emb)
        assert not rep.passed
        assert rep.coverage == 0.0

    def test_i_x5_pure_function(self, emb):
        answer = "数据导出支持 CSV 与 XLSX 格式。 [2]"
        r1 = verify(answer, EVIDENCE, emb)
        r2 = verify(answer, EVIDENCE, emb)
        assert r1.coverage == r2.coverage
        assert r1.faithfulness == r2.faithfulness
        assert r1.passed == r2.passed

    def test_empty_answer(self, emb):
        rep = verify("", EVIDENCE, emb)
        assert not rep.passed

    def test_split_sentences(self):
        sents = split_sentences("第一句。第二句！第三句？")
        assert len(sents) == 3

    def test_split_sentences_keeps_version_numbers(self):
        """回归：版本号/小数中的句点不是句子边界。"""
        sents = split_sentences("企业版支持 SAML 2.0 与 OIDC 两种 SSO 协议。配置路径为管理后台。")
        assert len(sents) == 2
        assert sents[0] == "企业版支持 SAML 2.0 与 OIDC 两种 SSO 协议。"

    def test_split_sentences_v32(self):
        assert len(split_sentences("v3.2 版本完成重构，性能提升 40%。")) == 1

    def test_partial_support_blocks_answer(self, emb):
        """半句有据、半句编造 → 覆盖率不足 → 拒答。"""
        answer = "用户可以在设置页面重置登录密码。月球背面有外星人基地。 [1]"
        rep = verify(answer, EVIDENCE, emb)
        assert not rep.passed
        assert rep.coverage == 0.5

    def test_citation_must_match_best_evidence(self, emb):
        """引用张冠李戴：句子证据在 [2] 却标注 [1] → 判无效。"""
        answer = "数据导出支持 CSV 与 XLSX 格式，导出任务完成后会收到站内通知。 [1]"
        rep = verify(answer, EVIDENCE, emb)
        assert not rep.citations_valid

    def test_summary_readable(self, emb):
        rep = verify("用户可以在设置页面重置登录密码，重置链接会发送到注册邮箱。 [1]", EVIDENCE, emb)
        assert "coverage" in rep.summary()
