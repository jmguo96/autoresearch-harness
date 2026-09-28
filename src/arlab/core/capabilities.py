"""One permission view for model discovery, schemas and actual tool dispatch.

Disabled capabilities are removed, not advertised as tools which always fail.
The full Decision remains the on-disk format so old event records stay readable.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Union, get_args

from pydantic import Field, create_model

from arlab.core.schema import Decision, ResearchContract, StrictModel
from arlab.core.tool_catalog import TOOLS

NETWORK_TOOLS = frozenset({
    "search_literature", "search_web", "fetch_source",
    "search_repositories", "read_repository",
})


# 单一能力视图同时驱动模型工具目录、动作 schema 和实际执行授权。
# 研究访问、通用网页搜索与 Python 候选执行是不同权限。
def enabled_tools(contract: ResearchContract) -> dict[str, str]:
    names = set(TOOLS)
    if not contract.exploration.adaptive:
        names.discard("continue_trial")
    if not contract.research_access.enabled:
        names -= NETWORK_TOOLS
    if not contract.research_access.web_search_enabled:
        names.discard("search_web")
    if contract.code_execution.mode == "disabled":
        names.discard("edit_algorithm")
    return {name: description for name, description in TOOLS.items() if name in names}


@lru_cache(maxsize=16)
def _schema(names: tuple[str, ...]):
    variants = [cls for cls in get_args(Decision.model_fields["action"].annotation)
                if get_args(cls.model_fields["kind"].annotation)[0] in names]
    action = Annotated[Union[tuple(variants)], Field(discriminator="kind")]
    return create_model(
        "AvailableDecision", __base__=StrictModel,
        summary=(str, Field(min_length=1, max_length=2000)), action=(action, ...),
    )


def decision_schema(contract: ResearchContract):
    return _schema(tuple(enabled_tools(contract)))


def authorize_action(contract: ResearchContract, kind: str) -> None:
    if kind not in enabled_tools(contract):
        raise PermissionError(f"Tool {kind!r} is disabled by the owner; no request was executed")
