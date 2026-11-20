# ======================================================================
# 文件：public_protection_control.py
# 功能：根据历史反馈和 GOP 字节预算选择公共特征的复制档位。
# 类与函数汇总：
#   - ProtectionDecision：记录期望副本数、实际选择和选择原因。
#   - choose_protection_level：返回预算允许的公共包保护档位。
#   - choose_public_base_profile：在低码率画质档和保护档之间选择。
# 输入：各档计划发送字节、当前预算、上一份有效丢包反馈。
# ======================================================================

from dataclasses import dataclass
from math import isfinite
from typing import Mapping


PROTECTION_LEVELS = (0, 2, 4)


@dataclass(frozen=True)
class ProtectionDecision:
    """一次 GOP 决策。selected_copies 为 None 表示三档均无法发送。"""

    preferred_copies: int
    selected_copies: int | None
    reason: str


def choose_protection_level(
    wire_bytes_by_copies: Mapping[int, int],
    budget_bytes: int,
    prior_loss_fraction: float | None,
) -> ProtectionDecision:
    """
    参数：
        wire_bytes_by_copies：可用档位到完整发送计划字节数的映射，含配置的包开销。
        budget_bytes：当前 GOP 的硬字节上限。
        prior_loss_fraction：上一份有效的已发送包丢包反馈，范围为 [0,1]。
            None 表示启动后尚无反馈。未发送的包不进入丢包率分母。
    返回：
        期望档位、满足预算的实际档位和原因。预算不足时返回跳过决策。
    """
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError("GOP 字节预算必须是非负整数")
    if any(type(copies) is not int or copies not in PROTECTION_LEVELS
           or type(size) is not int or size <= 0
           for copies, size in wire_bytes_by_copies.items()):
        raise ValueError("期望 0、2、4 档位和正整数字节数")
    if prior_loss_fraction is not None and (
        not isfinite(prior_loss_fraction) or not 0 <= prior_loss_fraction <= 1
    ):
        raise ValueError("历史丢包率必须位于 [0,1]")

    # 联调使用的示范阈值。未收到反馈时先保留完整个体特征。
    loss = 0.0 if prior_loss_fraction is None else prior_loss_fraction
    preferred = 0 if loss < 0.05 else 2 if loss < 0.15 else 4
    feasible = [copies for copies, size in wire_bytes_by_copies.items() if size <= budget_bytes]
    if not feasible:
        return ProtectionDecision(preferred, None, "no_level_fits_budget")

    # 预算限制可能只允许更多副本的档位，因为副本替换了略大的个体包。
    selected = min(feasible, key=lambda copies: (abs(copies - preferred), copies))
    reason = "preferred_level" if selected == preferred else "budget_fallback"
    return ProtectionDecision(preferred, selected, reason)



PUBLIC_BASE_PROFILES = ("int4", "pool2_int8")


@dataclass(frozen=True)
class PublicBaseDecision:
    """可交给发送端的编码档位。selected_profile 为 None 时跳过该 GOP。"""

    preferred_profile: str
    selected_profile: str | None
    reason: str


def choose_public_base_profile(
    wire_bytes_by_profile: Mapping[str, int],
    budget_bytes: int,
    prior_loss_fraction: float | None,
) -> PublicBaseDecision:
    """
    参数：
        wire_bytes_by_profile：各档完整发送计划的字节数，含副本和已配置包开销。
        budget_bytes：当前 GOP 字节上限。
        prior_loss_fraction：已发原包的历史丢包估计，None 表示暂无反馈。
    返回：期望档、实际档及原因。期望档超预算时，尝试另一档。
    """
    if (type(budget_bytes) is not int or budget_bytes < 0
            or any(profile not in PUBLIC_BASE_PROFILES or type(size) is not int or size <= 0
                   for profile, size in wire_bytes_by_profile.items())):
        raise ValueError("公共档位或计划字节数无效")
    if prior_loss_fraction is not None and (
        not isfinite(prior_loss_fraction) or not 0 <= prior_loss_fraction <= 1
    ):
        raise ValueError("历史丢包率必须位于 [0,1]")

    # 8% 为本次联调的示范阈值。启动阶段使用画质档。
    preferred = "pool2_int8" if prior_loss_fraction is not None and prior_loss_fraction >= 0.08 else "int4"
    other = "int4" if preferred == "pool2_int8" else "pool2_int8"
    for profile in (preferred, other):
        if profile in wire_bytes_by_profile and wire_bytes_by_profile[profile] <= budget_bytes:
            return PublicBaseDecision(preferred, profile,
                                      "preferred_profile" if profile == preferred else "budget_fallback")
    return PublicBaseDecision(preferred, None, "no_profile_fits_budget")

