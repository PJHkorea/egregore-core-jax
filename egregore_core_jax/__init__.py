# ====================================================================
# [KR] Egregore-Core-JAX: 고성능 수치 가드레일 및 인프라 최적화 마스터 패키지
# [EN] Egregore-Core-JAX: High-Performance Numerical Guardrails & Optimization Infrastructure Package
# Copyright (c) 2026 PJHkorea | Licensed under the Apache License, Version 2.0
# ====================================================================

from .optimizers import configure_enterprise_llrd_optimizer
from .geometry import calculate_jacobian_spatial_curvature
from .math_guardrails import execute_smooth_leaky_guardrail

__version__ = "0.1.0"

# [KR] 외부 노출 퍼블릭 API 사양을 명시적으로 바인딩하여 트리 쉐이킹 및 런타임 가독성을 극대화합니다.
# [EN] Explicitly bind the public API specifications to maximize tree-shaking efficacy and runtime readability.
__all__ = [
    "configure_enterprise_llrd_optimizer",
    "calculate_jacobian_spatial_curvature",
    "execute_smooth_leaky_guardrail",
]
