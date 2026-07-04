import jax
import jax.numpy as jnp

@jax.jit(static_argnames=("safety_epsilon", "boundary_margin", "leaky_slope"))
def execute_smooth_leaky_guardrail(
    x_tensor: jax.Array,
    safety_epsilon: float = 1e-7,
    boundary_margin: float = 0.95,
    leaky_slope: float = 0.01
) -> jax.Array:
    """Clamps input ranges and preserves backpropagation gradients at boundaries.

    [KR] arccos 등 초월함수 연산의 입력 범위를 안전 영역으로 가두고(Clamping), 
         임계 경계면을 벗어날 때 역전파(Backpropagation) 미분 기울기가 완전히 사멸하는 현상을 방어하는 수치 안정성 커널입니다.

    [EN] A numerical stability kernel that safely clamps input ranges for transcendental operations (e.g., arccos) 
         and prevents backpropagation gradients from completely vanishing when exceeding critical boundaries.

    Args:
        x_tensor: 
            [KR] 수치 안정성 가드레일을 적용할 가속기 장치 내의 입력 데이터 텐서입니다.
            [EN] An input data tensor within the accelerator device targeted for numerical stability guardrails.
        safety_epsilon: 
            [KR] 부동소수점 오차로 인한 경계 오버플로우 및 싱큘래리티(Singularity) 폭발을 차단하는 정밀도 마진 상수입니다.
            [EN] A precision margin constant that blocks boundary overflow and singularity explosions caused by floating-point errors.
        boundary_margin: 
            [KR] 미세 기울기(Leaky Slope) 선형 확장이 가동될 임계 소프트 영역을 결정하는 스케일 비율입니다.
            [EN] A scale ratio determining the critical soft zone where the leaky slope linear extrapolation is activated.
        leaky_slope: 
            [KR] 임계 범위를 초과한 데이터에 부여하여 오차 복원 그레디언트를 영구 보존하는 미세 기울기 계수입니다.
            [EN] A small gradient coefficient assigned to out-of-bound data to permanently preserve error-restoration gradients.

    Returns:
        jax.Array: 
            [KR] 하드웨어 레벨에서 안전 범위 내로 정렬 및 클리핑 처리가 완료된 무결성 텐서입니다.
            [EN] An integrity-guaranteed tensor aligned and clipped safely within valid ranges at the hardware level.
    """

    # [KR] [수치 안정성] 하드/소프트 임계 영역 내에서 데이터의 절대 크기를 산출하고, 
    #      NaN 방어 및 선형 확장을 위한 경계치를 확정합니다.
    # [EN] [Numerical Stability] Compute absolute magnitude and set thresholds (hard/soft) 
    #      for boundary evaluation and safe linear extrapolation.
    absolute_x = jnp.abs(x_tensor)
    upper_numerical_bound = 1.0 - safety_epsilon
    critical_threshold = boundary_margin * upper_numerical_bound
    
    # [KR] [경사 소실 방지] 임계치를 초과하는 영역에 대해 선형 확장 기울기(Leaky Slope)를 적용하여 
    #      역전파 시 경사(Gradient)가 끊기지 않고 흐르도록 유도합니다.
    # [EN] [Gradient Preservation] Apply a linear slope to out-of-bound values, 
    #      ensuring continued gradient flow during backpropagation.
    restoration_gradient_delta = absolute_x - critical_threshold
    leaky_extrapolated_value = critical_threshold + (leaky_slope * restoration_gradient_delta)
    
    # [KR] [분기 없는 가속] 조건문(if) 없이 jnp.sign()과 곱연산을 통해 원본의 부호 정보를 
    #      선형 확장된 값에 결합하여 XLA 가속 환경에 최적화합니다.
    # [EN] [Branchless Optimization] Multiply by jnp.sign() to re-apply the original sign,
    #      enabling efficient calculation without branching (if/else) for XLA acceleration.
    signed_leaky_extension = jnp.sign(x_tensor) * leaky_extrapolated_value

    # [KR] [인프라 최적화] 하드웨어 가속기(GPU/TPU) 내부의 병렬 마스킹 기믹을 사용하여,
    #      임계 영역 내에 있는 원본 값과 임계 영역을 벗어나 선형 확장된 성분을 조건 분기 없이 고속으로 병합합니다.
    # [EN] [Infra Optimization] Leverage parallel masking mechanisms natively inside the hardware accelerator (GPU/TPU)
    #      to merge valid original values with the leaky extrapolated values without triggering hardware branch stalls.
    leaky_cos_matrix = jnp.where(
        absolute_x < critical_threshold,
        x_tensor,
        signed_leaky_extension
    )
    
    # [KR] [하드 가드레일] 소수점 자릿수 내림 오차 등으로 인해 드물게 임계 범위를 완전히 이탈하는 값을
    #      최종 안전 경계선([-upper, +upper]) 내로 강제 잠금(Clip)하여 acos 등 초월함수의 수치적 파괴를 원천 차단합니다.
    # [EN] [Hard Guardrail] Clamp the merged tensor tightly into safe numeric limits ([-upper, +upper]) 
    #      to eliminate infinitesimal truncation overflows, fundamentally blocking mathematical breakdown in transcendental functions.
    return jnp.clip(leaky_cos_matrix, a_min=-upper_numerical_bound, a_max=upper_numerical_bound)
