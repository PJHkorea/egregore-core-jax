# ====================================================================
# Copyright (c) 2026 PJHkorea | Licensed under the Apache License, Version 2.0
# ====================================================================

import jax
import jax.numpy as jnp

@jax.jit(static_argnums=(1,))
def calculate_jacobian_spatial_curvature(x_tensor: jax.Array, spatial_dim: int) -> jax.Array:
    """Calculates spatial macro-curvature for N-D batch inputs.

    [KR] 고차원 다양체 공간의 공간적 매크로 곡률(기하학적 분산 대리치)을 계산하는 순수 수리 연산 커널입니다.
         spatial_dim을 가속기 레벨의 정적 상수로 강제 구속하여 컴파일 타임의 구조적 무결성을 수호합니다.

    [EN] A pure mathematical computation kernel that evaluates spatial macro-curvature (geometric variance proxy).
         It locks spatial_dim as an accelerator-level static constant to preserve structural integrity at compile-time.

    Args:
        x_tensor: 
            [KR] 공간 곡률을 역산할 임의의 N차원 배치 입력 텐서 사양입니다 (예: [Batch, Dimension] 또는 [Batch, Time, Dimension]).
            [EN] An arbitrary N-dimensional batch input tensor specification to back-calculate spatial curvature (e.g., [B, D] or [B, T, D]).
        spatial_dim: 
            [KR] 정적 컴파일 궤적을 확정하고 하드웨어 메모리 정렬을 제어하는 특징 공간 고유 차원 크기입니다.
            [EN] An intrinsic feature space dimension size that finalizes the static compilation trajectory and governs hardware memory alignment.

    Returns:
        jax.Array: 
            [KR] 원본 입력의 특징 축(Last Axis)만 1차원 수축 뷰로 변환되어 지연 평가 스트림으로 출력되는 곡률 텐서입니다.
            [EN] A curvature tensor where only the last feature axis of the original input is reduced to a 1D view, driven into the lazy evaluation stream.
    """
    # [KR] 역산 후의 배치 차원 복원 및 정적 형상 보존을 위해 원본 입력 텐서의 차원 사양(Shape)을 컨텍스트로 확보합니다.
    # [EN] Capture the structural dimension specifications (Shape) of the original input tensor to preserve static metadata and facilitate output restoration.
    original_shape = x_tensor.shape
    
    # [KR] [인프라 최적화] 임의의 N차원 배치([B, D], [B, T, D] 등) 입력이 인입되더라도 정적 가상 뷰(Static Virtual View) 레벨로 처리될 수 있도록,
    #      정적 인자로 확정된 spatial_dim 사양에 맞춘 인라인 연속 메모리 레이아웃(2D Matrix Flattening)으로 평탄화 처리를 결합합니다.
    # [EN] [Infra Optimization] To seamlessly unify arbitrary N-dimensional batch inputs (e.g., [B, D] or [B, T, D]) into a static virtual view level,
    #      execute inline contiguous memory flattening (2D Matrix Reshaping) aligned precisely with the statically constrained spatial_dim specification.
    flattened_matrix = jnp.reshape(x_tensor, (-1, spatial_dim))
    
    # [KR] [인프라 최적화] 특징 공간 공간축(Spatial Axis) 기준의 매크로 평균 통계량을 도출합니다. 
    #      keepdims=True 구조를 고수하여 후속 편차 감산 연산 시 하드웨어 레벨의 브로드캐스팅(Broadcasting) 매핑 무결성을 사수합니다.
    # [EN] [Infra Optimization] Evaluate the macro-statistical mean trajectory along the spatial feature axis.
    #      Enforcing keepdims=True guarantees structural broadcasting mapping integrity during subsequent subtraction stages at the hardware level.
    batch_axis_mean = jnp.mean(flattened_matrix, axis=0, keepdims=True)
    
    # [KR] 평탄화된 원시 행렬에서 특징 공간 평균을 감산하여 대수적 기하 변형 편차(Mean-centered Delta) 벡터를 도출합니다.
    #      이 과정은 조건 분기문(if/else) 없이 완전한 선형 대수 연산 체인으로 통합되어 XLA 그래프 파편화를 영구 방어합니다.
    # [EN] Subtract the spatial mean baseline from the flattened matrix to derive the mean-centered algebraic transformation delta vector.
    #      This pipeline is integrated via pure linear algebraic streams without explicit branch predicates, preventing XLA graph fragmentation.
    mean_centered_delta = flattened_matrix - batch_axis_mean
    
    # [KR] [인프라 최적화] 편차 벡터의 원소별 제곱합을 수행하고 특징 축(axis=-1)을 기준으로 하드웨어 온칩 리덕션을 유도합니다.
    #      정적으로 고정된 spatial_dim 상수로 나누어 수치적 특이점을 제거한 안정적인 공간적 분산 대리치(Macro Curvature Proxy)를 도출합니다.
    # [EN] [Infra Optimization] Execute element-wise squaring of the delta vector and drive hardware on-chip reduction along the feature axis (axis=-1).
    #      Dividing by the statically frozen spatial_dim constant derives a numerically stable spatial macro-curvature proxy (geometric variance).
    flat_kappa = jnp.sum(jnp.square(mean_centered_delta), axis=-1, keepdims=True) / spatial_dim
    
    # [KR] [인프라 최적화] 복원할 타겟 차원의 튜플 연산을 컴파일러가 정적 구조로 안전하게 인지하도록 
    #      기존의 동적 슬라이싱 결합 대신 고정된 1차원 확장 사상(Shape Reconstruction) 구조로 리팩토링합니다.
    # [EN] [Infra Optimization] Refactor the dynamic slicing tuple concatenation into a rigid 1D extension layout
    #      to ensure the compiler safely recognizes the target output shape as a static structural construct at compile-time.
    target_output_shape = original_shape[:-1] + (1,)
    
    return jnp.reshape(flat_kappa, target_output_shape)
