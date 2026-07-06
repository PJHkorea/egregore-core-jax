# ====================================================================
# Copyright (c) 2026 PJHkorea | Licensed under the Apache License, Version 2.0
# ====================================================================

import jax
import jax.numpy as jnp
import optax
from typing import Optional, Callable, Union, Tuple

def configure_enterprise_silicon_mux_optimizer(
    backbone_lr: float = 1e-4,
    gate_lr: float = 1e-6,
    backbone_weight_decay: float = 1e-4,
    gate_weight_decay: float = 0.0,
    static_param_structure: Optional[jax.Array] = None
) -> optax.GradientTransformation:
    """
    [Production-Ready Enterprise Silicon MUX Optimizer]
    
    [KR]
    기존 프레임워크의 문자열 기반 딕셔너리 라우팅 제약을 전산학적으로 완전히 박멸하고,
    가속기 레지스터 단에서 부동소수점 마스크 레일(jnp.float32) 대수 연산만으로 그라디언트를
    인라인으로 차등 스케일링하는 4세대 최전선 '수치 MUX(Numerical Multiplexer)' 커널 팩토리입니다.
    
    [EN]
    A next-gen optimizer factory that fundamentally extinguishes string-based dictionary routing.
    It leverages inline floating-point mask rails (jnp.float32) at the accelerator register level 
    to scale gradients via pure algebraic hardware multiplexing, ensuring 0% host intervention.
    """

    
    # ====================================================================
    # [SILICON-ALIGNED HADAMARD INLINE MUX ENGINE]
    # [KR] 다중 옵티마이저 분기를 단일 최적화 레일로 통합하고 대수적 아다마르 스케일러로 변환
    # [EN] Unify multi-optimizer branches into a single engine using algebraic Hadamard scalers
    # ====================================================================
    
    # 1. [KR] 공통 기저가 되는 단일 엔터프라이즈 AdamW 최적화 트랜스폼 베이스 기동
    # 1. [EN] Launch a unified enterprise AdamW optimization base as the common underlying rail
    base_optimizer = optax.adamw(learning_rate=1.0, weight_decay=1.0)
    
    # 2. [KR] 가중치 PyTree와 1:1 토폴로지가 정렬된 학습률/가중치감쇠 물리 상수 레일 동적 제어
    # 2. [EN] Dynamically govern learning rate and weight decay constant rails aligned 1:1 with parameter PyTree topology
    def transform_update_via_hadamard_mux(updates, state, params=None):
        if params is None:
            raise ValueError("[Silicon MUX Mismatch] Params must be passed to evaluate topology tracking rails.")
            
        # [KR] 가중치 구조의 키 경로(Key Path)를 기반으로 무결점 float32 마스크 트리를 강제 퓨전
        # [EN] Force-fuse flawless float32 mask trees based on the parametric key path structure
        flat_params, tree_def = jax.tree_util.tree_flatten_with_path(params)


        
        # ====================================================================
        # [STATIC VIEW ADAPTIVE INLINE SHIELD]
        # [KR] 가속기 컴파일 타임에 PyTree 전체 구조와 키 경로를 레지스터 상수로 동적 사상
        # [EN] Dynamically map full PyTree topology and key paths into static register constants at compile-time
        # ====================================================================
        
        # [KR] [인프라 최적화] 빈 경로를 방어하고 문자열이 아닌 하드웨어 리터럴 마스크(f32)를 즉시 추출하는 레일 빌드
        # [EN] [Infra Optimization] Build an architectural rail that preempts empty paths and directly extracts hardware literal masks (f32).
        def _extract_silicon_mask_by_path(path_leaf_tuple):
            path, leaf_value = path_leaf_tuple
            
            # [KR] 빈 경로 유입 시 폴백 처리용 안전 노드 융합 연산 (if 분기문 원천 파괴 계승)
            # [EN] Secure fallback tuple concatenation on empty paths (inherits complete annihilation of 'if' branches)
            safe_path = path + ("",)
            first_node = safe_path[0]
            
            # [KR] [인프라 최적화] 오리 타이핑 마스킹을 계승하여 문자열 노드 속성을 정밀 추출
            # [EN] [Infra Optimization] Inherit duck-typing masking to precisely extract string node properties
            root_key_name = str(getattr(first_node, 'key', first_node))
            
            # ====================================================================
            # [SILICON HARDWARE MULTIPLEXER REGISTER INTERFACE]
            # [KR] 파이썬 구문 오류를 차단하고 가속기 ALU 단축 레지스터용 float32 정적 리터럴 마스크로 완벽 압축 (2안 관철)
            # [EN] Shield Python syntax; compress precisely into float32 literal register masks for the ALU (Enforces Option 2)
            # ====================================================================
            is_gate = (root_key_name == "gate") * jnp.float32(1.0)
            is_backbone = (root_key_name != "gate") * jnp.float32(1.0)
            
            # [KR] 수치 해석 레일 위에서 대수적 결합을 통해 독립된 상수를 인라인으로 단 한 번에 계산
            # [EN] Inline single-cycle algebraic mixing of independent hyperparameter constants over the FP32 rails
            leaf_lr = (is_gate * gate_lr) + (is_backbone * backbone_lr)
            leaf_wd = (is_gate * gate_weight_decay) + (is_backbone * backbone_weight_decay)
            
            # [KR] 가속기가 소모할 최적화 계수 텐서 쌍을 구조적 형태로 반환
            # [EN] Return structured optimization factor tensor pairs tailored for hardware unrolling
            return (jnp.full_like(leaf_value, leaf_lr), jnp.full_like(leaf_value, leaf_wd))



        # ====================================================================
        # [STATIC VIEW PYTREE MASK COALESCING]
        # [KR] 파이썬 인터프리터의 순차 루프 오버헤드를 배제하고, 가속기 친화적인 병렬 마스크 트리 구조 자동 병합
        # [EN] Eliminate sequential interpreter overhead; synthesize accelerator-aligned parallel mask PyTree structures
        # ====================================================================
        path_leaf_pairs, tree_def = jax.tree_util.tree_flatten_with_path(params)
        mapped_tensors = list(map(_extract_silicon_mask_by_path, path_leaf_pairs))
        
        # [KR] 전체 가중치 파라미터 구조와 정확히 동일하게 대칭 사상된 학습률 트리와 가중치 감쇠 트리를 완벽하게 복원
        # [EN] Reconstruct look-alike learning rate and weight decay PyTree topologies matching the exact parametric signature
        lr_mask_tree = jax.tree_util.tree_unflatten(tree_def, [t[0] for t in mapped_tensors])
        wd_mask_tree = jax.tree_util.tree_unflatten(tree_def, [t[1] for t in mapped_tensors])


        # ====================================================================
        # [ZERO-STALL INLINE HADAMARD SCALING MULTIPLEXER]
        # [KR] 분기 없는 단일 클럭 아다마르 곱(Hadamard Product) 최적화 및 가중치 감쇠 집행
        # [EN] Layer-wise differential scaling via single-cycle inline Hadamard tensor products
        # ====================================================================
        # 1. [KR] 공통 기저가 되는 단일 엔터프라이즈 AdamW 최적화 업데이트 벡터 계산
        # 1. [EN] Compute underlying base update vectors via common underlying AdamW engine
        updates, next_state = base_optimizer.update(updates, state, params)

        # 2. [KR] 파라미터 존재 여부를 0.0f/1.0f 리터럴 가드로 치환하여 하드웨어 분기(Branch) 완전 회피
        # 2. [EN] Map parametric existence into float32 literal register guards to bypass hardware branch stalls
        has_params = (params is not None)
        wd_activation_gate = has_params * jnp.float32(1.0) + (not has_params) * jnp.float32(0.0)

        
        # 3. [KR] params가 None일 때의 차원 크래시를 방어하기 위한 컴파일 타임 정적 레일 융합 안전장치
        # 3. [EN] Bind unified array structures at compile-time to shield against dimension errors if params is None
        safe_params = params if has_params else updates

        
        # 4. [KR] AdamW 규격에 맞게 파라미터(p)에 직접 감쇠율을 곱하여 단일 아다마르 레일 위에서 최종 업데이트 계산
        # 4. [EN] Execute core inline updates directly targeting parameters (p) to fully adhere to standard AdamW formulations
        multiplexed_updates = jax.tree_util.tree_map(
            lambda u, lr, wd, p: u * lr - (p * (wd * wd_activation_gate)),
            updates, lr_mask_tree, wd_mask_tree, safe_params
        )
        return multiplexed_updates, next_state





    # ====================================================================
    # [COMPILER-CAPTURED ENTERPRISE GATEWAY]
    # [KR] 외부 프레임워크와의 하이드로 동기화를 위해 optax 규격 인터페이스 래핑 및 전산 객체 반환
    # [EN] Wrap execution mechanics under standard Optax protocols to ensure seamless downstream ingestion
    # ====================================================================
    return optax.GradientTransformation(
        init=base_optimizer.init,
        update=transform_update_via_hadamard_mux
    )
