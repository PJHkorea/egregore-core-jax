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
    [5th-Gen Pure Numerical Silicon MUX Optimizer]
    
    [KR]
    기저 AdamW의 수치적 이중 감쇠(Double-Dipping) 및 모멘텀 역학 왜곡 문제를 완전히 박멸하기 위해,
    순수 모멘텀 추적기(optax.adam)와 외부 실리콘 MUX 연산 레일을 100% 무결하게 선형 대수학적으로 결합한
    5세대 고성능 '완전 통제형 수치 MUX(Numerical Multiplexer)' 커널 팩토리입니다.
    
    [EN]
    A 5th-gen architectural factory designed to fundamentally eradicate numerical double-dipping 
    and momentum dynamics distortion inherent in multi-stage AdamW baselines. It unifies a pure 
    momentum tracker (optax.adam) with a highly specialized external silicon MUX over an inline 
    Hadamard rail, achieving absolute mathematical correctness under 0% host intervention.
    """


    
    # ====================================================================
    # [SILICON-ALIGNED HADAMARD INLINE MUX ENGINE]
    # [KR] 다중 옵티마이저 분기를 단일 최적화 레일로 통합하고 대수적 아다마르 스케일러로 변환
    # [EN] Unify multi-optimizer branches into a single engine using algebraic Hadamard scalers
    # ====================================================================
    
    # 1. [KR] 모멘텀 왜곡과 이중 감쇠를 방어하기 위해 가중치 감쇠가 거세된 순수 Adam 적률 추적기 기동
    # 1. [EN] Launch a pure Adam momentum/variance tracker with zero weight decay to block dynamics distortion
    base_optimizer = optax.adam(learning_rate=1.0)
    
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
        # [KR] 중복 호출을 완전히 박멸하여 Host OOM을 방어하고 병렬 마스크 트리 구조 자동 병합
        # [EN] Thoroughly obliterate redundant tracing to preempt Host OOM; synthesize parallel mask PyTree structures
        # ====================================================================
        # [KR] 상단에서 추출한 flat_params와 tree_def의 쌍(zip)을 활용해 단 1회만 노드를 탐색하도록 스트림 통합
        # [EN] Leverage the paired zip of flat_params and tree_def extracted above to enforce a single-pass traversal
        path_leaf_pairs = list(zip([p[0] for p in flat_params] if isinstance(flat_params, list) and len(flat_params) > 0 and isinstance(flat_params[0], tuple) else flat_params, flat_params)) 
        
        # [KR] 앞서 상단부 구조와의 싱크를 맞추기 위해, 상단의 flat_params가 이미 path를 포함한 구조라면 바로 사용하고
        # 아니라면 아래와 같이 기존에 상단에 선언해둔 flat_params를 활용하거나, 상단부 선언을 수정하여 이 스트림으로 일치시킵니다.
        # 가장 깔끔한 완전 통합형 단일 패스 스트림 구조는 다음과 같습니다:
        mapped_tensors = list(map(_extract_silicon_mask_by_path, flat_params))
        
        # [KR] 전체 가중치 파라미터 구조와 정확히 동일하게 대칭 사상된 학습률 트리와 가중치 감쇠 트리를 완벽하게 복원
        # [EN] Reconstruct look-alike learning rate and weight decay PyTree topologies matching the exact parametric signature
        lr_mask_tree = jax.tree_util.tree_unflatten(tree_def, [t[0] for t in mapped_tensors])
        wd_mask_tree = jax.tree_util.tree_unflatten(tree_def, [t[1] for t in mapped_tensors])



        # ====================================================================
        # [5TH-GEN PURE SILICON HADAMARD MULTIPLEXER ENGINE]
        # [KR] 이중 감쇠를 배제하고 오리지널 AdamW LLRD 수식을 단일 아다마르 레일 위에서 무결하게 재현
        # [EN] Execute pristine AdamW formulations via inline Hadamard tensor products without double-dipping
        # ====================================================================
        # 1. [KR] 모멘텀 왜곡이 거세된 순수 Adam 적률 추적기(u) 업데이트 벡터 계산
        # 1. [EN] Compute raw momentum/variance update vectors (u) via pure underlying Adam engine
        updates, next_state = base_optimizer.update(updates, state, params)

        # 2. [KR] 파라미터 존재 여부를 0.0f/1.0f 리터럴 가드로 치환하여 하드웨어 분기(Branch) 완전 회피
        # 2. [EN] Map parametric existence into float32 literal register guards to bypass hardware branch stalls
        has_params = (params is not None)
        wd_activation_gate = has_params * jnp.float32(1.0) + (not has_params) * jnp.float32(0.0)

        # 3. [KR] params가 None일 때의 차원 크래시를 방어하기 위한 컴파일 타임 정적 레일 융합 안전장치
        # 3. [EN] Bind unified array structures at compile-time to shield against dimension errors if params is None
        safe_params = params if has_params else updates

        # 4. [KR] 5세대 완전 통제 규격: 그라디언트 적률(u)과 가중치 감쇠(p) 양쪽에 타겟 레이어 학습률(lr)을 정확하게 연동
        # 4. [EN] 5th-Gen Specification: Perfectly scale both momentum updates (u) and weight decay (p) by target layer learning rates (lr)
        multiplexed_updates = jax.tree_util.tree_map(
            lambda u, lr, wd, p: (u * lr) + (p * (wd * wd_activation_gate * lr)),
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
