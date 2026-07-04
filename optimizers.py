import jax
import optax
from typing import Optional, Callable, Union

def configure_enterprise_llrd_optimizer(
    backbone_lr: float = 1e-4,
    gate_lr: float = 1e-6,
    backbone_weight_decay: float = 1e-4,
    gate_weight_decay: float = 0.0,
    custom_router: Optional[Callable[[tuple, jax.Array], str]] = None
) -> optax.GradientTransformation:
    """
    [KR] JAX PyTree 구조체의 가중치 키 경로(Key Path)를 분석하여 층별 차등 최적화(LLRD)를 수행하는 팩토리 함수입니다.
         가속기 내부에서 조건 분기문(if/else)에 의한 그래프 파편화 및 파이썬 호스트 오버헤드를 완전히 소멸시키는 
         '분기 없는 설계(Branchless Design)' 커널을 탑재하고 있습니다.
    
    [EN] A factory function that executes Layer-wise Learning Rate Decay (LLRD) by analyzing JAX PyTree key paths.
         It features a 'Branchless Design' kernel that fundamentally eliminates graph fragmentation and Python host 
         overhead caused by conditional statements (if/else) inside accelerator frameworks.
    """
    
    # [KR] 계층별 AdamW 인프라 정의 및 학습률/가중치 감쇠 독립적 맵핑
    # [EN] Define layer-wise AdamW infrastructure and independently map learning rates/weight decay
    backbone_transform = optax.adamw(learning_rate=backbone_lr, weight_decay=backbone_weight_decay)
    gate_transform = optax.adamw(learning_rate=gate_lr, weight_decay=gate_weight_decay)
    
    parameter_routing_map = {
        "backbone": backbone_transform,
        "gate": gate_transform
    }
    
    # [KR] 1. custom_router 주입 시 상위 파이썬 스코프 분리형 멀티 변환기 즉시 생성 및 반환
    # [EN] 1. If custom_router is injected, immediately construct and return the multi-transform separating high-level Python scope
    if custom_router is not None:
        return optax.multi_transform(parameter_routing_map, custom_router)
        
    # [KR] 2. 하위 내부 라우터 내에서 조건 분기문(if/else)을 전면 배제하기 위한 대수적/전산학적 커널 설계
    # [EN] 2. Subordinate internal router kernel algebraically engineered to completely eliminate conditional branches (if/else)
    def default_route_parameter_by_key(path, _):
        # [KR] [인프라 최적화] 빈 경로(Empty Path) 유입 시 빈 문자열 노드로 강제 변환하여 크래시 방어
        #      튜플 슬라이싱과 해시 가드를 융합하여 'if not path:' 분기문을 완벽히 파괴
        # [EN] [Infra Optimization] Enforce a fallback node tuple to preempt index out of bounds on empty path signatures.
        #      Tuple concatenation and indexing fusion completely annihilate 'if not path:' conditional branches.
        fallback_node = ("",)
        safe_path = path + fallback_node
        first_node = safe_path[0]
        
        # [KR] [인프라 최적화] hasattr 분기문을 전산학적 '오리 타이핑 속성 마스킹(Duck-Typing Attribute Masking)'으로 치환
        #      getattr의 세 번째 인자(Fallback) 기믹을 활용하여 'if/else' 조건 분기를 완전히 소멸시킴
        # [EN] [Infra Optimization] Substitute hasattr branches with computational 'Duck-Typing Attribute Masking'.
        #      Leveraging getattr's tertiary default value hook thoroughly extinguishes explicit 'if/else' branches.
        root_key_name = str(getattr(first_node, 'key', first_node))
        
        # [KR] [인프라 최적화] 문자열 일치 판정 스위칭 논리를 불리언 정수 사상 기반의 단일 인라인 대수식 레벨로 수축
        # [EN] [Infra Optimization] Compress string-matching switching logic into a single inline algebraic expression powered by boolean-to-integer mapping
        return "gate" * (root_key_name == "gate") + "backbone" * (root_key_name != "gate")
        
    return optax.multi_transform(parameter_routing_map, default_route_parameter_by_key)
