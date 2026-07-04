
## 🧩 Core Modules Specifications (핵심 모듈 명세)

### 1. 🛠️ `optimizers.py` — Branchless PyTree LLRD Engine
JAX/Optax 유저들을 위해 인터프리터의 조건 분기 개입을 제거한 초고속 계층별 차등 최적화(LLRD) 팩토리 모듈입니다.

*   **[KR] 목적 및 기능**: 복잡한 PyTree 가중치 사전 스캔 시 발생하던 파이썬 호스트 오버헤드와 `hasattr` 분기 예측 실패(Branch Stall)를 'getattr 오리 타이핑 속성 마스킹'과 '삼항 대수식 마스크'를 통해 평탄화했습니다. 빈 경로 유입 크래시를 방어하는 `safe_path` 융합 기술이 탑재되어 있으며, `custom_router` 의존성 주입(DI)을 지원하여 LLM 등 모든 JAX 모델에 범용 결합을 위함이 목적입니다.
*   **[EN] Purpose & Function**: A high-velocity Layer-wise Learning Rate Decay (LLRD) factory module crafted to eliminate conversational python interpreter overhead for JAX/Optax developers. It flattens the conventional `hasattr` branch stalls during complex PyTree scanning by utilizing 'getattr Duck-Typing Attribute Masking' and inline algebraic string-matching. Equipped with a robust `safe_path` concatenation technique to safeguard against empty path crashes, it offers a `custom_router` dependency injection interface for universal integration into any JAX-based architecture, including LLMs.

 - **💡 TL;DR (핵심 요약)**:
 - **[KR] 문제와 해법**: JAX에서 레이어별로 학습률을 다르게 줄 때(LLRD), 가중치 트리 경로를 스캔하느라 파이썬의 `if/else` 조건문을 쓰게 되면 GPU/TPU 컴파일 그래프가 조각나고 병목이 발생합니다. 이 모듈은 조건 분기를 완전히 파괴하고 `"gate" * (is_gate) + "backbone" * (not_is_gate)` 같은 **인라인 대수식 문자열 연산**만으로 파라미터를 하드웨어 레벨에서 초고속 라우팅합니다.
 - **[EN] Problem & Solution**: When implementing Layer-wise Learning Rate Decay (LLRD) in JAX, using standard Python `if/else` predicates to scan weight tree paths fragments the GPU/TPU compilation graph and triggers severe branch stalls. This module thoroughly obliterates conditional branching, utilizing inline **algebraic string-manipulation** like `"gate" * (is_gate) + "backbone" * (not_is_gate)` to execute ultra-high-speed parameter routing natively within hardware-level constraints.


---

### 2. 📐 `geometry.py` — Static Virtual View Curvature Sensor
다차원 배치 입력의 유동적인 확장 속에서도 정적 컴파일 무결성을 목표로 하는 고속 다양체 곡률(공간 분산) 계측 모듈입니다.

*   **[KR] 목적 및 기능**: JAX 환경에서 `jnp.reshape`나 차원 튜플 조립 시 유저들의 골치를 썩이던 형상 추적 에러(`ConcretizationTypeError`)를 `static_argnums=(1,)` 정적 상수 락(Lock) 가드레일로 막아봤습니다. 임의의 N차원 배치 입력(`[B, D]`, `[B, T, D]` 등)이 인입되더라도 단일 정적 가상 뷰 레벨의 인라인 연속 메모리 레이아웃으로 변환하여 하드웨어 온칩 리덕션 속도를 극대화함이 목적입니다.
*   **[EN] Purpose & Function**: A high-efficiency spatial macro-curvature (geometric variance proxy) measurement module engineered to maintain static compilation integrity across fluid multi-dimensional batch scaling. To shield developers from stubborn tracer crashes (`ConcretizationTypeError`) frequently triggered by dynamic reshaping, we implemented a rigid `static_argnums=(1,)` parameter lock guardrail. It seamlessly aligns arbitrary inputs (e.g., `[B, D]`, `[B, T, D]`) into an inline contiguous memory layout, successfully maximizing hardware on-chip reduction velocities at a single **Static Virtual View** level.

 - **💡 TL;DR (핵심 요약)**:
 - **[KR] 문제와 해법**: JAX에서 입력 데이터의 형상(Shape)을 바꾸거나 쪼갤 때, 차원이 조금만 유동적으로 변해도 컴파일러 추적이 깨지며 런타임 크래시(`ConcretizationTypeError`)가 발생합니다. 이 모듈은 입력 데이터가 2D(`[B, D]`)이든 시계열 3D(`[B, T, D]`)이든 상관없이, 하드웨어 친화적인 가상 2D 연속 매트릭스로 평탄화하여 고속 온칩 리덕션을 마친 뒤 원래 차원으로 복원해 내는 에러 제로형 다양체 제어 엔진입니다.
 - **[EN] Problem & Solution**: Modifying or slicing tensor shapes in fluid JAX environments frequently breaks the compiler's tracking state, throwing stubborn tracing crashes (`ConcretizationTypeError`). Regardless of whether the input is a simple 2D (`[B, D]`) or a sequential 3D (`[B, T, D]`) tensor, this engine seamlessly flattens data into a hardware-friendly virtual 2D matrix to unlock maximum on-chip reduction velocities before safely reconstructing its original shape.


---

### 3. 🛡️ `math_guardrails.py` — Transcendental Function Integrity Guard
초월함수 연산의 수치적 파괴(`NaN`)를 방어하고 경계면에서의 역전파 미분 그레디언트를 보존하는 수치 안정성 커널 모듈입니다.

*   **[KR] 목적 및 기능**: `config` 딕셔너리 주입에 의한 추적 오염을 원천 제거하기 위해 제어 상수들을 독립 인자로 분리하고 컴파일 타임 하드웨어 명령어 상수로 고정했습니다. `jnp.sign` 부호 비트 마스킹과 `jnp.where` 병렬 마스킹 기믹을 융합하여 `if/else` 분기 없는 단일 융합 커널(Fused Kernel)을 형성, `arccos` 등 초월함수의 수치 파괴를 차단함이 목적입니다.
*   **[EN] Purpose & Function**: A numerical robustness kernel meticulously designed to shield transcendental operations (e.g., `arccos`) from floating-point collapse (`NaN`) and preserve crucial backpropagation gradients at numerical boundaries. It isolates configuration constants into distinct independent function arguments to clear tracking contamination, freezing them entirely as compile-time hardware literals. By seamlessly fusing `jnp.sign` bit-masking and parallel `jnp.where` masking predicates, it constructs a highly streamlined **Fused Kernel** operating without conditional branches, thoroughly blocking mathematical breakdowns.

  - **💡 TL;DR (핵심 요약)**:
  - **[KR] 문제와 해법**: 비유클리드 기하학 연산에서 `arccos(x)` 등을 쓸 때, 데이터가 경계값($\pm 1.0$)에 도달하면 분모가 0이 되어 대형 모델 학습이 `NaN`으로 터져버립니다. 그렇다고 값을 무작정 자르면 미분 값이 0이 되어 역전파 학습이 영구 정지됩니다. 이 모듈은 값을 안전 경계 내에 가두면서도, 경계면 바깥에 미세 기울기(`Leaky Slope`)를 부여하여 오차 복원 그레디언트를 끝까지 살려내는 수리 안정성을 목적으로 하는 커널입니다.
  - **[EN] Problem & Solution**: When executing non-Euclidean geometric operations like `arccos(x)`, data approaching strict mathematical limits ($\pm 1.0$) drives denominators to zero, causing large-scale model training to collapse via `NaN` explosions. Conversely, standard hard-clipping permanently suffocates backpropagation by wiping out gradients to zero. This kernel is engineered for numerical stability, securely locking values within safe limits while injecting a delicate `Leaky Slope` beyond critical boundaries to keep error-restoration gradients alive until the very end.


---

## ⚙️ Advanced XLA Compiler & Infrastructure Optimizations

This repository demonstrates how to shrink complex topo-geometric physics simulations into high-density, compiler-fused primitives.

1. **Branchless Gating Routing**: Obliterated Python interpreter branch stalls (`if/else`, `hasattr`) via duck-typing attribute masking and dynamic inline boolean-integer string scaling (`"gate" * is_gate + "backbone" * not_is_gate`).
2. **Einstein Notation Fluidity**: Replaced bulky multidimensional tensor tracking loops with a single `jnp.einsum("...d,dh->...h")` statement, achieving zero-overhead arbitrary batch support.
3. **Hardware-Native Tree Reduction**: Prevented XLA graph fragmentation by leveraging `jax.tree_util.tree_reduce` to compute global gradient norms directly inside accelerator SRAM as a single fused kernel.

---

## ⚖️ License (라이선스 사양)

This repository is officially open-sourced under the **Apache License 2.0**.

*   **[KR] 라이선스 안내**: 본 `egregore-core-jax` 저장소에 공개된 모든 독립 모듈 및 코어 가속 칩 소스코드는 **Apache License 2.0** 조건에 따라 자유롭게 배포, 수정 및 상용화가 가능합니다. 
*   **[EN] License Notice**: All independent modules and core acceleration chip source codes disclosed in this `egregore-core-jax` repository can be freely distributed, modified, and commercialized under the terms and conditions of the **Apache License 2.0**.

---

### 🧩 Source and Re-licensing Notice (소스 출처 및 재라이선싱 고지)

*   **[KR]** 본 저장소의 일부 핵심 모듈(예: 필터링 및 토폴로지 가제트)은 원작자(Original Author)가 직접 개발한 기존 GPLv3 라이선스 기반의 `Egregore` 프로젝트에서 일부 기능을 추출 및 모듈화하여, 본 저장소(`egregore-core-jax`)를 통해 **Apache License 2.0으로 공식 재라이선싱(Re-licensing)하여 배포하는 소스코드**입니다. 원작자의 권한으로 라이선스를 전환하여 재배포하는 것이므로 사용 및 상용화에 법적 제약이 없습니다.
*   **[EN]** Certain core modules (e.g., filtering and topology gadgets) within this repository have been isolated and modularized from the original GPLv3-licensed `Egregore` project by the original author. These modules are **officially re-licensed and distributed under the Apache License 2.0** within this new repository (`egregore-core-jax`). As this transition is executed under the sole authority of the original author, there are no legal restrictions on its commercial use or modification.

---
> ⚠️ **[KR] 면책 조항**: 본 저장소의 모든 소스코드는 "있는 그대로(AS IS)" 제공되며, 명시적 또는 묵시적인 어떠한 보증도 제공하지 않습니다.
> 
> ⚠️ **[EN] Disclaimer**: All code within this repository is provided "AS IS", without warranty of any kind, express or implied.

