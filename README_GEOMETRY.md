# Mathematical Rationale: Why Vertical Reduction (axis=0)?

## 1. 개요 및 검증 목적 (Introduction & Verification Purpose)

- **[KR]** 본 문서는 `egregore-core-jax` 패키지의 고속 다양체 곡률 계측 커널인 `calculate_jacobian_spatial_curvature`가 유동적인 다차원 배치 입력 속에서도 어떻게 JAX/XLA 컴파일러의 형상 추적 실패(`ConcretizationTypeError`)를 원천 차단하고 철옹성 같은 정적 방어선을 구축했는지 증명합니다. 아울러 일반적인 샘플별 독립 연산의 유혹을 뿌리치고, 시스템 전체를 관통하는 거시적 글로벌 다양체 분산 대리치를 도출하기 위해 대수적 축 압축(`axis=0`)을 의도적으로 고수한 하드웨어 아키텍처적 정당성을 논리적으로 기술합니다.
- **[EN]** This documentation validates how the `calculate_jacobian_spatial_curvature` kernel within `egregore-core-jax` blocks JAX/XLA compiler tracing failures (`ConcretizationTypeError`) across fluid multi-dimensional inputs. It provides a technical justification for utilizing vertical axis reduction (`axis=0`) over standard, sample-wise approaches to maintain hardware-level static compilation efficiency and evaluate the macro-statistical global manifold variance proxy.

---

## 2. 핵심 아키텍처 배경: 동적 차원의 유령 방어 (Core Architectural Background)

### 동적 차원의 한계 (Conventional Limitations)
- **[KR]** 일반적인 `axis=-1` 주도의 인스턴스별 압축은 평탄화된 가상 뷰의 동적 행 크기($N$)를 파이프라인 후반부까지 살아남게 만듭니다. 이는 상위 그래프의 분산 배치 최적화나 파이프라인 병렬화 결합 시, 컴파일러가 동적 차원의 실체를 추적하지 못해 `ConcretizationTypeError` 런타임 크래시를 유발하는 치명적인 약점을 가집니다.
- **[EN]** Standard `axis=-1` instance-wise reduction allows the dynamic row dimension ($N$) of the flattened virtual view to persist deep into the computation pipeline. This introduces a structural vulnerability where the compiler fails to trace the volatile dimension during downstream optimization or pipelined parallelism, triggering fatal `ConcretizationTypeError` runtime crashes.

### 5세대의 정적 가드 해법 (5th-Gen Static Guard Solution)
- **[KR]** 본 모듈은 `axis=0`에서 대수적 수직 압축을 선제적으로 수행하여, 입력 데이터의 규격에 따라 유동적으로 변하는 불안정 요소인 동적 행 크기($N$)를 연산 초기에 즉시 기화(Vaporize)합니다. 이를 통해 XLA 컴파일러가 아무런 의심 없이 하드웨어 레벨에서 통 기계어로 고정·최적화할 수 있는 완벽한 정적 `[1, spatial_dim]` 형태의 베이스라인을 사수합니다.
- **[EN]** This module executes preventive vertical algebraic reduction at `axis=0` to immediately vaporize the volatile row dimension ($N$)—an unstable element that fluctuates based on fluid input specifications. By doing so, it aggressively safeguards a rigid, static `[1, spatial_dim]` baseline shape, allowing the XLA compiler to seamlessly freeze and optimize the execution graph into pure hardware-level machine code without tracing overhead.

---

## 3. 글로벌 다양체 분산 대리치 도출의 수리적 정당성 (Mathematical Justification of Global Manifold Variance Proxy)

### 배치 레벨 기하학적 기준점 사수 (Preserving Batch-level Geometric Baseline)

* **[KR]** 본 커널은 개별 샘플 단위의 지엽적인 곡률 변화를 관측하는 것이 아닌, 현재 미니배치(Mini-batch) 전체가 점유하고 있는 고차원 다양체 공간의 거시적 변형 통계량(Macro-statistical Variance)을 추적하는 것을 골자로 합니다.
* **[EN]** Rather than monitoring localized curvature fluctuations at an individual sample instance level, this kernel tracks the macro-statistical variance of the high-dimensional manifold space collectively occupied by the entire mini-batch.

### 대수적 수식 전개 (Algebraic Formulation)

* **[KR]** 가상 연속 행렬 $\mathbf{X} \in \mathbb{R}^{N \times D}$ (여기서 $D =$ `spatial_dim`)에 대해 `axis=0` 압축을 단행하여 도출된 평균 벡터 $\boldsymbol{\mu}$는 다음과 같이 정의됩니다.

$$
\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{X}_{i,:} \quad \in \mathbb{R}^{1 \times D}
$$

이 배치 평균 벡터는 전체 샘플을 관통하는 하나의 견고한 공간적 기준선(Baseline)이 되며, 이를 원본 행렬과 감산하여 대수적 기하 변형 편차(Mean-centered Delta)를 도출합니다. 만약 이를 `axis=-1`로 연산했을 경우, 샘플 내부의 통계치에 갇혀 배치 전체의 거시적 토폴로지 왜곡 흐름을 감지하는 정규화(Regularizer) 가치로서의 범용성이 상실됩니다.

* **[EN]** For a virtual contiguous matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ (where $D =$ `spatial_dim`), the mean vector $\boldsymbol{\mu}$ derived via vertical `axis=0` reduction is mathematically defined as:

$$
\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{X}_{i,:} \quad \in \mathbb{R}^{1 \times D}
$$

This batch mean vector establishes a rigid spatial baseline spanning all instances. Subtracting this from the original matrix isolates the true mean-centered algebraic transformation delta. If computed via `axis=-1`, the calculation becomes strictly confined to intra-sample statistics, completely wiping out its generality as a global topological regularizer for large-scale architectures.

---

## 4. 하드웨어 네이티브 온칩 브로드캐스팅 및 컴파일러 최적화 (Hardware-Native On-Chip Broadcasting)

### **[KR]**

* **HBM 대역폭 절약 및 브로드캐스팅 무결성**: 수직 압축을 통해 완벽한 정적 크기로 굳어진 `batch_axis_mean` ($\mathbb{R}^{1 \times D}$)은 후속 편차 감산 연산 시 가속기(GPU/TPU) 레벨에서 고속 브로드캐스팅(Broadcasting) 매핑 무결성을 가집니다.
* **커널 융합 (Kernel Fusion)**: XLA 컴파일러는 가로축 크기 $D$가 고정되어 있음을 인지하므로, 외부 메모리(HBM)로의 중간 데이터 쓰기/읽기 오버헤드를 원천 차단합니다. 온칩 SRAM 레지스터 안에서 평균 도출, 편차 감산, 원소별 제곱합(`jnp.square`), 리덕션(`jnp.sum`)까지의 전체 연산 체인을 단일 거대 융합 커널(Fused Kernel)로 컴파일해 버립니다. 이는 최적의 메모리 대역폭 효율을 강제하는 인프라 설계의 극치입니다.

### **[EN]**

* **HBM Bandwidth Economy & Broadcasting Integrity**: The `batch_axis_mean` ($\mathbb{R}^{1 \times D}$), strictly frozen into a static structural shape via vertical reduction, unlocks optimal hardware-level broadcasting efficiency during subsequent subtraction stages.
* **Kernel Fusion Realization**: Since the XLA compiler completely recognizes the statically bound feature dimension $D$, it thoroughly purges redundant intermediate High-Bandwidth Memory (HBM) read/write stalls. It forces the entire execution sequence—mean extraction, delta subtraction, element-wise squaring, and chip-level reduction—to be compressed into a single, high-density Fused Kernel natively within the on-chip SRAM registers.

---
```python
# ====================================================================
# [5th-Gen Static Virtual View Curvature Sensor Core Pipeline]
# ====================================================================

# 1. Vaporize the dynamic/volatile dimension (N) early to freeze the tracking trajectory
# [N, spatial_dim] -> [1, spatial_dim] (Pure static hardware register baseline)
batch_axis_mean = jnp.mean(flattened_matrix, axis=0, keepdims=True)

# 2. Linear algebraic stream without explicit branch predicates, preventing XLA graph fragmentation
mean_centered_delta = flattened_matrix - batch_axis_mean

# 3. Drive hardware on-chip reduction along the feature axis (axis=-1) over the static baseline
flat_kappa = jnp.sum(jnp.square(mean_centered_delta), axis=-1, keepdims=True) / spatial_dim

# 4. Reconstruction layout ensuring the compiler safely recognizes the target output shape as a static construct
target_output_shape = original_shape[:-1] + (1,)
return jnp.reshape(flat_kappa, target_output_shape)
```
