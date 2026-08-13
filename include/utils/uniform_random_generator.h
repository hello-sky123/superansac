#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "macros.h"

namespace superansac::utils {

// -------- split mix64 (seed expander) --------
// 最初由 Guy L. Steele Jr. 等人在 2014 年为 Java 8 设计，代码极短、执行极快且统计特性良好，现在被广泛应用于
// 各种 C++ 项目中，尤其是作为更复杂随机数生成器（如 xoshiro256**）的种子初始化器
struct SplitMix64 {
  uint64_t x;
  explicit SplitMix64(const uint64_t seed) : x(seed) {}

  uint64_t next() {
    uint64_t z = x += 0x9E3779B97f4A7C15ul;  // 约等于 2^64 / φ，其中 φ = (1 + √5) / 2 是黄金分割比
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;  // 比特混合
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    return z ^ (z >> 31);
  }
};

// -------- xoshiro256** (public-domain) --------
// 目前世界上最优秀的通用伪随机数生成器（PRNG）之一，被 Lua、Julia 等现代编程语言的默认随机数引擎
struct Xoshiro256ss {
  uint64_t s[4] = {};  // 256 位状态空间，足够大以避免周期性问题

  // 左旋函数：将 x 左移 k 位，并将溢出的高位循环到低位
  static uint64_t rotl(const uint64_t x, const int k) { return (x << k) | (x >> (64 - k)); }

  // 构造函数：使用 SplitMix64 初始化状态（不能是全零状态，否则会一直输出零）
  explicit Xoshiro256ss(const uint64_t seed = 1) {
    SplitMix64 sm(seed);
    for (unsigned long& i : s) i = sm.next();
  }

  uint64_t operator()() {
    // 提取当前状态的一部分生成随机数
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    // 更新状态以准备下一次调用
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
  }

  static constexpr uint64_t min() { return 0; }

  static constexpr uint64_t max() { return ~uint64_t{0}; }
};

// -------- Lemire mapping to [0, n) (closed -> adjust) --------
// 在一个指定区间 [lo, hi] 内，极速且无偏差（Unbiased）地生成均匀分布的随机整数，相比于传统的取模运算
// 不仅消除了“模偏差”，还干掉了昂贵的除法运算，尤其适合在高性能场景下使用
inline uint64_t uniform_u64_closed(Xoshiro256ss& rng, const uint64_t lo, const uint64_t hi) {
  const uint64_t n = hi - lo + 1;
  auto m = static_cast<__uint128_t>(rng()) * static_cast<__uint128_t>(n);
  auto l = static_cast<uint64_t>(m);  // 取乘积的低 64 位（小数部分）
  // 几乎不会进入这里
  if (l < n) {
    const uint64_t t = -n % n;
    while (l < t) {
      m = static_cast<__uint128_t>(rng()) * static_cast<__uint128_t>(n);
      l = static_cast<uint64_t>(m);
    }
  }
  return static_cast<uint64_t>(m >> 64) + lo;
}

template <typename T>
T uniform_closed(Xoshiro256ss& rng, T lo, T hi) {
  static_assert(std::is_integral_v<T>, "integral type required");
  return static_cast<T>(
      uniform_u64_closed(rng, static_cast<uint64_t>(lo), static_cast<uint64_t>(hi)));
}

// -------- Adapter to satisfy UniformRandomBitGenerator for <random> dists --------
// 适配器：将 Xoshiro256ss 封装为符合 C++ 标准库 <random> 的 UniformRandomBitGenerator 接口
struct XoshiroAdapter {
  using result_type = uint64_t;  // 定义结果类型为 64 位无符号整数

  Xoshiro256ss* p{};  // 存放指针，避免随机数状态意外拷贝

  explicit XoshiroAdapter(Xoshiro256ss& r) : p(&r) {}

  result_type operator()() const { return (*p)(); }

  static constexpr result_type min() { return 0; }

  static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }
};

// -------- Drop-in class with original API --------
// 该类提供了一个简单的接口，用于生成均匀分布的随机数，并支持从指定范围内生成唯一的随机集合
template <typename IndexType>
class UniformRandomGenerator {
 public:
  using value_type = IndexType;

  UniformRandomGenerator() = default;

  // Legacy API: returns a generator usable by std::discrete_distribution
  XoshiroAdapter& getGenerator() { return adapter_; }

  // Legacy API: set range (stored only; no heavy distribution object)
  FORCE_INLINE void resetGenerator(const IndexType& min_v, const IndexType& max_v) {
    min_ = min_v;
    max_ = max_v;
  }

  // Legacy API: draw one
  FORCE_INLINE IndexType getRandomNumber() {
    return uniform_closed<IndexType>(engine_, min_, max_);
  }

  // Unique draws without replacement (sparse -> Floyd, dense -> partial Fisher–Yates)
  FORCE_INLINE void generateUniqueRandomSet(IndexType* sample, const IndexType& k) {
    generateUniqueRandomSet(sample, k, max_);
  }

  FORCE_INLINE void generateUniqueRandomSet(IndexType* sample, const IndexType& k,
                                            const IndexType& max_v) {
    resetGenerator(0, max_v);
    choose_without_replacement(sample, k, static_cast<uint64_t>(max_v) + 1);
  }

  FORCE_INLINE void generateUniqueRandomSet(IndexType* sample, const IndexType k,
                                            const IndexType max_v, const IndexType toSkip) {
    resetGenerator(0, max_v);
    // draw k+1 then drop toSkip if present
    std::vector<IndexType> tmp(static_cast<size_t>(k) + 1);
    choose_without_replacement(tmp.data(), static_cast<uint64_t>(k) + 1,
                               static_cast<uint64_t>(max_v) + 1);
    size_t w = 0;
    for (auto v : tmp)
      if (v != toSkip && w < static_cast<size_t>(k)) sample[w++] = v;
    while (w < static_cast<size_t>(k)) {
      IndexType x;
      do {
        x = uniform_closed<IndexType>(engine_, 0, max_v);
      } while (x == toSkip || std::find(sample, sample + w, x) != sample + w);
      sample[w++] = x;
    }
  }

 private:
  // Choose k values from [0, N) without replacement into sample (unordered)
  // 无放回随机抽样，当 k * 8 <= N 时使用 Floyd 算法（稀疏情况），否则使用部分 Fisher–Yates 洗牌算法（稠密情况）
  FORCE_INLINE void choose_without_replacement(IndexType* sample, const uint64_t k,
                                               const uint64_t N) {
    if (k * 8ul <= N) {  // Floyd for sparse case
      // One reusable set per k, so this per-iteration call does not allocate.
      // The output order is the unordered_set's iteration order, which for a
      // given insertion sequence depends only on the bucket count; clear()
      // preserves the buckets, and caching per k keeps the bucket count equal
      // to that of a fresh set with reserve(k * 3) -- so the produced samples
      // are identical to the previous fresh-set version.
      // std::unordered_set 频繁的内存分配（new）和哈希重排（rehash）是性能杀手，使用 thread_local 缓存
      // 可以显著提升性能，维护 k --> S_k 的映射，避免重复分配和重排
      thread_local std::unordered_map<uint64_t, std::unordered_set<uint64_t>> setsByK;
      auto [it, isNew] = setsByK.try_emplace(k);  // 找到对应 k 的 Set
      std::unordered_set<uint64_t>& S = it->second;
      if (isNew)
        // Reserve more space to avoid rehashing: k * 3 ensures load factor stays below 0.66
        S.reserve(static_cast<size_t>(k) * 3);
      else
        S.clear();
      for (uint64_t j = N - k; j < N; ++j) {
        uint64_t t = uniform_u64_closed(engine_, 0, j);
        if (!S.insert(t).second) S.insert(j);
      }
      size_t i = 0;
      for (auto v : S) sample[i++] = static_cast<IndexType>(v);
    } else {  // partial Fisher–Yates for dense case
      // Reusable scratch; fully rewritten below.
      thread_local std::vector<uint64_t> a;
      a.resize(N);
      for (uint64_t i = 0; i < N; ++i) a[i] = i;
      for (uint64_t i = 0; i < k; ++i) {
        // j ∈ [i, N - 1] uniformly at random
        const uint64_t j = i + uniform_u64_closed(engine_, 0, N - 1 - i);
        std::swap(a[i], a[j]);
      }
      for (uint64_t i = 0; i < k; ++i) sample[i] = static_cast<IndexType>(a[i]);
    }
  }

  // State
  Xoshiro256ss engine_{0x1234567890ABCDEFul};
  XoshiroAdapter adapter_{engine_};
  IndexType min_ = 0;
  IndexType max_ = std::numeric_limits<IndexType>::max() - 1;
};

}  // namespace superansac::utils
