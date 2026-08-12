// Copyright (c) 2020, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount __popcnt
#endif

namespace superansac::sturm {

// Constructs the quotients needed for evaluating the sturm sequence.
// 斯图姆定理，用于确定多项式在给定区间内实数根的个数，f_vec 输入数组，包含了两个多项式的系数，f(x), f'(x)
template <int N>
void build_sturm_seq(const double* f_vec, double* s_vec) {
  double f[3 * N];
  double* f1 = f;           // 初始为原多项式系数升幂排列
  double* f2 = f1 + N + 1;  // 初始为导数
  double* f3 = f2 + N;      // 指向将要计算出的下一多项式

  std::copy(f_vec, f_vec + (2 * N + 1), f);

  // 原本 q1 = f1[top] / f2[top]，当强制 f2 的最高项系数为 ±1 时，可简化 q1 q0
  for (int i = 0; i < N - 1; ++i) {
    const double q1 = f1[N - i] * f2[N - 1 - i];
    const double q0 = f1[N - 1 - i] * f2[N - 1 - i] - f1[N - i] * f2[N - 2 - i];

    // 计算未取负号的余式
    f3[0] = f1[0] - q0 * f2[0];
    for (int j = 1; j < N - 1 - i; ++j) {
      f3[j] = f1[j] - q1 * f2[j - 1] - q0 * f2[j];
    }
    const double c = -std::abs(f3[N - 2 - i]);
    const double ci = 1.0 / c;
    // 将最高项的系数化为 ±1
    for (int j = 0; j < N - 1 - i; ++j) {
      f3[j] = f3[j] * ci;
    }

    // juggle pointers (f1,f2,f3) -> (f2,f3,f1)
    double* tmp = f1;
    f1 = f2;
    f2 = f3;
    f3 = tmp;

    s_vec[3 * i] = q0;
    s_vec[3 * i + 1] = q1;
    s_vec[3 * i + 2] = c;
  }

  s_vec[3 * N - 3] = f1[0];
  s_vec[3 * N - 2] = f1[1];
  s_vec[3 * N - 1] = f2[0];
}

// Evaluates polynomial using Horner's method.
// Assumes that f[N] = 1.0
template <int N>
double poly_val(const double* f, const double x) {
  double fx = x + f[N - 1];
  for (int i = N - 2; i >= 0; --i) {
    fx = x * fx + f[i];
  }
  return fx;
}

// Daniel Thul is responsible for this template-trickery :)
// 用一个 32 位无符号整数来表示浮点数组中的元素是否小于零
template <int D>
unsigned int flag_negative(const double* const f) {
  return (f[D] < 0) << D | flag_negative<D - 1>(f);
}

template <>
inline unsigned int flag_negative<0>(const double* const f) {
  return f[0] < 0;
}

// Evaluates the sturm sequence and counts the number of sign changes
template <int N, std::enable_if_t<N<32, void>* = nullptr> int sign_changes(const double* s_vec,
                                                                           const double x) {
  double f[N + 1];
  f[N] = s_vec[3 * N - 1];                             // 序列最后一个多项式（常数项）的值
  f[N - 1] = s_vec[3 * N - 3] + x * s_vec[3 * N - 2];  // 倒数第二个多项式（一次多项式）的值

  for (int i = N - 2; i >= 0; --i) {
    f[i] = (s_vec[3 * i] + x * s_vec[3 * i + 1]) * f[i + 1] + s_vec[3 * i + 2] * f[i + 2];
  }

  // In testing this turned out to be slightly faster compared to a naive loop
  unsigned int S = flag_negative<N>(f);  // 提取符号掩码

  return __builtin_popcount((S ^ S >> 1) & ~(0xFFFFFFFF << N));  // 相邻异或寻找变号
}

template <int N, std::enable_if_t<N >= 32, void>* = nullptr>
int sign_changes(const double* s_vec, const double x) {
  double f[N + 1];
  f[N] = s_vec[3 * N - 1];
  f[N - 1] = s_vec[3 * N - 3] + x * s_vec[3 * N - 2];

  for (int i = N - 2; i >= 0; --i) {
    f[i] = (s_vec[3 * i] + x * s_vec[3 * i + 1]) * f[i + 1] + s_vec[3 * i + 2] * f[i + 2];
  }

  int count = 0;
  bool neg1 = f[0] < 0;  // 提取第 0 个元素的符号
  for (int i = 0; i < N; ++i) {
    const bool neg2 = f[i + 1] < 0;
    if (neg1 ^ neg2) {
      ++count;
    }
    neg1 = neg2;
  }
  return count;
}

// Computes the Cauchy bound on the real roots. 柯西根界定理
// Experiments with more complicated (expensive) bounds did not seem to have a good trade-off.
template <int N>
double get_bounds(const double* f_vec) {
  double max = 0;
  for (int i = 0; i < N; ++i) {
    max = std::max(max, std::abs(f_vec[i]));
  }
  return 1.0 + max;
}

// Applies Ridder's bracketing method until we get close to root, followed by newton iterations
template <int N>
void ridders_method_newton(const double* f_vec, double a, double b, double* roots, int& n_roots,
                           const double tol) {
  double fa = poly_val<N>(f_vec, a);
  double fb = poly_val<N>(f_vec, b);

  if (!(fa < 0 ^ fb < 0)) return;

  constexpr double tol_newton = 1e-3;

  // 里德斯方法是一种基于区间收缩的算法，不会让根跑出缩小后的区间，且具有 1.414 的超线性收敛性
  for (int iter = 0; iter < 30; ++iter) {
    // 里德斯方法并不负责找到最终精度的根，它只负责把区间缩小到一个足够安全的微小范围内
    if (std::abs(a - b) < tol_newton) {
      break;
    }
    const double c = (a + b) * 0.5;
    const double fc = poly_val<N>(f_vec, c);
    const double s = std::sqrt(fc * fc - fa * fb);
    if (!(s > 0)) break;
    // 缩小根所在的区间
    const double d = fa < fb ? c + (a - c) * fc / s : c + (c - a) * fc / s;
    const double fd = poly_val<N>(f_vec, d);

    if (fd >= 0 ? fc < 0 : fc > 0) {
      a = c;
      fa = fc;
      b = d;
      fb = fd;
    } else if (fd >= 0 ? fa < 0 : fa > 0) {
      b = d;
      fb = fd;
    } else {
      a = d;
      fa = fd;
    }
  }

  // We switch to Newton's method once we are close to the root
  double x = (a + b) * 0.5;

  const double* fp_vec = f_vec + N + 1;
  for (int iter = 0; iter < 10; ++iter) {
    const double fx = poly_val<N>(f_vec, x);
    if (std::abs(fx) < tol) {
      break;
    }
    const double fpx = static_cast<double>(N) * poly_val<N - 1>(fp_vec, x);  // 导数求值
    const double dx = fx / fpx;
    x = x - dx;
    if (std::abs(dx) < tol) {
      break;
    }
  }

  roots[n_roots++] = x;
}

template <int N>
void isolate_roots(const double* f_vec, const double* s_vec, const double a, const double b,
                   const int sa, const int sb, double* roots, int& n_roots, const double tol,
                   const int depth) {
  if (depth > 300) return;

  int n_rts =
      sa - sb;  // 计算区间 [a, b] 内的实根个数，sa 和 sb 分别是 Sturm 序列在 a 和 b 处的符号变化数

  // 切分包含多个根的区间，递归裂变
  if (n_rts > 1) {
    double c = (a + b) * 0.5;
    const int sc = sign_changes<N>(s_vec, c);
    isolate_roots<N>(f_vec, s_vec, a, c, sa, sc, roots, n_roots, tol, depth + 1);
    isolate_roots<N>(f_vec, s_vec, c, b, sc, sb, roots, n_roots, tol, depth + 1);
  } else if (n_rts == 1) {
    ridders_method_newton<N>(f_vec, a, b, roots, n_roots, tol);
  }
}

// 多项式求根的总入口
template <int N>
int bisect_sturm(const double* coeffs, double* roots, const double tol = 1e-10) {
  if (coeffs[N] == 0.0)
    return 0;  // return bisect_sturm<N - 1>(coeffs, roots,tol); // This explodes compile times...

  double f_vec[2 * N + 1];
  double s_vec[3 * N];

  // f_vec is the polynomial and its first derivative.
  std::copy(coeffs, coeffs + N + 1, f_vec);

  // Normalize w.r.t. leading coeff
  double c_inv = 1.0 / f_vec[N];
  for (int i = 0; i < N; ++i) f_vec[i] *= c_inv;
  f_vec[N] = 1.0;

  // Compute the derivative with normalized coefficients
  for (int i = 0; i < N - 1; ++i) {
    f_vec[N + 1 + i] = f_vec[i + 1] * ((i + 1) / static_cast<double>(N));
  }
  f_vec[2 * N] = 1.0;

  // Compute sturm sequences
  build_sturm_seq<N>(f_vec, s_vec);

  // All real roots are in the interval [-r0, r0]
  const double r0 = get_bounds<N>(f_vec);
  double a = -r0;
  double b = r0;

  const int sa = sign_changes<N>(s_vec, a);
  const int sb = sign_changes<N>(s_vec, b);

  int n_roots = sa - sb;
  if (n_roots == 0) return 0;

  n_roots = 0;
  isolate_roots<N>(f_vec, s_vec, a, b, sa, sb, roots, n_roots, tol, 0);

  return n_roots;
}

template <>
inline int bisect_sturm<1>(const double* coeffs, double* roots, double) {
  if (coeffs[1] == 0.0) {
    return 0;
  }
  roots[0] = -coeffs[0] / coeffs[1];
  return 1;
}

template <>
inline int bisect_sturm<0>(const double*, double*, double) {
  return 0;
}

// Computes the characteristic polynomial of a matrix using Danilevsky's method with pivoting
template <typename Derived>
void charpoly_danilevsky_piv(Eigen::MatrixBase<Derived>& A, double* p) {
  const int n = A.rows();

  // 从矩阵底行往上遍历，通过构造一系列的变换矩阵 M，一步步将矩阵 A 转化为 Frobenius 形式，从而得到特征多项式的系数
  for (int i = n - 1; i > 0; --i) {
    int piv_ind = i - 1;
    double piv = std::abs(A(i, i - 1));

    // Find largest pivot
    for (int j = 0; j < i - 1; j++) {
      if (std::abs(A(i, j)) > piv) {
        piv = std::abs(A(i, j));
        piv_ind = j;
      }
    }
    if (piv_ind != i - 1) {
      // Perform permutation（同时交换两行和对应的两列，依然是一个完美的相似变换）
      A.row(i - 1).swap(A.row(piv_ind));
      A.col(i - 1).swap(A.col(piv_ind));
    }
    piv = A(i, i - 1);

    Eigen::VectorXd v = A.row(i);
    A.row(i - 1) = v.transpose() * A;  // 这相当于执行了 M^{-1} * A

    Eigen::VectorXd v_inv = -v;
    v_inv(i - 1) = 1.0;
    v_inv /= piv;
    v_inv(i - 1) -= 1.0;
    Eigen::VectorXd Acol = A.col(i - 1);
    for (int j = 0; j <= i; j++) A.row(j) = A.row(j) + Acol(j) * v_inv.transpose();

    A.row(i).setZero();
    A(i, i - 1) = 1.0;
  }
  p[n] = 1.0;
  for (int i = 0; i < n; i++) p[i] = -A(0, n - i - 1);
}

}  // namespace superansac::sturm
