// Copyright (c) 2021, Viktor Larsson
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
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include "../../scoring/magsac_scoring.h"

namespace poselib {

// Robust loss functions
class TrivialLoss {
 public:
  explicit TrivialLoss(double) {}  // dummy to ensure we have consistent calling interface

  TrivialLoss() = default;

  static double loss(const double r2) { return r2; }

  static double weight(double) { return 1.0; }
};

class TruncatedLoss {
 public:
  explicit TruncatedLoss(const double threshold) : squared_thr(threshold * threshold) {}

  [[nodiscard]] double loss(const double r2) const { return std::min(r2, squared_thr); }

  [[nodiscard]] double weight(const double r2) const { return r2 < squared_thr ? 1.0 : 0.0; }

 private:
  const double squared_thr;
};

// The method from
//  Le and Zach, Robust Fitting with Truncated Least Squares: A Bilevel Optimization Approach, 3DV 2021
// for truncated least squares optimization with IRLS. 保持了截断的思路，让权重保持连续（原本权重在阈值处不连续）
class TruncatedLossLeZach {
 public:
  explicit TruncatedLossLeZach(const double threshold)
      : squared_thr(threshold * threshold), mu(0.5) {}

  [[nodiscard]] double loss(const double r2) const { return std::min(r2, squared_thr); }

  [[nodiscard]] double weight(const double r2) const {
    const double r2_hat = r2 / squared_thr;
    const double zstar = std::min(r2_hat, 1.0);

    if (r2_hat < 1.0) {
      return 0.5;
    }
    // assumes mu > 0.5
    const double r2m1 = r2_hat - 1.0;
    const double rho = (2.0 * r2m1 + std::sqrt(4.0 * r2m1 * r2m1 * mu * mu + 2 * mu * r2m1)) / mu;
    const double a = (r2_hat + mu * rho * zstar - 0.5 * rho) / (1 + mu * rho);
    const double zbar = std::max(0.0, std::min(a, 1.0));
    return (zstar - zbar) / rho;
  }

 private:
  const double squared_thr;

 public:
  // hyper-parameter for penalty strength
  double mu;
  // schedule for increasing mu in each iteration
  static constexpr double alpha = 1.5;
};

// The method from
//  Le and Zach, Robust Fitting with Truncated Least Squares: A Bilevel Optimization Approach, 3DV 2021
// for truncated least squares optimization with IRLS.
//
// Currently unreachable, and measurement says leave it that way. Every bundle
// solver overrides loss_type to CAUCHY whenever a sample is supplied
// (solver_fundamental_matrix_bundle_adjustment.h and its essential-matrix and
// PnP counterparts), and the non-minimal solvers are always called with one, so
// the LossType::MAGSACPlusPlus that pybind_functions.cpp selects for MAGSAC
// scoring never takes effect.
//
// The sign is at least right now: since the scoring classes moved to the
// published rho(r), getLoss() rises monotonically from 0 and LM minimises it.
// Before that it returned a gain that fell with the residual, so this class
// would have driven the refinement the wrong way had it ever run.
//
// Removing the CAUCHY override to switch it on was tried and is worse.
// Essential-matrix convergence over 120 seeds falls from 54/120 to 38/120 and
// median |E - E_gt| on converged runs from 0.0044 to 0.0073; fundamental-matrix
// Sampson error loosens 0.3397 to 0.3441 px, buying inlier precision 0.9892 ->
// 0.9923. The likely cause is that rho is truncated at sigma_max, not at
// k*sigma_max where the paper makes it continuous -- the gamma table only spans
// x in [0, 1), which a cut at k*sigma_max would overrun -- leaving a 1.9x step
// in the cost at the threshold (0.9657 to 1.8299 at sigma_max = 3) that LM's
// accept test has to climb over.
class MAGSACPlusPlusLoss {
 public:
  superansac::scoring::MAGSACScoring magsac_scoring;

  explicit MAGSACPlusPlusLoss(const double threshold, const size_t degrees_of_freedom = 2) {
    magsac_scoring.setThreshold(threshold);
    magsac_scoring.initialize(degrees_of_freedom);
  }

  [[nodiscard]] double loss(const double r2) const { return magsac_scoring.getLoss(r2); }

  [[nodiscard]] double weight(const double r2) const { return magsac_scoring.getWeight(r2); }
};

class HuberLoss {
 public:
  explicit HuberLoss(const double threshold) : thr(threshold) {}

  [[nodiscard]] double loss(const double r2) const {
    const double r = std::sqrt(r2);
    if (r <= thr) {
      return r2;
    }
    return thr * (2.0 * r - thr);
  }

  [[nodiscard]] double weight(const double r2) const {
    const double r = std::sqrt(r2);
    if (r <= thr) {
      return 1.0;
    }
    return thr / r;
  }

 private:
  const double thr;
};

class CauchyLoss {
 public:
  explicit CauchyLoss(const double threshold) : inv_sq_thr(1.0 / (threshold * threshold)) {}

  [[nodiscard]] double loss(const double r2) const { return std::log1p(r2 * inv_sq_thr); }

  [[nodiscard]] double weight(const double r2) const {
    return std::max(std::numeric_limits<double>::min(), inv_sq_thr / (1.0 + r2 * inv_sq_thr));
  }

 private:
  const double inv_sq_thr;
};

}  // namespace poselib
