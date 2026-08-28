// Copyright (C) 2024 ETH Zurich.
// All rights reserved.
//
// MAGSAC + SPRT scoring. API-compatible with MAGSACScoring but performs
// SPRT-based preemptive verification while keeping MAGSAC loss.
//
// Please contact the author if you have any questions.
// Author of original MAGSAC: Daniel Barath (majti89@gmail.com)
// SPRT adaptation: <your name/email>
#pragma once

#include <Eigen/Core>

#include <boost/math/special_functions/gamma.hpp>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "../estimators/abstract_estimator.h"
#include "../models/model.h"
#include "../utils/macros.h"
#include "../utils/types.h"
#include "abstract_scoring.h"
#include "magsac_look_up_table.h"
#include "score.h"

namespace superansac::scoring {

class MAGSACSPRTScoring : public AbstractScoring {
 protected:
  // ====== MAGSAC core (with optimizations) ======
  static constexpr bool kUseLookUpTable = true;
  // Guards 1/cost when a model fits every point exactly (cost 0).
  static constexpr double kCostEpsilon = 1e-12;

  size_t degreesOfFreedom = 0;
  size_t dofIndex_ = 0;  // Cached DOF index for lookup table
  double k = 0.0;
  double Cn = 0.0;
  double squaredSigmaMax = 0.0;
  double squaredSigmaMaxPerTwo = 0.0;
  double squaredSigmaMaxPerFour = 0.0;
  double twoTimesSquaredSigmaMax = 0.0;
  double invTwoTimesSquaredSigmaMax = 0.0;  // Precomputed inverse for optimization
  double nPlus1Per2 = 0.0;
  double nMinus1Per2 = 0.0;
  double twoNPlus1 = 0.0;
  double lossOutlier = 0.0;
  double premultiplier = 0.0;
  double value0 = 0.0;
  double squaredTruncatedThreshold = 0.0;
  double weightPremultiplier = 0.0;

  // Cached pointer to the interleaved (lower, upper) row of the active DOF;
  // one cache line per lookup. Values are identical to the constexpr tables.
  const double* gammaTable_ = nullptr;

  [[nodiscard]] FORCE_INLINE std::pair<double, double> getGammaValues(
      const double residual_) const {
    auto idx = static_cast<size_t>(residual_ * lookupTableSize);
    if (idx >= lookupTableSize) idx = lookupTableSize - 1;
    const double* kEntry = gammaTable_ + 2 * idx;
    return {kEntry[0], kEntry[1]};
  }

  [[nodiscard]] FORCE_INLINE double getUpperGammaValue(const double residual_) const {
    auto idx = static_cast<size_t>(residual_ * lookupTableSize);
    if (idx >= lookupTableSize) idx = lookupTableSize - 1;
    return gammaTable_[2 * idx + 1];
  }

  static constexpr double getOutlierLoss(const size_t& dof) {
    switch (dof) {
      case 2:
        return 0.609974735;
      case 3:
        return 0.779573319;
      case 4:
        return 0.920321825;
      case 5:
        return 1.04299872;
      case 6:
        return 1.15306832;
      default:
        throw std::runtime_error("Unsupported degrees of freedom.");
    }
  }

  static constexpr double getK(const size_t& dof) {
    switch (dof) {
      case 2:
        return 3.03485426;
      case 3:
        return 3.36821418;
      case 4:
        return 3.64372119;
      case 5:
        return 3.88410511;
      case 6:
        return 4.10023095;
      default:
        throw std::runtime_error("Unsupported degrees of freedom.");
    }
  }

  static constexpr double getSubtractTerm(const size_t& dof) {
    switch (dof) {
      case 2:
        return 0.00426544;
      case 3:
        return 0.00343949;
      case 4:
        return 0.00361126;
      case 5:
        return 0.00452559;
      case 6:
        return 0.00647484;
      default:
        throw std::runtime_error("Unsupported degrees of freedom.");
    }
  }

  // ====== SPRT state ====== 在打分过程中提前否决明显不好的模型
  struct SPRTHistory {
    double epsilon = 0.05;  // P(inlier | good)
    double delta = 0.005;   // P(inlier | bad)
    double A = 1.0;         // LR threshold
  };

  // Tunable (balanced for performance and accuracy)
  static constexpr bool kUseRuntimeA = false;  // set true to use K-based threshold
  static constexpr double kDefaultAlpha =
      0.02;  // Balanced false positive rate (was 0.05, tried 0.01)
  static constexpr double kDefaultBeta =
      0.02;  // Balanced false negative rate (was 0.05, tried 0.01)
  static constexpr double kMinEpsilon = 1e-3;
  static constexpr double kMinDelta = 1e-4;
  static constexpr double kMaxDeltaFrac = 0.5;
  static constexpr double kUpdateTolFrac = 0.05;

  // Runtime model (used only if kUseRuntimeA)
  double tM_ms = 0.05;
  double mS = 1.0;

  SPRTHistory sprt_{.epsilon = 0.05, .delta = 0.005, .A = 0.0};
  size_t lastUpdateIteration_ = 0;

  // Rejection statistics for delta update
  size_t rejectedCount_ = 0;
  double rejectedInlierFracSum_ = 0.0;

  // Reset SPRT state (called from initialize to ensure clean state)
  void resetSPRT() {
    rejectedCount_ = 0;
    rejectedInlierFracSum_ = 0.0;
    lastUpdateIteration_ = 0;
  }

  static double clampProb(const double x, const double lo, const double hi) {
    return std::max(lo, std::min(hi, x));
  }

  static double waldA(const double alpha = kDefaultAlpha, const double beta = kDefaultBeta) {
    return (1.0 - beta) / alpha;
  }

  static double informationC(const double eps, const double del) {
    return (1.0 - del) * std::log((1.0 - del) / (1.0 - eps)) + del * std::log(del / eps);
  }
  [[nodiscard]] double estimateThresholdA_runtime(const double eps, const double del) const {
    const double C = informationC(eps, del);
    if (C <= 0.0) return waldA();
    const double K = tM_ms * C / std::max(1.0, mS) + 1.0;
    double A_prev = K, A = K;
    for (int i = 0; i < 10; ++i) {
      A = K + std::log(std::max(1e-12, A_prev));
      if (std::abs(A - A_prev) < 1.5e-8) break;
      A_prev = A;
    }
    return std::max(A, 1.0);
  }

  template <typename EstimatorT>
  void microBenchmarkResiduals(const DataMatrix& X, const EstimatorT* E, size_t trials = 256) {
    if constexpr (!kUseRuntimeA) return;
    const auto n = static_cast<size_t>(X.rows());
    if (n == 0) return;
    trials = std::min(trials, n);
    auto t0 = std::chrono::high_resolution_clock::now();
    volatile double sink = 0.0;
    for (size_t i = 0; i < trials; ++i) {
      sink += E->squaredResidual(X.row(static_cast<int>(i)).data(), models::Model());
    }
    (void)sink;
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    tM_ms = std::max(0.01, ms / static_cast<double>(trials) * 4.0);
    mS = 1.0;
  }

  void refreshA() {
    if (kUseRuntimeA) sprt_.A = estimateThresholdA_runtime(sprt_.epsilon, sprt_.delta);
    sprt_.A = waldA();
  }

  // MAGSAC loss and weight
  [[nodiscard]] FORCE_INLINE double magsacLoss(const double& kSquaredResidual_) const {
    if (kSquaredResidual_ < squaredThreshold) {
      const double r = kSquaredResidual_ * invTwoTimesSquaredSigmaMax;
      double loss = 0.0;
      if constexpr (kUseLookUpTable) {
        const auto g = getGammaValues(r);
        loss = squaredSigmaMaxPerTwo * g.first + kSquaredResidual_ * 0.25 * (g.second - value0);
      }
      return premultiplier * loss;
    }
    return lossOutlier;
  }
  [[nodiscard]] FORCE_INLINE double magsacWeight(const double kSquaredResidual_) const {
    if (kSquaredResidual_ >= squaredThreshold) return 0.0;
    const double r = kSquaredResidual_ * invTwoTimesSquaredSigmaMax;
    return weightPremultiplier * (getUpperGammaValue(r) - value0);
  }

 public:
  MAGSACSPRTScoring() { refreshA(); }

  ~MAGSACSPRTScoring() override = default;

  // ====== Initialization (no 'override' here) ======
  void initialize(const estimator::Estimator* kEstimator_) {
    if (threshold == 0.0) throw std::runtime_error("Threshold must be set before initialize().");
    initialize(kEstimator_->getDegreesOfFreedom());
  }

  // Match MAGSACScoring’s overload
  void initialize(const size_t kDegreesOfFreedom_) {
    if (kDegreesOfFreedom_ < 2 || kDegreesOfFreedom_ > 6)
      throw std::invalid_argument(
          "MAGSAC scoring supports 2 to 6 degrees of freedom; the gamma look-up "
          "tables hold no other rows.");

    degreesOfFreedom = kDegreesOfFreedom_;
    dofIndex_ = degreesOfFreedom - 2;  // Cache DOF index for lookup table optimization
    gammaTable_ = interleavedGammaLookupTable(dofIndex_);  // Interleaved (lower, upper) row
    k = getK(degreesOfFreedom);
    Cn = 1.0 / (std::pow(2.0, static_cast<double>(degreesOfFreedom) / 2.0) *
                boost::math::tgamma(static_cast<double>(degreesOfFreedom) / 2.0));
    squaredSigmaMax = threshold * threshold;
    squaredSigmaMaxPerTwo = squaredSigmaMax / 2.0;
    squaredSigmaMaxPerFour = squaredSigmaMaxPerTwo / 2.0;
    twoTimesSquaredSigmaMax = 2.0 * squaredSigmaMax;
    invTwoTimesSquaredSigmaMax = 1.0 / twoTimesSquaredSigmaMax;  // Precomputed inverse
    nPlus1Per2 = static_cast<double>(degreesOfFreedom + 1) / 2.0;
    nMinus1Per2 = static_cast<double>(degreesOfFreedom - 1) / 2.0;
    twoNPlus1 = std::pow(2.0, nPlus1Per2);
    premultiplier = 1.0 / threshold * Cn * twoNPlus1;
    value0 = getSubtractTerm(degreesOfFreedom);
    squaredTruncatedThreshold = k * k * squaredSigmaMax;
    weightPremultiplier = 1.0 / threshold * Cn * std::pow(2.0, nMinus1Per2);
    lossOutlier = threshold * getOutlierLoss(degreesOfFreedom);

    // Reset SPRT state to ensure clean state between runs
    resetSPRT();
  }

  // ====== AbstractScoring API ======
  FORCE_INLINE void setThreshold(const double kThreshold_) override {
    threshold = kThreshold_;
    squaredThreshold = threshold * threshold;
  }

 protected:
  FORCE_INLINE Score
  scoreImpl(const DataMatrix& kData_, const models::Model& kModel_,
            const estimator::Estimator* kEstimator_, std::vector<size_t>& inliers_,
            const bool kStoreInliers_, const Score& kBestScore_,
            std::vector<const std::vector<size_t>*>* kPotentialInlierSets_) const override {
    static const Score kEmptyScore;

    const int N = static_cast<int>(kData_.rows());
    if (N == 0) return {};

    // Note: Removed permutation - sequential order is sufficient for SPRT
    if constexpr (kUseRuntimeA) {
      if (tM_ms <= 0.0)
        const_cast<MAGSACSPRTScoring*>(this)->microBenchmarkResiduals(kData_, kEstimator_, 128);
    }

    const double eps = clampProb(sprt_.epsilon, kMinEpsilon, 1.0 - 1e-6);
    const double del = clampProb(sprt_.delta, kMinDelta, kMaxDeltaFrac);
    const double A = std::max(1.0, sprt_.A);

    double lambdaLR = 1.0;
    size_t inlierCount = 0;
    double scoreVal = 0.0;

    auto accumulate_point = [&](const int iPos, const size_t trueIdx, const double sqr) -> bool {
      if (sqr < squaredThreshold) {
        lambdaLR *= del / eps;
        if (kStoreInliers_) inliers_.push_back(trueIdx);
        ++inlierCount;
        scoreVal += magsacLoss(sqr);
      } else {
        lambdaLR *= (1.0 - del) / (1.0 - eps);
        scoreVal += lossOutlier;
      }

      if (lambdaLR > A) {
        // record partial stats to refine delta
        const double obsFrac = static_cast<double>(inlierCount) / static_cast<double>(iPos + 1);
        const_cast<MAGSACSPRTScoring*>(this)->rejectedInlierFracSum_ += obsFrac;
        const_cast<MAGSACSPRTScoring*>(this)->rejectedCount_ += 1;
        return false;
      }

      // rho >= 0 and rho(0) = 0, so scoreVal is already a lower bound on the
      // final cost; once it passes the incumbent's cost the model cannot win.
      const double kBestQuality = kBestScore_.getValue();
      if (kBestQuality > 0.0 && scoreVal > 1.0 / kBestQuality) return false;

      return true;
    };

    if (kPotentialInlierSets_ == nullptr) {
      constexpr int kMaxBlockSize = 256;
      double sqrBuffer[kMaxBlockSize];
      int blockSize = 16;
      for (int base = 0; base < N;) {
        const int kCount = std::min(blockSize, N - base);
        kEstimator_->squaredResiduals(kData_, kModel_, base, kCount, sqrBuffer);
        for (int j = 0; j < kCount; ++j) {
          if (!accumulate_point(base + j, base + j, sqrBuffer[j])) return kEmptyScore;
        }
        base += kCount;
        blockSize = std::min(blockSize << 1, kMaxBlockSize);
      }
    } else {
      int tested = 0;
      for (const auto* setPtr : *kPotentialInlierSets_) {
        for (size_t trueIdx : *setPtr) {
          if (!accumulate_point(tested, trueIdx,
                                kEstimator_->squaredResidual(
                                    kData_.row(static_cast<int>(trueIdx)).data(), kModel_)))
            return kEmptyScore;
          ++tested;
        }
      }
      const int remaining = N - tested;
      scoreVal += remaining * lossOutlier;
    }

    return {inlierCount, 1.0 / (scoreVal + kCostEpsilon)};
  }

  FORCE_INLINE void getWeightsImpl(const DataMatrix& kData_, const models::Model& kModel_,
                                   const estimator::Estimator* kEstimator_,
                                   std::vector<double>& weights_,
                                   const std::vector<size_t>* kIndices_) const override {
    if (kIndices_ == nullptr) {
      const int N = static_cast<int>(kData_.rows());
      weights_.resize(N);
      // One batched call for all residuals, then transform in place.
      kEstimator_->squaredResiduals(kData_, kModel_, 0, N, weights_.data());
      for (int i = 0; i < N; ++i) weights_[i] = magsacWeight(weights_[i]);
    } else {
      const int N = static_cast<int>(kIndices_->size());
      weights_.resize(N);
      for (int i = 0; i < N; ++i) {
        const double sqr = kEstimator_->squaredResidual(
            kData_.row(static_cast<int>((*kIndices_)[i])).data(), kModel_);
        weights_[i] = magsacWeight(sqr);
      }
    }
  }

 public:
  void updateSPRTParameters(const Score& currentBest, const int iterationIndex,
                            const size_t totalPoints) override {
    if (currentBest.getInlierNumber() > 0 && currentBest.getValue() > 0.0) {
      const double newEps = clampProb(static_cast<double>(currentBest.getInlierNumber()) /
                                          static_cast<double>(std::max<size_t>(1, totalPoints)),
                                      kMinEpsilon, 0.999);

      if (std::abs(newEps - sprt_.epsilon) / std::max(sprt_.epsilon, 1e-6) > kUpdateTolFrac) {
        sprt_.epsilon = newEps;
        if (rejectedCount_ > 0) {
          const double avgBadInlier = rejectedInlierFracSum_ / static_cast<double>(rejectedCount_);
          sprt_.delta = clampProb(avgBadInlier, kMinDelta, kMaxDeltaFrac);
        } else {
          sprt_.delta = std::min(sprt_.delta, sprt_.epsilon / 10.0);
        }
        refreshA();
        if (iterationIndex >= 0)
          lastUpdateIteration_ = iterationIndex;
        else
          ++lastUpdateIteration_;
        rejectedCount_ = 0;
        rejectedInlierFracSum_ = 0.0;
      }
    } else {
      if (rejectedCount_ >= 5) {
        const double avgBadInlier = rejectedInlierFracSum_ / static_cast<double>(rejectedCount_);
        const double newDelta = clampProb(avgBadInlier, kMinDelta, kMaxDeltaFrac);
        if (std::abs(newDelta - sprt_.delta) / std::max(sprt_.delta, 1e-6) > kUpdateTolFrac) {
          sprt_.delta = newDelta;
          refreshA();
        }
        rejectedCount_ = 0;
        rejectedInlierFracSum_ = 0.0;
      }
    }
  }
};

}  // namespace superansac::scoring
