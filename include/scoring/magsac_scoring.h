// Copyright (C) 2024 ETH Zurich.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of ETH Zurich nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (majti89@gmail.com)
#pragma once

#include <Eigen/Core>

#include <boost/math/special_functions/gamma.hpp>
#include <fstream>

#include "../estimators/abstract_estimator.h"
#include "../models/model.h"
#include "../utils/macros.h"
#include "../utils/types.h"
#include "abstract_scoring.h"
#include "magsac_look_up_table.h"
#include "score.h"

namespace superansac::scoring {

class MAGSACScoring : public AbstractScoring {
 protected:
  static constexpr bool kUseLookUpTable = true;
  static constexpr bool kGenerateLookUpTable = false;

  // 分布参数（决定数学形式）
  size_t degreesOfFreedom{};  // 残差分布的自由度
  size_t dofIndex_{};         // Cached DOF index for lookup table
  double k{};                 // 该分布的 0.99 分位数
  double Cn{};                // 归一化常数
  // 阈值的各种预乘形式（纯性能优化）
  double squaredSigmaMax{};             // σ_max²
  double squaredSigmaMaxPerTwo{};       // σ_max² / 2
  double squaredSigmaMaxPerFour{};      // σ_max² / 4
  double twoTimesSquaredSigmaMax{};     // 2σ_max²
  double invTwoTimesSquaredSigmaMax{};  // Precomputed inverse for optimization
  // 指数与幂的预计算
  double nPlus1Per2{};   // (n + 1) / 2
  double nMinus1Per2{};  // (n - 1) / 2
  double twoNPlus1{};    // 2 ^ ((n + 1) / 2)
  // 损失函数常数项
  double premultiplier{};        // 1 / σ_max * Cn * 2 ^ ((n + 1) / 2)
  double weightPremultiplier{};  // 1 / σ_max * Cn * (n - 1) / 2
  double value0{};               // Γ_upper((n - 1) / 2, k² / 2)，减除项
  double zeroResidualLoss{};     // 残差为零时的损失（最优值）
  double lossOutlier{};          // 外点的固定损失（最差值）

  // Calculate the upper incomplete gamma function
  static double upperIncompleteGamma(const double a, const double x) {
    return boost::math::tgamma(a) - boost::math::tgamma_lower(a, x);
  }

  // Cached pointer to the interleaved (lower, upper) row of the active DOF;
  // one cache line per lookup. Values are identical to the constexpr tables.
  const double* gammaTable_ = nullptr;

  // 传入的是归一化残差
  [[nodiscard]] FORCE_INLINE std::pair<double, double> getGammaValues(
      const double residual_) const {
    // 表在 x ∈ [0, 1) 上均匀采样 10000 点，所以乘 lookupTableSize 再截断取整就得到最近的下界格点
    auto index = static_cast<size_t>(residual_ * lookupTableSize);
    index = index < lookupTableSize ? index : lookupTableSize - 1;
    const double* kEntry = gammaTable_ + 2 * index;
    return {kEntry[0], kEntry[1]};
  }

  [[nodiscard]] FORCE_INLINE double getUpperGammaValue(const double residual_) const {
    auto index = static_cast<size_t>(residual_ * lookupTableSize);
    index = index < lookupTableSize ? index : lookupTableSize - 1;
    return gammaTable_[2 * index + 1];
  }

  [[nodiscard]] FORCE_INLINE double getLowerGammaValue(const double residual_) const {
    auto index = static_cast<size_t>(residual_ * lookupTableSize);
    index = index < lookupTableSize ? index : lookupTableSize - 1;
    return gammaTable_[2 * index];
  }

 public:
  // Constructor
  MAGSACScoring() = default;

  // Destructor
  ~MAGSACScoring() override = default;

  FORCE_INLINE void updateSPRTParameters(const Score& currentBest, int iterationIndex,
                                         size_t totalPoints) override {}

  // 自由度对应残差的维数，上游 MAGSAC 论文覆盖的模型： - 2： 点到点的 2D 重投影误差（单应、基础矩阵的 Sampson 距离
  // 本质矩阵、绝对位姿、刚体变换），- 3： 3D 点到点，- 4： 两视图 4 维联合残差，- 5、6： 更高维的联合残差
  static constexpr double getOutlierLoss(const size_t kDegreesOfFreedom_) {
    switch (kDegreesOfFreedom_) {
      case 2:
        return 0.215658;  // 0.220642416155;
      case 3:
        return 0.306123;
      case 4:
        return 0.488088;
      case 5:
        return 0.921592;
      case 6:
        return 2.03833;
      default:
        throw std::runtime_error("The degrees of freedom is not supported.");
    }
  }

  // 卡分布的 0.99 分位数
  static constexpr double getK(const size_t kDegreesOfFreedom_) {
    switch (kDegreesOfFreedom_) {
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
        throw std::runtime_error("The degrees of freedom is not supported.");
    }
  }

  // Γ_upper((n - 1) / 2, k² / 2)，与上面 getK() 的 k 取自同一套值，MAGSAC++ 里 pho(r) 里的减除项
  static constexpr double getSubtractTerm(const size_t kDegreesOfFreedom_) {
    switch (kDegreesOfFreedom_) {
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
        throw std::runtime_error("The degrees of freedom is not supported.");
    }
  }

  // Initialize the gamma lookup table
  void initialize(const estimator::Estimator* kEstimator_) {
    initialize(kEstimator_->getDegreesOfFreedom());
  }

  // Initialize the gamma lookup table
  void initialize(const size_t kDegreesOfFreedom_) {
    if (threshold == 0.0)
      throw std::runtime_error("The threshold is not set for the MAGSAC scoring object.");

    degreesOfFreedom = kDegreesOfFreedom_;  // Degrees of freedom
    // 减 2 是因为我们只支持 2 到 6 的自由度，查找表从 0 开始索引
    dofIndex_ = degreesOfFreedom - 2;  // Cache DOF index for lookup table optimization
    gammaTable_ = interleavedGammaLookupTable(dofIndex_);  // Interleaved (lower, upper) row
    k = getK(degreesOfFreedom);  //kEstimator_->getK(); // The 0.99 quantile of the distribution
    // 卡方分布的归一化常数
    Cn = 1.0 / (std::pow(2, static_cast<double>(degreesOfFreedom) / 2.0) *
                boost::math::tgamma(static_cast<double>(degreesOfFreedom) / 2.0));
    squaredSigmaMax = threshold * threshold;               // The squared threshold
    squaredSigmaMaxPerTwo = squaredSigmaMax / 2.0;         // The squared threshold divided by two
    squaredSigmaMaxPerFour = squaredSigmaMaxPerTwo / 2.0;  // The squared threshold divided by four
    twoTimesSquaredSigmaMax = 2.0 * squaredSigmaMax;       // Two times the squared threshold
    invTwoTimesSquaredSigmaMax =
        1.0 / twoTimesSquaredSigmaMax;  // Precomputed inverse for optimization
    nPlus1Per2 = (static_cast<double>(degreesOfFreedom) + 1.0) / 2.0;  // (n + 1) / 2
    nMinus1Per2 = (static_cast<double>(degreesOfFreedom) - 1) / 2.0;   // (n - 1) / 2
    twoNPlus1 = std::pow(2.0, nPlus1Per2);                             // 2 ^ ((n + 1) / 2)
    premultiplier = 1.0 / threshold * Cn * twoNPlus1;                  // The premultiplier
    // The value of the upper incomplete gamma function at k * k / 2
    // value0 = upperIncompleteGamma(nMinus1Per2, k * k / 2.0);
    value0 = getSubtractTerm(degreesOfFreedom);
    weightPremultiplier = 1.0 / threshold * Cn * nMinus1Per2;  // The weight premultiplier
    // lossOutlier = threshold * Cn * nMinus1Per2 * boost::math::tgamma_lower(nPlus1Per2, k * k / 2.0);
    lossOutlier = threshold * getOutlierLoss(degreesOfFreedom);  // The loss of an outlier

    // The same expression at x = 0: the largest per-point gain, used as the early-exit bound.
    zeroResidualLoss = marginalisedInlierLoss(0.0);
  }

  // Set the threshold
  FORCE_INLINE void setThreshold(const double kThreshold_) override {
    threshold = kThreshold_;                   // Set the threshold
    squaredThreshold = threshold * threshold;  // Set the squared threshold
  }

  // The marginalised loss for an inlier, before premultiplier is applied.
  // kX_ is the normalised residual r^2 / (2 sigma_max^2). Shared by getLoss() and by
  // both scoreImpl() paths so the formula exists in exactly one place; previously it
  // was spelled out four times in this file and any change had to be mirrored by hand.
  [[nodiscard]] FORCE_INLINE double marginalisedInlierLoss(const double kX_) const {
    if constexpr (kUseLookUpTable) {
      const std::pair<double, double> kGamma = getGammaValues(kX_);
      return squaredSigmaMaxPerTwo * kGamma.first +
             squaredSigmaMaxPerFour * (kGamma.second - value0);
    }
    return squaredSigmaMaxPerTwo * boost::math::tgamma_lower(nPlus1Per2, kX_) +
           squaredSigmaMaxPerFour * (upperIncompleteGamma(nMinus1Per2, kX_) - value0);
  }

  // Loss function
  [[nodiscard]] FORCE_INLINE double getLoss(const double kSquaredResidual_) const {
    double loss = 0;
    // If the residual is smaller than the threshold, store it as an inlier and
    // increase the score.
    if (kSquaredResidual_ < squaredThreshold) {
      // Increase the score (use precomputed inverse for optimization)
      loss = marginalisedInlierLoss(kSquaredResidual_ * invTwoTimesSquaredSigmaMax);

      loss = premultiplier * loss;  // Increase the loss value
    } else
      loss = lossOutlier;
    return loss;
  }

  // Loss function
  [[nodiscard]] FORCE_INLINE double getWeight(const double kSquaredResidual_) const {
    // If the residual is smaller than the threshold, store it as an inlier and
    // increase the score.
    if (kSquaredResidual_ < squaredThreshold) {
      double residualPerTwoTimesSquaredSigmaMax = kSquaredResidual_ * invTwoTimesSquaredSigmaMax;
      double upperIncompleteGammaValue = getUpperGammaValue(residualPerTwoTimesSquaredSigmaMax);
      return weightPremultiplier * (upperIncompleteGammaValue - value0);
    }

    return 0.0;
  }

 protected:
  // Sample function
  FORCE_INLINE Score
  scoreImpl(const DataMatrix& kData_, const models::Model& kModel_,
            const estimator::Estimator* kEstimator_, std::vector<size_t>& inliers_,
            const bool kStoreInliers_, const Score& kBestScore_,
            std::vector<const std::vector<size_t>*>* kPotentialInlierSets_) const override {
    // Create a static empty Score
    static const Score kEmptyScore;
    // The number of points
    const int kPointNumber = kData_.rows();
    // The squared residual
    double squaredResidual;
    // Score and inlier number
    size_t inlierNumber = 0;
    double scoreValue = 0.0;
    // The score of the previous best model
    const double kBestInlierNumber = kBestScore_.getInlierNumber();
    double loss;

    if (kPotentialInlierSets_ != nullptr) {
      const double kBestScoreValue = kBestScore_.getValue();
      const double kBestPossibleGain = premultiplier * zeroResidualLoss;

      // Process potential inlier sets
      size_t testedPoints = 0;
      for (const auto& potentialInlierSet : *kPotentialInlierSets_) {
        // Increase the number of tested points
        testedPoints += potentialInlierSet->size();

        for (const auto& pointIdx : *potentialInlierSet) {
          // Calculate the point-to-model residual
          squaredResidual = kEstimator_->squaredResidual(kData_.row(pointIdx).data(), kModel_);

          // If the residual is smaller than the threshold, store it as an inlier and
          // increase the score.
          if (squaredResidual < squaredThreshold) {
            if (kStoreInliers_)  // Store the point as an inlier if needed.
              inliers_.emplace_back(pointIdx);

            // Increase the inlier number
            ++inlierNumber;

            loss = marginalisedInlierLoss(squaredResidual * invTwoTimesSquaredSigmaMax);

            // Commenting "premultiplier" as it does not affect the final result. It is just a constant.
            scoreValue += premultiplier * loss;  // Increase the loss value
          } else
            scoreValue += lossOutlier;

          if (kBestPossibleGain * (kPointNumber - pointIdx) + scoreValue < kBestScoreValue)
            return kEmptyScore;
        }
      }

      // Increase the score by the loss of the untested outliers
      scoreValue += (kData_.rows() - testedPoints) * lossOutlier;
    } else {
      // Pre-compute values for early exit optimization
      const double kBestPossibleGain = premultiplier * zeroResidualLoss;
      const double kBestScoreValue = kBestScore_.getValue();

      // Iterate through all points in blocks: residuals for each block are
      // computed with a single (batched) virtual call, then consumed by the
      // unchanged sequential logic. Identical decisions and arithmetic; on
      // early exit at most the rest of the current block was computed in vain.
      // Blocks grow geometrically so that models rejected early only
      // overshoot by a small block.
      constexpr int kMaxBlockSize = 256;
      double sqrBuffer[kMaxBlockSize];
      int blockSize = 16;
      for (int base = 0; base < kPointNumber;
           base += blockSize, blockSize = std::min(blockSize << 1, kMaxBlockSize)) {
        const int kCount = std::min(blockSize, kPointNumber - base);
        kEstimator_->squaredResiduals(kData_, kModel_, base, kCount, sqrBuffer);
        for (int j = 0; j < kCount; ++j) {
          const int pointIdx = base + j;
          squaredResidual = sqrBuffer[j];

          // If the residual is smaller than the threshold, store it as an inlier and
          // increase the score.
          if (squaredResidual < squaredThreshold) {
            if (kStoreInliers_)  // Store the point as an inlier if needed.
              inliers_.emplace_back(pointIdx);

            // Increase the inlier number
            ++inlierNumber;

            loss = marginalisedInlierLoss(squaredResidual * invTwoTimesSquaredSigmaMax);

            // Commenting "premultiplier" as it does not affect the final result. It is just a constant.
            scoreValue += premultiplier * loss;  // Increase the loss value
          } else
            scoreValue += lossOutlier;

          // Early exit AFTER processing: if remaining perfect inliers can't beat best score
          if (kBestPossibleGain * (kPointNumber - pointIdx - 1) + scoreValue < kBestScoreValue)
            return kEmptyScore;
        }
      }
    }

    return {inlierNumber, scoreValue};
  }

  // Get weights for the points
  FORCE_INLINE void getWeightsImpl(
      const DataMatrix& kData_,                             // Data matrix
      const models::Model& kModel_,                         // The model to be scored
      const estimator::Estimator* kEstimator_,              // Estimator
      std::vector<double>& weights_,                        // The weights of the points
      const std::vector<size_t>* kIndices_) const override  // The indices of the points
  {
    double residualPerTwoTimesSquaredSigmaMax, upperIncompleteGamma;

    if (kIndices_ == nullptr) {
      // The number of points
      const int kPointNumber = kData_.rows();
      // Allocate memory for the weights
      weights_.resize(kPointNumber);

      // Compute all squared residuals with one batched call directly into
      // the weights buffer, then transform them in place.
      kEstimator_->squaredResiduals(kData_, kModel_, 0, kPointNumber, weights_.data());
      for (int pointIdx = 0; pointIdx < kPointNumber; ++pointIdx) {
        // The squared residual
        double squaredResidual = weights_[pointIdx];

        // If the residual is smaller than the threshold, store it as an inlier and
        // increase the score.
        if (squaredResidual < squaredThreshold) {
          residualPerTwoTimesSquaredSigmaMax = squaredResidual * invTwoTimesSquaredSigmaMax;
          upperIncompleteGamma = getUpperGammaValue(residualPerTwoTimesSquaredSigmaMax);

          weights_[pointIdx] = weightPremultiplier * (upperIncompleteGamma - value0);
          // Commenting "weightPremultiplier" as it does not affect the final result. It is just a constant.
          //weights_[pointIdx] = upperIncompleteGamma - value0;
        } else
          weights_[pointIdx] = 0.0;
      }
    } else {
      // The number of points
      const int kPointNumber = kIndices_->size();
      // Allocate memory for the weights
      weights_.resize(kPointNumber);

      // Iterate through all points, calculate the squaredResiduals and store the points as inliers if needed.
      for (int pointIdx = 0; pointIdx < kPointNumber; ++pointIdx) {
        // Calculate the point-to-model residual
        double squaredResidual =
            kEstimator_->squaredResidual(kData_.row((*kIndices_)[pointIdx]).data(), kModel_);

        // If the residual is smaller than the threshold, store it as an inlier and
        // increase the score.
        if (squaredResidual < squaredThreshold) {
          residualPerTwoTimesSquaredSigmaMax = squaredResidual * invTwoTimesSquaredSigmaMax;
          upperIncompleteGamma = getUpperGammaValue(residualPerTwoTimesSquaredSigmaMax);
          weights_[pointIdx] = weightPremultiplier * (upperIncompleteGamma - value0);
          // Commenting "weightPremultiplier" as it does not affect the final result. It is just a constant.
          //weights_[pointIdx] = upperIncompleteGamma - value0;
        } else
          weights_[pointIdx] = 0.0;
      }
    }
  }
};

}  // namespace superansac::scoring
