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

#include <boost/math/special_functions/beta.hpp>

#include "../estimators/abstract_estimator.h"
#include "../models/model.h"
#include "../utils/macros.h"
#include "../utils/types.h"
#include "abstract_scoring.h"
#include "score.h"

namespace superansac {
namespace scoring {

// MINPRAN: pick the threshold whose inlier count is least likely to have arisen
// by chance. `randomness` is that probability -- the regularised incomplete beta
// I_p(k, N - k + 1) for k points inside a region of relative measure p.
//
// This was previously computed as C(N, k-1) times the unnormalised Beta
// integral, evaluated with a 1000-point trapezoid rule. That product is a
// different function and, as measured, monotonic in the threshold, so the argmin
// was always the tightest candidate and the noise scale was never located.
// Calling boost::math::ibeta directly restores a real minimum -- on one model the
// curve now bottoms at k=30 (0.861) instead of k=8 (0.908) -- and removes 25
// quadrature sweeps of 1000 points and two pow() calls each per scored model.
//
// Accuracy on this estimator is still well behind MAGSAC (0.82 px at recall 0.14
// against 0.30 px at 0.87 on synthetic homography data, 200 inliers among 300
// points at sigma 1.5), so the threshold search has further problems; what is
// fixed here is the statistic and its cost, not the estimator as a whole.
class MINPRANScoring : public AbstractScoring {
 protected:
  const size_t kStepNumber;
  std::vector<std::vector<unsigned long long>> binomCoeffs;

 public:
  // Constructor
  MINPRANScoring() : kStepNumber(25) {}

  // Destructor
  ~MINPRANScoring() {}

  // Set the threshold
  FORCE_INLINE void setThreshold(const double kThreshold_) {
    threshold = kThreshold_;
    squaredThreshold = threshold * threshold;
  }

  FORCE_INLINE void updateSPRTParameters(const Score& currentBest, int iterationIndex,
                                         size_t totalPoints) {}

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
    int inlierNumber = 0;
    // The score of the previous best model
    const double kBestScoreValue = kBestScore_.getValue();
    //
    std::vector<std::pair<double, size_t>> residuals;
    residuals.reserve(kPointNumber);

    // Iterate through all points, calculate the squaredResiduals and store the points as inliers if needed.
    for (int pointIdx = 0; pointIdx < kPointNumber; ++pointIdx) {
      // Calculate the point-to-model residual
      squaredResidual = kEstimator_->squaredResidual(kData_.row(pointIdx).data(), kModel_);

      // If the residual is smaller than the threshold, store it as an inlier and
      // increase the score.
      if (squaredResidual < squaredThreshold)
        residuals.emplace_back(std::make_pair(squaredResidual, pointIdx));
    }

    // Sort the residuals
    std::sort(residuals.begin(), residuals.end(),
              [](const std::pair<double, size_t>& a, const std::pair<double, size_t>& b) {
                return a.first < b.first;
              });

    // No point fell inside the threshold, so there is nothing to score.
    // residuals.back() would be undefined behaviour here, and kMaxResidual then
    // reaches the threshold loop as garbage: a zero step makes it run forever,
    // and a negative or NaN one leaves the model unscored.
    if (residuals.empty()) return kEmptyScore;

    // Get the maximum residual
    const double kMaxResidual = residuals.back().first;
    const double kResidualStep = kMaxResidual / kStepNumber;
    // All surviving residuals are identical (a single distinct value), so the
    // step is zero and `currentThreshold += kResidualStep` would never advance.
    if (!(kResidualStep > 0.0)) return kEmptyScore;
    const size_t kSampleSize = kEstimator_->sampleSize();
    double currentThreshold = 0;
    size_t currentMaxIdx = 0;
    double randomness, minRandomness = std::numeric_limits<double>::max(), bestThreshold = 0,
                       bestInlierNumber = 0;

    // Iterate through the thresholds
    for (currentThreshold = kResidualStep; currentThreshold <= kMaxResidual;
         currentThreshold += kResidualStep) {
      // Count the inliers of the current threshold
      while (currentMaxIdx + 1 < residuals.size() &&
             residuals[currentMaxIdx + 1].first < currentThreshold)
        ++currentMaxIdx;

      // If the number of inliers is smaller than the sample size, continue
      if (currentMaxIdx < kSampleSize + 1) continue;

      // Probability that at least kInlierCount of kPointNumber points land inside
      // the current threshold by chance: the regularised incomplete beta
      // I_p(k, N - k + 1). currentMaxIdx is an index, so the count below the
      // threshold is one greater. residuals hold SQUARED distances, so p is
      // formed with squaredThreshold.
      const size_t kInlierCount = currentMaxIdx + 1;
      const double kP = currentThreshold / squaredThreshold;
      if (!(kP > 0.0) || kP > 1.0 || kInlierCount > static_cast<size_t>(kPointNumber)) continue;
      const double randomness =
          boost::math::ibeta(static_cast<double>(kInlierCount),
                             static_cast<double>(kPointNumber - kInlierCount + 1), kP);

      // Check if the randomness is NaN or inf
      if (std::isnan(randomness) || std::isinf(randomness)) continue;

      // Calculate the final result
      if (randomness < minRandomness) {
        minRandomness = randomness;
        bestThreshold = currentThreshold;
        bestInlierNumber = kInlierCount;
      }
    }

    // Store the inliers
    if (kStoreInliers_) {
      inliers_.reserve(bestInlierNumber);
      for (size_t i = 0; i <= bestInlierNumber; ++i) inliers_.push_back(residuals[i].second);
    }

    if (bestInlierNumber < kSampleSize) return kEmptyScore;

    return Score(bestInlierNumber, 1.0 / abs(minRandomness));
  }

  // Get weights for the points
  FORCE_INLINE void getWeightsImpl(
      const DataMatrix& kData_,                             // Data matrix
      const models::Model& kModel_,                         // The model to be scored
      const estimator::Estimator* kEstimator_,              // Estimator
      std::vector<double>& weights_,                        // The weights of the points
      const std::vector<size_t>* kIndices_) const override  // The indices of the points
  {
    if (kIndices_ == nullptr) {
      weights_.resize(kData_.rows());
      for (size_t i = 0; i < kData_.rows(); ++i) weights_[i] = 1.0;
    } else {
      weights_.resize(kIndices_->size());
      for (size_t i = 0; i < kIndices_->size(); ++i) weights_[i] = 1.0;
    }
  }
};

}  // namespace scoring
}  // namespace superansac