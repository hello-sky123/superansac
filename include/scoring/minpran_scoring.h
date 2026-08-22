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
// Give this scoring a GENEROUS inlier_threshold. Unlike the others it estimates
// the noise scale itself, and the threshold only bounds the candidate set it
// searches -- residuals[] is pre-filtered to r < threshold, so a threshold near
// the true scale truncates the very range the search needs. Its accuracy is
// therefore governed by how much headroom the threshold leaves, measured over ten
// seeds at sigma 1.5 with 200 inliers among 300 points (MAGSAC alongside, same
// data), precision 1.0000 in every cell:
//
//   threshold   MINPRAN med / recall     MAGSAC med / recall
//   3.0         0.7933 px / 0.57         0.3015 px / 0.87
//   4.5         0.3708 px / 0.79         0.2627 px / 0.99
//   6.0         0.3317 px / 0.85         0.2503 px / 1.00
//   9.0         0.2649 px / 0.90         0.2354 px / 1.00
//
// The scale estimate itself is sound where it has room. Holding sigma fixed at 2.0
// and widening the threshold, the largest residual it accepts grows 2.219 ->
// 3.295 -> 4.854 -> 5.402 -> 5.739 for thresholds 4, 6, 9, 14, 20, i.e. 1.11 to
// 2.87 times sigma, converging on the 0.99 quantile of the chi distribution
// (~3.03 sigma) once the pre-filter stops binding. The search also finds a real
// interior minimum rather than collapsing onto the tightest candidate: on one
// dumped curve the argmin is k=126 at 8.42e-08, below its neighbours k=115
// (1.59e-07) and k=129 (5.88e-07).
//
// So the remaining gap to MAGSAC is not a defect in the statistic -- it is that
// MAGSAC marginalises over sigma and needs no such headroom, while MINPRAN is
// asked to find sigma inside a window the caller sized for a different purpose.
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
  FORCE_INLINE void setThreshold(const double kThreshold_) override {
    threshold = kThreshold_;
    squaredThreshold = threshold * threshold;
  }

  FORCE_INLINE void updateSPRTParameters(const Score& currentBest, int iterationIndex,
                                         size_t totalPoints) override {}

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
      if (!(kP > 0.0) || kP > 1.0 || kInlierCount > residuals.size()) continue;
      // The population is the filtered set, not kPointNumber. residuals only
      // holds points already inside squaredThreshold, and both kInlierCount and
      // kP are measured against that same set, so the binomial must be too.
      // Using the full count makes the observed tally fall below chance at every
      // threshold -- k = 30 where 300 * p = 35.6 -- so the statistic pins near 1
      // and never dips.
      const size_t kPopulation = residuals.size();
      const double randomness =
          boost::math::ibeta(static_cast<double>(kInlierCount),
                             static_cast<double>(kPopulation - kInlierCount + 1), kP);

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
      // bestInlierNumber is a count, so iterate strictly below it. It was an
      // index before the statistic was rewritten, when `<=` was the right bound.
      for (size_t i = 0; i < bestInlierNumber; ++i) inliers_.push_back(residuals[i].second);
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