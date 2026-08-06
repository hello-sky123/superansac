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

#include "inlier_selectors/types.h"
#include "local_optimization/types.h"
#include "neighborhood/types.h"
#include "samplers/types.h"
#include "scoring/types.h"
#include "termination/types.h"

namespace superansac {
struct ARSamplerSettings {
  double estimatorVariance = 0.9765;
  double randomness = 0.01;
};

struct LocalOptimizationSettings {
  size_t maxIterations = 20,  // Reduced from 50 (diminishing returns after 20)
      graphCutNumber = 20;
  double sampleSizeMultiplier = 3.0;  // Reduced from 7.0 (3x sufficient for robustness)
  double spatialCoherenceWeight = 0.1;
};

struct NeighborhoodSettings {
  double neighborhoodSize = 20.0, neighborhoodGridDensity = 4.0;
  size_t nearestNeighborNumber = 6;
};

// DEGENSAC (Chum et al. 2005) dominant-plane degeneracy handling for
// fundamental-matrix estimation. When enabled, every so-far-best
// minimal-sample model is tested for H-degeneracy and re-estimated by the
// plane-and-parallax algorithm if degenerate.
struct DegensacSettings {
  bool enabled = false;  // Off by default
  double homographyThreshold =
      2.0;  // H-consistency threshold (pixels; scaled with the data normalization)
  size_t innerRansacIterations = 50;  // Plane-and-parallax draws per degenerate event
};

struct RANSACSettings {
  // 迭代与收敛
  size_t topKForLocalOptimization =
      3;  // Number of best models used for local optimization (reduced from 5)
  size_t minIterations = 1000;   // Minimum number of iterations
  size_t maxIterations = 5000;   // Maximum number of iterations
  double confidence = 0.99;      // Confidence
  double inlierThreshold = 1.5;  // Inlier threshold

  // 七个策略选择器：决定各工厂实例化哪个实现
  scoring::ScoringType scoring = scoring::ScoringType::MAGSAC;    // Scoring type
  samplers::SamplerType sampler = samplers::SamplerType::PROSAC;  // Sampler type
  neighborhood::NeighborhoodType neighborhood =
      neighborhood::NeighborhoodType::Grid;  // Neighborhood type
  local_optimization::LocalOptimizationType localOptimization =
      local_optimization::LocalOptimizationType::NestedRANSAC;  // Local optimization type
  local_optimization::LocalOptimizationType finalOptimization =
      local_optimization::LocalOptimizationType::IRLS;  // Final optimization type
  termination::TerminationType terminationCriterion =
      termination::TerminationType::RANSAC;  // Termination criterion type
  inlier_selector::InlierSelectorType inlierSelector =
      inlier_selector::InlierSelectorType::None;  // Inlier selector type

  // 模型专属开关
  bool fundamentalOrientationCheck =
      false;  // Reject minimal F models violating the oriented epipolar constraint（定向对极约束）
  bool fundamentalSymmetricValidation =
      false;  // Validate new-best F models with the symmetric epipolar distance（对称对极距离二次校验）
  size_t essentialCheiralityPointNumber =
      1;  // Cheirality-vote points for E pose disambiguation (raise to ~25 for dense/RoMA matches;
  // splg gains nothing as cv2.recoverPose redoes cheirality)（位姿消歧的 cheirality 投票点数）
  bool homographyBundleRefinement =
      true;  // Nonlinear (LM) reprojection-error refinement of the non-minimal homography fit (H only)
  size_t homographyBundleMaxIterations =
      5;  // LM iterations: early stopping regularizes (full convergence over-fits the plane) and stays within the runtime budget

  // 最终精修调度
  size_t finalRefinementRounds =
      0;  // Extra score-gated re-collect-and-refit rounds in the final LSQ optimization
  bool finalRefinementSchedule =
      false;  // Use a shrinking-threshold schedule in the final refinement rounds
  double finalRefinementScheduleStart =
      3.0;  // Widest sigma multiplier in the final-refinement schedule
  double finalRefinementScheduleEnd =
      0.5;  // Narrowest sigma multiplier in the final-refinement schedule
  bool finalRefinementWeighted =
      false;  // Feed per-point confidences into the final-refinement bundle adjustment

  // 主循环行为开关：影响流程本身，与模型类型无关
  bool localOptimizationInsideTheLoop =
      false;  // Locally optimize as soon as a new best model is found, not only at the end
  bool useSprt = true;           // Use the SPRT test to abandon hopeless models early
  bool finalPolishTopK = false;  // Run the final optimizer on every top-K candidate, then pick

  // 五个嵌套子结构
  ARSamplerSettings arSamplerSettings;
  LocalOptimizationSettings localOptimizationSettings, finalOptimizationSettings;
  NeighborhoodSettings neighborhoodSettings;
  DegensacSettings degensacSettings;
};

}  // namespace superansac