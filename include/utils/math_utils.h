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
//     * Neither the name of Czech Technical University nor the
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
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include <Eigen/Eigen>

namespace superansac::utils {
// Pivoting In-Place Gauss Elimination to solve problem A * x = b,
// where A is the known coefficient matrix, b is the inhomogeneous part and x is the unknown vector.
// Form: matrix_ = [A, b].
template <size_t Size>
void gaussElimination(Eigen::Matrix<double, Size, Size + 1>&
                          matrix_,  // The matrix to which the elimination is applied
                      Eigen::Matrix<double, Size, 1>& result_)  // The solution of equation
{
  // Pivotisation
  for (size_t i = 0; i < Size; ++i) {
    // Find the row with the largest pivot element
    size_t maxRow = i;
    for (size_t k = i + 1; k < Size; ++k) {
      if (std::abs(matrix_(k, i)) > std::abs(matrix_(maxRow, i))) {
        maxRow = k;
      }
    }

    // Swap the current row with the row with the largest pivot
    // 部分选主元（只换行，不换列），目的是数值稳定： 用最大的数作除数，避免小数做分母把误差放大
    if (maxRow != i) {
      matrix_.row(i).swap(matrix_.row(maxRow));
    }

    // Elimination process
    for (size_t k = i + 1; k < Size; ++k) {
      double factor = matrix_(k, i) / matrix_(i, i);
      matrix_.row(k).segment(i, Size + 1 - i) -= factor * matrix_.row(i).segment(i, Size + 1 - i);
    }
  }

  // Back-substitution
  for (int i = Size - 1; i >= 0; --i) {
    result_(i) = matrix_(i, Size);
    for (int j = i + 1; j < Size; ++j) {
      result_(i) -= matrix_(i, j) * result_(j);
    }
    result_(i) /= matrix_(i, i);
  }
}

}  // namespace superansac::utils
