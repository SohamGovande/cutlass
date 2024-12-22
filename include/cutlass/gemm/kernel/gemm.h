/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h"

#include "cutlass/gemm/warp/mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_policy.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"
#include "cutlass/arch/arch.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator_params.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"
#include "cutlass/gemm/threadblock/mma_base.h"
#include "cutlass/layout/tensor_op_multiplicand_sm80.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/threadblock/mma_base.h"
#include <type_traits>

// Helper template to force a compile-time error
template <typename T>
struct TypeDebugger;

template <typename T>
static void printType()
{
  // Use static_assert to force the compiler to display the type in an error
  static_assert(sizeof(TypeDebugger<T>) == 0, "Printing type...");
}
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass
{
  namespace gemm
  {
    namespace kernel
    {

      /////////////////////////////////////////////////////////////////////////////////////////////////

      template <
          typename Mma_,                ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_,           ///! Epilogue
          typename ThreadblockSwizzle_, ///! Threadblock swizzling function
          bool SplitKSerial             ///! If true, code supporting split-K via serial reduction is enabled.
          >
      struct Gemm
      {

        using Mma = Mma_;
        using Epilogue = Epilogue_;
        using OutputOp = typename Epilogue::OutputOp;
        using ThreadblockSwizzle = ThreadblockSwizzle_;
        static bool const kSplitKSerial = SplitKSerial;

        /// Warp count (concept: GemmShape)
        using WarpCount = typename Mma::WarpCount;
        static int const kThreadCount = 32 * WarpCount::kCount;

        /// Parameters structure
        struct Params
        {
          cutlass::gemm::GemmCoord problem_size;
          cutlass::gemm::GemmCoord grid_tiled_shape;
          int swizzle_log_tile;
          typename Mma::IteratorA::Params params_A;
          typename Mma::IteratorA::TensorRef ref_A;
          typename Mma::IteratorB::Params params_B;
          typename Mma::IteratorB::TensorRef ref_B;
          typename Epilogue::OutputTileIterator::Params params_C;
          typename Epilogue::OutputTileIterator::TensorRef ref_C;
          typename Epilogue::OutputTileIterator::Params params_D;
          typename Epilogue::OutputTileIterator::TensorRef ref_D;
          typename OutputOp::Params output_op;
          int *semaphore;
          int gemm_k_size;
          // For gather+scatter operations
          int const *gather_A_indices;
          int const *gather_B_indices;
          int const *scatter_D_indices;

          //
          // Methods
          //

          CUTLASS_HOST_DEVICE
          Params() : swizzle_log_tile(0), semaphore(0), gemm_k_size(0) {}

          CUTLASS_HOST_DEVICE
          Params(
              cutlass::gemm::GemmCoord const &problem_size,
              cutlass::gemm::GemmCoord const &grid_tiled_shape,
              typename Mma::IteratorA::TensorRef ref_A,
              typename Mma::IteratorB::TensorRef ref_B,
              typename Epilogue::OutputTileIterator::TensorRef ref_C,
              typename Epilogue::OutputTileIterator::TensorRef ref_D,
              typename OutputOp::Params output_op = typename OutputOp::Params(),
              int *workspace = nullptr,
              int const *gather_A_indices = nullptr,
              int const *gather_B_indices = nullptr,
              int const *scatter_D_indices = nullptr) : problem_size(problem_size),
                                                        grid_tiled_shape(grid_tiled_shape),
                                                        swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
                                                        params_A(ref_A.layout()),
                                                        ref_A(ref_A),
                                                        params_B(ref_B.layout()),
                                                        ref_B(ref_B),
                                                        params_C(ref_C.layout()),
                                                        ref_C(ref_C),
                                                        params_D(ref_D.layout()),
                                                        ref_D(ref_D),
                                                        output_op(output_op),
                                                        gather_A_indices(gather_A_indices),
                                                        gather_B_indices(gather_B_indices),
                                                        scatter_D_indices(scatter_D_indices)
          {

            int total_gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
            int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();

            gemm_k_size = gemm_k_iterations * Mma::Shape::kK;

            semaphore = workspace;
          }
        };

        /// Shared memory storage structure
        union SharedStorage
        {
          typename Mma::SharedStorage main_loop;
          typename Epilogue::SharedStorage epilogue;
        };

        //
        // Methods
        //

        CUTLASS_HOST_DEVICE
        Gemm() {}

        /// Determines whether kernel satisfies alignment
        CUTLASS_HOST_DEVICE
        static Status can_implement(
            cutlass::gemm::GemmCoord const &problem_size,
            typename Mma::IteratorA::TensorRef ref_A,
            typename Mma::IteratorB::TensorRef ref_B,
            typename Epilogue::OutputTileIterator::TensorRef ref_C,
            typename Epilogue::OutputTileIterator::TensorRef ref_D)
        {

          static int const kAlignmentA = (platform::is_same<typename Mma::IteratorA::Layout,
                                                            layout::ColumnMajorInterleaved<32>>::value)
                                             ? 32
                                         : (platform::is_same<typename Mma::IteratorA::Layout,
                                                              layout::ColumnMajorInterleaved<64>>::value)
                                             ? 64
                                             : Mma::IteratorA::AccessType::kElements;
          static int const kAlignmentB = (platform::is_same<typename Mma::IteratorB::Layout,
                                                            layout::RowMajorInterleaved<32>>::value)
                                             ? 32
                                         : (platform::is_same<typename Mma::IteratorB::Layout,
                                                              layout::RowMajorInterleaved<64>>::value)
                                             ? 64
                                             : Mma::IteratorB::AccessType::kElements;
          static int const kAlignmentC = (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                                                            layout::ColumnMajorInterleaved<32>>::value)
                                             ? 32
                                         : (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                                                              layout::ColumnMajorInterleaved<64>>::value)
                                             ? 64
                                             : Epilogue::OutputTileIterator::kElementsPerAccess;

          if (!TensorRef_aligned(ref_A, kAlignmentA))
          {
            return Status::kErrorMisalignedOperand;
          }

          if (!TensorRef_aligned(ref_B, kAlignmentB))
          {
            return Status::kErrorMisalignedOperand;
          }

          if (!TensorRef_aligned(ref_C, kAlignmentC))
          {
            return Status::kErrorMisalignedOperand;
          }

          if (!TensorRef_aligned(ref_D, kAlignmentC))
          {
            return Status::kErrorMisalignedOperand;
          }

          return Status::kSuccess;
        }

        /// Executes one GEMM
        CUTLASS_DEVICE
        void operator()(Params const &params, SharedStorage &shared_storage)
        {

          // Compute threadblock location
          ThreadblockSwizzle threadblock_swizzle;

          cutlass::gemm::GemmCoord threadblock_tile_offset =
              threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

          // Early exit if CTA is out of range
          if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
              params.grid_tiled_shape.n() <= threadblock_tile_offset.n())
          {

            return;
          }

          // Compute initial location in logical coordinates
          cutlass::MatrixCoord tb_offset_A{
              threadblock_tile_offset.m() * Mma::Shape::kM,
              threadblock_tile_offset.k() * params.gemm_k_size,
          };

          cutlass::MatrixCoord tb_offset_B{
              threadblock_tile_offset.k() * params.gemm_k_size,
              threadblock_tile_offset.n() * Mma::Shape::kN};

          // Problem size is a function of threadblock index in the K dimension
          int problem_size_k = min(
              params.problem_size.k(),
              (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;

          // Compute position within threadblock
          int thread_idx = threadIdx.x;

          // Construct iterators to A and B operands
          typename Mma::IteratorA iterator_A(
              params.params_A,
              params.ref_A.data(),
              {params.problem_size.m(), problem_size_k},
              thread_idx,
              tb_offset_A,
              params.gather_A_indices);

          typename Mma::IteratorB iterator_B(
              params.params_B,
              params.ref_B.data(),
              {problem_size_k, params.problem_size.n()},
              thread_idx,
              tb_offset_B,
              params.gather_B_indices);

          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = canonical_warp_idx_sync();
          int lane_idx = threadIdx.x % 32;

          //
          // Main loop
          //

          using ElementInputA = cutlass::bfloat16_t;
          using LayoutInputA = cutlass::layout::RowMajor;
          using ElementInputB = cutlass::bfloat16_t;
          using LayoutInputB = cutlass::layout::RowMajor;
          using ElementAccumulator = float;
          using LayoutOutput = cutlass::layout::RowMajor;

          using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 128, 32>;
          using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;

          using IteratorA = cutlass::transform::threadblock::PredicatedTileAccessIterator<cutlass::MatrixShape<256, 32>, ElementInputA, LayoutInputA, 1, cutlass::transform::PitchLinearWarpRakedThreadMap<cutlass::PitchLinearShape<32, 256>, 256, cutlass::PitchLinearShape<4, 8>, 8>, cutlass::Array<cutlass::bfloat16_t, 8, false>, false, cutlass::layout::NoPermute>;
          using SmemIteratorA = cutlass::transform::threadblock::RegularTileAccessIterator<
              cutlass::MatrixShape<256, 32>,                                  // Shape
              ElementInputA,                                                  // Element
              cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 32>, // Layout
              0,                                                              // Advance rank
              cutlass::transform::PitchLinearWarpRakedThreadMap<
                  cutlass::PitchLinearShape<32, 256>, // Shape
                  256,                                // Interleaved
                  cutlass::PitchLinearShape<4, 8>,    // Lane
                  8>,                                 // Vector length
              16                                      // Vector length
              >;
          constexpr auto CacheOpA = cutlass::arch::CacheOperation::Global;
          using IteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<cutlass::MatrixShape<32, 128>, ElementInputB, LayoutInputB, 0, cutlass::transform::PitchLinearWarpRakedThreadMap<cutlass::PitchLinearShape<128, 32>, 256, cutlass::PitchLinearShape<8, 4>, 8>, cutlass::Array<cutlass::bfloat16_t, 8, false>, false, cutlass::layout::NoPermute>;
          using SmemIteratorB = cutlass::transform::threadblock::RegularTileAccessIterator<cutlass::MatrixShape<32, 128>, ElementInputB, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<16, 64>, 0, cutlass::transform::PitchLinearWarpRakedThreadMap<cutlass::PitchLinearShape<128, 32>, 256, cutlass::PitchLinearShape<8, 4>, 8>, 16>;
          constexpr auto CacheOpB = cutlass::arch::CacheOperation::Global;

          constexpr cutlass::gemm::SharedMemoryClearOption SharedMemoryClear = cutlass::gemm::SharedMemoryClearOption::kNone;
          using WarpTensorOp = cutlass::gemm::warp::MmaTensorOp<
              ShapeMMAWarp,
              ElementInputA,
              cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 32>,
              ElementInputB,
              cutlass::layout::RowMajorTensorOpMultiplicandCongruous<16, 64>,
              ElementAccumulator,
              LayoutOutput,
              cutlass::gemm::warp::MmaTensorOpPolicy<
                  cutlass::arch::Mma<
                      cutlass::gemm::GemmShape<16, 8, 16>,
                      32,
                      cutlass::bfloat16_t,
                      cutlass::layout::RowMajor,
                      cutlass::bfloat16_t,
                      cutlass::layout::ColumnMajor,
                      float,
                      cutlass::layout::RowMajor,
                      cutlass::arch::OpMultiplyAdd>,
                  cutlass::MatrixShape<1, 1>>,
              1,
              false>;
          using Policy = cutlass::gemm::threadblock::MmaPolicy<
              WarpTensorOp,
              cutlass::MatrixShape<0, 0>,
              cutlass::MatrixShape<0, 0>,
              1>;

          using MmaType = cutlass::gemm::threadblock::MmaMultistage<ShapeMMAThreadBlock, IteratorA, SmemIteratorA, CacheOpA, IteratorB, SmemIteratorB, CacheOpB, ElementAccumulator, LayoutOutput, Policy, 3, SharedMemoryClear>;
          static_assert(std::is_same<MmaType, Mma>::value, "MmaType is not Mma");

          // this invokes MmaMultistage(). You should inline the constructor and declare all fields as local variables.
          MmaType mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

          typename MmaType::FragmentC accumulators;

          accumulators.clear();

          if (!kSplitKSerial || gemm_k_iterations > 0)
          {
            // this invokes MmaMultistage::operator(). Your job is to inline this operator.
            mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
          }

          //
          // Epilogue
          //

          OutputOp output_op(params.output_op);

          //
          // Masked tile iterators constructed from members
          //

          threadblock_tile_offset =
              threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

          // assume identity swizzle
          MatrixCoord threadblock_offset(
              threadblock_tile_offset.m() * Mma::Shape::kM,
              threadblock_tile_offset.n() * Mma::Shape::kN);

          int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

          // Construct the semaphore.
          Semaphore semaphore(params.semaphore + block_idx, thread_idx);

          // Tile iterator loading from source tensor.
          typename Epilogue::OutputTileIterator iterator_C(
              params.params_C,
              params.ref_C.data(),
              params.problem_size.mn(),
              thread_idx,
              threadblock_offset,
              params.scatter_D_indices);

          // Tile iterator writing to destination tensor.
          typename Epilogue::OutputTileIterator iterator_D(
              params.params_D,
              params.ref_D.data(),
              params.problem_size.mn(),
              thread_idx,
              threadblock_offset,
              params.scatter_D_indices);

          Epilogue epilogue(
              shared_storage.epilogue,
              thread_idx,
              warp_idx,
              lane_idx);

          // Wait on the semaphore - this latency may have been covered by iterator construction
          if (kSplitKSerial && params.grid_tiled_shape.k() > 1)
          {

            // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
            if (threadblock_tile_offset.k())
            {
              iterator_C = iterator_D;
            }

            semaphore.wait(threadblock_tile_offset.k());
          }

          // Execute the epilogue operator to update the destination tensor.
          epilogue(output_op, iterator_D, accumulators, iterator_C);

          //
          // Release the semaphore
          //

          if (kSplitKSerial && params.grid_tiled_shape.k() > 1)
          {

            int lock = 0;
            if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1)
            {

              // The final threadblock resets the semaphore for subsequent grids.
              lock = 0;
            }
            else
            {
              // Otherwise, the semaphore is incremented
              lock = threadblock_tile_offset.k() + 1;
            }

            semaphore.release(lock);
          }
        }
      };

      /////////////////////////////////////////////////////////////////////////////////////////////////

    } // namespace kernel
  } // namespace gemm
} // namespace cutlass
