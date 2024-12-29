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
          uint8_t const *sparsity_B;
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
              threadblock_tile_offset.m() * 256 /* Mma::Shape::kM */,
              threadblock_tile_offset.k() * params.gemm_k_size,
          };

          cutlass::MatrixCoord tb_offset_B{
              threadblock_tile_offset.k() * params.gemm_k_size,
              threadblock_tile_offset.n() * 128 /* Mma::Shape::kN */};

          // Problem size is a function of threadblock index in the K dimension
          int problem_size_k = min(
              params.problem_size.k(),
              (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + 32 /* Mma::Shape::kK */ - 1) / 32 /* Mma::Shape::kK */;

          // Compute position within threadblock
          int thread_idx = threadIdx.x;

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
          using SmemIteratorA = cutlass::transform::threadblock::RegularTileAccessIterator<cutlass::MatrixShape<256, 32>, ElementInputA, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 32>, 0, cutlass::transform::PitchLinearWarpRakedThreadMap<cutlass::PitchLinearShape<32, 256>, 256, cutlass::PitchLinearShape<4, 8>, 8>, 16>;
          constexpr auto CacheOpA = cutlass::arch::CacheOperation::Global;
          using IteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<cutlass::MatrixShape<32, 128>, ElementInputB, LayoutInputB, 0, cutlass::transform::PitchLinearWarpRakedThreadMap<cutlass::PitchLinearShape<128, 32>, 256, cutlass::PitchLinearShape<8, 4>, 8>, cutlass::Array<cutlass::bfloat16_t, 8, false>, false, cutlass::layout::NoPermute>;
          using SmemIteratorB = cutlass::transform::threadblock::RegularTileAccessIterator<cutlass::MatrixShape<32, 128>, ElementInputB, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<16, 64>, 0, cutlass::transform::PitchLinearWarpRakedThreadMap<cutlass::PitchLinearShape<128, 32>, 256, cutlass::PitchLinearShape<8, 4>, 8>, 16>;
          constexpr auto CacheOpB = cutlass::arch::CacheOperation::Global;

          IteratorA iterator_A(
              params.params_A,
              params.ref_A.data(),
              {params.problem_size.m(), problem_size_k},
              thread_idx,
              tb_offset_A,
              params.gather_A_indices);

          IteratorB iterator_B(
              params.params_B,
              params.ref_B.data(),
              {problem_size_k, params.problem_size.n()},
              thread_idx,
              tb_offset_B,
              params.gather_B_indices);

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
          constexpr auto Stages = 3;

          using MmaTypeForbidden = cutlass::gemm::threadblock::MmaMultistage<ShapeMMAThreadBlock, IteratorA, SmemIteratorA, CacheOpA, IteratorB, SmemIteratorB, CacheOpB, ElementAccumulator, LayoutOutput, Policy, Stages, SharedMemoryClear>;
          using Detail = MmaTypeForbidden::Detail;
          // Declare variables that were fields in MmaMultistage
          typename WarpTensorOp::IteratorA warp_tile_iterator_A_(
              shared_storage.main_loop.operand_A_ref(), lane_idx);
          typename WarpTensorOp::IteratorB warp_tile_iterator_B_(
              shared_storage.main_loop.operand_B_ref(), lane_idx);
          WarpTensorOp warp_mma_;
          SmemIteratorA smem_iterator_A_(
              shared_storage.main_loop.operand_A_ref(), thread_idx);
          SmemIteratorB smem_iterator_B_(
              shared_storage.main_loop.operand_B_ref(), thread_idx);
          int smem_write_stage_idx_ = 0;
          int smem_read_stage_idx_ = 0;

          // Compute warp location within threadblock tile by mapping the warp_id to three coordinates:
          //   warp_idx_m: warp's position within the threadblock along the M dimension
          //   warp_idx_n: warp's position within the threadblock along the N dimension
          //   warp_idx_k: warp's position within the threadblock along the K dimension

          const int WarpCount_kM = ShapeMMAThreadBlock::kM / ShapeMMAWarp::kM; // 256/64 = 4
          const int WarpCount_kN = ShapeMMAThreadBlock::kN / ShapeMMAWarp::kN; // 128/64 = 2
          const int WarpCount_kK = ShapeMMAThreadBlock::kK / ShapeMMAWarp::kK; // 32/32 = 1

          int warp_idx_mn = warp_idx % (WarpCount_kM * WarpCount_kN);
          int warp_idx_k = warp_idx / (WarpCount_kM * WarpCount_kN);

          int warp_idx_m = warp_idx_mn % WarpCount_kM;
          int warp_idx_n = warp_idx_mn / WarpCount_kM;

          auto workerid = threadIdx.x / 32;

          // kWarpGemmIterations
          constexpr auto kWarpGemmIterations = ShapeMMAWarp::kK / WarpTensorOp::Policy::MmaShape::kK; // 32/16=2
          constexpr auto kSparsityBSize = 4;

          // Add per-warp offsets in units of warp-level tiles
          warp_tile_iterator_A_.add_tile_offset(
              {warp_idx_m, kWarpGemmIterations * warp_idx_k});
          warp_tile_iterator_B_.add_tile_offset(
              {kWarpGemmIterations * warp_idx_k, warp_idx_n});

          // Fragment of accumulator tile
          using FragmentC = typename WarpTensorOp::FragmentC;
          FragmentC accumulators;

          accumulators.clear();

          // Begin inlined code from MmaMultistage::operator()

          // Function to advance shared memory read stage
          auto advance_smem_read_stage = [&]()
          {
            ++smem_read_stage_idx_;

            if (smem_read_stage_idx_ == Stages)
            {
              // Wrap back around to the 'start' of the circular buffer in shared memory
              warp_tile_iterator_A_.add_tile_offset({0, -Stages * Policy::kPartitionsK * kWarpGemmIterations});
              warp_tile_iterator_B_.add_tile_offset({-Stages * Policy::kPartitionsK * kWarpGemmIterations, 0});
              smem_read_stage_idx_ = 0;
            }
          };

          auto global_sparsity_B_iterator = params.sparsity_B + threadblock_tile_offset.n() * kSparsityBSize +
                                            params.grid_tiled_shape.n() * kSparsityBSize * workerid;
          auto smem_sparsity_B_iterator_write = shared_storage.main_loop.sparsity_B.data() + workerid * kSparsityBSize;
          auto load_sparsity_into_smem = [&](int cur_k_block, int cur_k_subtile)
          {
            if (gemm_k_iterations <= 0)
              return;
            auto smem_sparsity_B = smem_sparsity_B_iterator_write + smem_write_stage_idx_ * kWarpGemmIterations * kSparsityBSize;
            cutlass::arch::cp_async<kSparsityBSize, cutlass::arch::CacheOperation::Always>(smem_sparsity_B, global_sparsity_B_iterator);
            global_sparsity_B_iterator += params.grid_tiled_shape.n() * kSparsityBSize * kWarpGemmIterations;

            // *smem_sparsity_B = *global_sparsity_B;
            // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
            // {
            //   printf("Index %d: [%d, %d] = %d (smem_write_stage_idx_ = %d)\n", smem_sparsity_B_index, cur_k_block, cur_k_subtile, value, smem_write_stage_idx_);
            // }
          };

          // Function to advance shared memory write stage
          auto advance_smem_write_stage = [&](IteratorA &iterator_A, IteratorB &iterator_B)
          {
            // Advance global iterators
            iterator_A.add_tile_offset({0, 1});
            iterator_B.add_tile_offset({1, 0});

            // Advance shared iterators
            smem_iterator_A_.add_tile_offset({0, 1});
            smem_iterator_B_.add_tile_offset({1, 0});

            // Increment shared memory write stage index
            ++smem_write_stage_idx_;

            if (smem_write_stage_idx_ == Stages)
            {
              // Wrap back around to the 'start' of the circular buffer in shared memory
              smem_iterator_A_.add_tile_offset({0, -Stages});
              smem_iterator_B_.add_tile_offset({-Stages, 0});
              smem_write_stage_idx_ = 0;
            }
          };

          // Function to perform async copy tiles and advance
          auto copy_tiles_and_advance = [&](IteratorA &iterator_A, IteratorB &iterator_B,
                                            int group_start_A = 0, int group_start_B = 0)
          {
            iterator_A.set_iteration_index(group_start_A *
                                           IteratorA::kAccessesPerVector);
            smem_iterator_A_.set_iteration_index(group_start_A);

            // Async Copy for operand A
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < Detail::kAccessesPerGroupA; ++j)
            {
              if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA)
              {
                typename IteratorA::AccessType *dst_ptr =
                    reinterpret_cast<typename IteratorA::AccessType *>(
                        smem_iterator_A_.get());

                int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value *
                                      IteratorA::ThreadMap::kElementsPerAccess /
                                      IteratorA::kAccessesPerVector / 8;

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorA::kAccessesPerVector; ++v)
                {
                  auto gmem_ptr = iterator_A.get();

                  if (SharedMemoryClear == SharedMemoryClearOption::kZfill)
                  {
                    cutlass::arch::cp_async_zfill<kSrcBytes, CacheOpA>(
                        dst_ptr + v, gmem_ptr, iterator_A.valid());
                  }
                  else
                  {
                    cutlass::arch::cp_async<kSrcBytes, CacheOpA>(
                        dst_ptr + v, gmem_ptr, iterator_A.valid());
                  }

                  ++iterator_A;
                }

                ++smem_iterator_A_;
              }
            }

            iterator_B.set_iteration_index(group_start_B *
                                           IteratorB::kAccessesPerVector);
            smem_iterator_B_.set_iteration_index(group_start_B);

            // Async Copy for operand B
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < Detail::kAccessesPerGroupB; ++j)
            {
              if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB)
              {
                typename IteratorB::AccessType *dst_ptr =
                    reinterpret_cast<typename IteratorB::AccessType *>(
                        smem_iterator_B_.get());

                int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value *
                                      IteratorB::ThreadMap::kElementsPerAccess /
                                      IteratorB::kAccessesPerVector / 8;

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorB::kAccessesPerVector; ++v)
                {
                  auto gmem_ptr = iterator_B.get();

                  if (SharedMemoryClear == SharedMemoryClearOption::kZfill)
                  {
                    cutlass::arch::cp_async_zfill<kSrcBytes, CacheOpB>(
                        dst_ptr + v, gmem_ptr, iterator_B.valid());
                  }
                  else
                  {
                    cutlass::arch::cp_async<kSrcBytes, CacheOpB>(
                        dst_ptr + v, gmem_ptr, iterator_B.valid());
                  }

                  ++iterator_B;
                }
                ++smem_iterator_B_;
              }
            }
          };

          // Declare PipeState struct as method-local struct
          struct PipeState
          {
            using WarpLoadedFragmentA = typename WarpTensorOp::FragmentA;
            using WarpLoadedFragmentB = typename WarpTensorOp::FragmentB;
            using WarpTransformedFragmentA = typename WarpTensorOp::TransformedFragmentA;
            using WarpTransformedFragmentB = typename WarpTensorOp::TransformedFragmentB;

            FragmentC tmp_accum_;

            WarpLoadedFragmentA warp_loaded_frag_A_[2];
            WarpTransformedFragmentA warp_transformed_frag_A_[2];

            WarpLoadedFragmentB warp_loaded_frag_B_[2];
            WarpTransformedFragmentB warp_transformed_frag_B_[2];
          };

          // Prologue (start fetching iterations of global fragments into shared memory)
          // Issue several complete stages
          CUTLASS_PRAGMA_UNROLL
          for (int stage = 0; stage < Stages - 1; ++stage, --gemm_k_iterations)
          {
            iterator_A.set_iteration_index(0);
            smem_iterator_A_.set_iteration_index(0);

            // Async Copy for operand A
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j)
            {
              typename IteratorA::AccessType *dst_ptr = reinterpret_cast<typename IteratorA::AccessType *>(smem_iterator_A_.get());

              CUTLASS_PRAGMA_UNROLL
              for (int v = 0; v < IteratorA::kAccessesPerVector; ++v)
              {
                int const kSrcBytes =
                    sizeof_bits<typename IteratorA::Element>::value *
                    IteratorA::ThreadMap::kElementsPerAccess /
                    IteratorA::kAccessesPerVector / 8;

                cutlass::arch::cp_async_zfill<kSrcBytes, CacheOpA>(dst_ptr + v, iterator_A.get(), iterator_A.valid());

                ++iterator_A;
              }

              ++smem_iterator_A_;
            }

            iterator_B.set_iteration_index(0);
            smem_iterator_B_.set_iteration_index(0);

            // Async Copy for operand B
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j)
            {
              typename IteratorB::AccessType *dst_ptr = reinterpret_cast<typename IteratorB::AccessType *>(smem_iterator_B_.get());

              CUTLASS_PRAGMA_UNROLL
              for (int v = 0; v < IteratorB::kAccessesPerVector; ++v)
              {
                int const kSrcBytes =
                    sizeof_bits<typename IteratorB::Element>::value *
                    IteratorB::ThreadMap::kElementsPerAccess /
                    IteratorB::kAccessesPerVector / 8;

                cutlass::arch::cp_async_zfill<kSrcBytes, CacheOpB>(
                    dst_ptr + v, iterator_B.get(), iterator_B.valid());

                ++iterator_B;
              }

              ++smem_iterator_B_;
            }

            if (workerid < kWarpGemmIterations)
            {
              load_sparsity_into_smem(stage, workerid);
            }

            // Move to the next write stage
            advance_smem_write_stage(iterator_A, iterator_B);

            // Defines the boundary of a stage of cp.async.
            cutlass::arch::cp_async_fence();
          }

          // Wait until we have at least one completed global fetch stage
          cutlass::arch::cp_async_wait<Stages - 2>();
          __syncthreads();

          // Initialize destination accumulators with source accumulators
          PipeState pipe_state;

          // Load first warp-tile's A fragment from shared memory
          warp_tile_iterator_A_.set_kgroup_index(0);
          warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
          ++warp_tile_iterator_A_;

          // Load first warp-tile's B fragment from shared memory
          warp_tile_iterator_B_.set_kgroup_index(0);
          warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
          ++warp_tile_iterator_B_;

          // Transform, if necessary, the first warp-tile's shared memory fragments
          warp_mma_.transform(
              pipe_state.warp_transformed_frag_A_[0],
              pipe_state.warp_transformed_frag_B_[0],
              pipe_state.warp_loaded_frag_A_[0],
              pipe_state.warp_loaded_frag_B_[0]);

          if (Detail::kStagedAccumulation)
          {
            pipe_state.tmp_accum_.clear();
          }

          // Mainloop
          CUTLASS_GEMM_LOOP
          for (int load_k_block_sparsity = Stages - 1;
               gemm_k_iterations > (-Stages + 1);
               load_k_block_sparsity++)
          {
            // Unroll the warp-level MMA tiles of a threadblock's mainloop iteration
            CUTLASS_PRAGMA_UNROLL
            for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k)
            {
              // A = 256x16, B = 16x128

              // Load the next warp-tile's A fragment from shared memory
              warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIterations);
              warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
              ++warp_tile_iterator_A_;

              // Load the next warp-tile's B fragment from shared memory
              warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIterations);
              warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
              ++warp_tile_iterator_B_;

              // Except for the first warp-tile, all warp-tiles convert their incoming shared memory fragments as necessary
              if (warp_mma_k > 0)
              {
                warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
              }
              using ArchMmaOperator = typename WarpTensorOp::ArchMmaOperator;
              using MmaIterations = typename WarpTensorOp::MmaIterations;

              ArchMmaOperator arch_mma_op;
              using MmaOperandA = typename ArchMmaOperator::FragmentA;
              using MmaOperandB = typename ArchMmaOperator::FragmentB;
              using MmaOperandC = typename ArchMmaOperator::FragmentC;

              MmaOperandA const *ptr_A = reinterpret_cast<MmaOperandA const *>(&pipe_state.warp_transformed_frag_A_[warp_mma_k % 2]);
              MmaOperandB const *ptr_B = reinterpret_cast<MmaOperandB const *>(&pipe_state.warp_transformed_frag_B_[warp_mma_k % 2]);
              MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&accumulators);

              auto cur_k_block = load_k_block_sparsity - (Stages - 1);
              uint8_t local_sparsity_B = shared_storage.main_loop.sparsity_B.data()[((cur_k_block % Stages) * kWarpGemmIterations + warp_mma_k) * kSparsityBSize];
              auto laneid = threadIdx.x % 32;
              auto warp_subtile_x = (workerid / WarpCount_kM);
              // auto offset_N = threadblock_tile_offset.n();
              // auto offset_K = params.grid_tiled_shape.n() * (cur_k_block * kWarpGemmIterations + warp_mma_k);
              // uint8_t local_sparsity_B_global = params.sparsity_B[offset_K + offset_N];
              // if (local_sparsity_B_global != local_sparsity_B && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && laneid == 0 && workerid == 0)
              // {
              //   printf("Worker %d: Mismatch at [%d, %d] = %d (global = %d)\n", workerid, cur_k_block, warp_mma_k, local_sparsity_B, local_sparsity_B_global);
              // }

              CUTLASS_PRAGMA_UNROLL
              for (int two_n = 0; two_n < MmaIterations::kColumn; two_n += 2)
              {
                auto bit_to_access = warp_subtile_x * WarpCount_kM + (two_n / 2);
                if (!(local_sparsity_B & (1 << (7 - bit_to_access))))
                  continue;
                CUTLASS_PRAGMA_UNROLL
                for (int n = two_n; n < two_n + 2; n++)
                {
                  CUTLASS_PRAGMA_UNROLL
                  for (int m = 0; m < MmaIterations::kRow; ++m)
                  {
                    int m_serpentine = (n & 1) ? (MmaIterations::kRow - 1 - m) : m;

                    arch_mma_op(ptr_D[m_serpentine + n * MmaIterations::kRow],
                                ptr_A[m_serpentine],
                                ptr_B[n],
                                ptr_D[m_serpentine + n * MmaIterations::kRow]);
                  }
                }
              }

              // Except for the last warp-tile, all warp-tiles issue their share of
              // global->shared fragment copies
              if (warp_mma_k < kWarpGemmIterations - 1)
              {

                int group_start_iteration_A, group_start_iteration_B;
                group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
                group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;
                if (workerid == warp_mma_k)
                  load_sparsity_into_smem(load_k_block_sparsity, warp_mma_k);

                copy_tiles_and_advance(
                    iterator_A,
                    iterator_B,
                    group_start_iteration_A,
                    group_start_iteration_B);
              }

              // The second-to-last warp-tile also:
              //   - performs the last warp-tile's share of global->shared fragment copies
              //   - moves to the next global fetch stage
              if (warp_mma_k + 2 == kWarpGemmIterations)
              {

                // Performs the last warp-tile's share of global->shared fragment copies
                int group_start_iteration_A = (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
                int group_start_iteration_B = (warp_mma_k + 1) * Detail::kAccessesPerGroupB;

                copy_tiles_and_advance(
                    iterator_A,
                    iterator_B,
                    group_start_iteration_A,
                    group_start_iteration_B);
                if (workerid == warp_mma_k + 1)
                  load_sparsity_into_smem(load_k_block_sparsity, warp_mma_k + 1);

                // Inserts a memory fence between stages of cp.async instructions.
                cutlass::arch::cp_async_fence();

                // Wait until we have at least one completed global fetch stage
                cutlass::arch::cp_async_wait<Stages - 2>();
                __syncthreads();

                // Move to the next global fetch stage
                advance_smem_write_stage(iterator_A, iterator_B);
                advance_smem_read_stage();

                // Disable global fetching when done with global fetch iterations
                --gemm_k_iterations;
                iterator_A.clear_mask(gemm_k_iterations == 0);
                iterator_B.clear_mask(gemm_k_iterations == 0);
              }

              // The last warp-tile also converts the shared memory fragments used by
              // the first warp-tile of the next iteration, if necessary (so we can
              // immediately start issuing MMA instructions at the top of the loop )
              if (warp_mma_k + 1 == kWarpGemmIterations)
              {

                warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
              }
            }
          }

          if (Detail::kStagedAccumulation)
          {
            plus<FragmentC> plus_accum;
            accumulators = plus_accum(accumulators, pipe_state.tmp_accum_);
          }

          // Commit and drain all pending and predicated cp.async pnz from the GEMM mainloop
          cutlass::arch::cp_async_fence();
          cutlass::arch::cp_async_wait<0>();
          __syncthreads();
          // End of inlined code from MmaMultistage::operator()

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
              threadblock_tile_offset.m() * 256 /* Mma::Shape::kM */,
              threadblock_tile_offset.n() * 128 /* Mma::Shape::kN */);

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