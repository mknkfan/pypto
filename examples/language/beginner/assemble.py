# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tile assemble: write a source tile into a target tile at a specified offset.

Hardware semantics (PTO backend):
  tile.assemble maps to the TINSERT instruction. The hardware mode is inferred
  automatically from the memory spaces of the operands:

  Acc→Mat (TInsertMode::NZ) — source from Acc (L0C), target in Mat (L1):
    - target tile: in Mat (L1), fractal layout
    - source tile: in Acc (L0C), fractal layout (always FP32, output of tile.matmul)
    Data flow:
      a, b (GM) → Mat → Left/Right → tile.matmul → Acc (FP32)
      x   (GM) → Mat (FP32) [target]
      TINSERT NZ: Acc → Mat [at offset]
      Mat → Vec → GM

  Vec→Vec (TInsertMode::ND_VEC) — both tiles in Vec (UB), RowMajor/ND layout:
    - target tile: in Vec (UB), ND layout
    - source tile: in Vec (UB), ND layout
    Data flow:
      x   (GM) → Vec (UB) [target]
      src (GM) → Vec (UB) [source]
      TINSERT ND_VEC: Vec → Vec [at offset]
      Vec → GM

Programs (one representative per distinct pattern):
  TileAssembleAccMatProgram          — Acc→Mat: matmul(a[32,16], b[16,16]) → x[32,32] at [0, 16]
  TileAssembleVecProgram             — Vec→Vec: src[32,16] → x[32,32] at [0, 0]  (single-shot)
  TileAssembleRowByRowProgram        — Vec→Vec: loop i, pl.slice row i, assemble at [i, 0]
  TileAssembleDoubleLoopProgram      — Vec→Vec: nested loops b×i, pl.slice row b*8+i, assemble at [b*8+i, 0]
  TileAssembleLoopColBroadcastProgram — Vec→Vec: loop c, same src[32,8] at [0, c*8]  (no pl.slice)
  TileAssembleDoubleLoopBroadcastProgram — Vec→Vec: nested b×c, same src[16,16] at [b*16, c*16]  (no pl.slice)
"""

import pypto.language as pl


@pl.program
class TileAssembleAccMatProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_assemble(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        a: pl.Tensor[[32, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        # Load target into Mat (L1)
        tile_x = pl.load(x, offsets=[0, 0], shapes=[32, 32], target_memory=pl.MemorySpace.Mat)
        # Produce Acc (L0C, FP32) via matmul: GM → Mat → Left/Right → matmul
        tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[32, 16], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[16, 16], target_memory=pl.MemorySpace.Mat)
        tile_a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_src = pl.matmul(tile_a, tile_b)  # FP32 Acc (L0C) — same dtype as tile_x
        # Assemble: insert tile_src into the right half of tile_x at offset [0, 16]
        result = pl.tile.assemble(tile_x, tile_src, [0, 16])
        # Move Mat → Vec before store
        result_vec = pl.move(result, target_memory=pl.MemorySpace.Vec)
        out_y = pl.store(result_vec, offsets=[0, 0], output_tensor=y)
        return out_y

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        a: pl.Tensor[[32, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        y = self.tile_assemble(x, a, b, y)
        return y


@pl.program
class TileAssembleVecProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_assemble(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        # Load target and source into Vec (UB) — ND/RowMajor layout
        tile_x = pl.load(x, offsets=[0, 0], shapes=[32, 32], target_memory=pl.MemorySpace.Vec)
        tile_src = pl.load(src, offsets=[0, 0], shapes=[32, 16], target_memory=pl.MemorySpace.Vec)
        # Assemble: insert src into the left half of x at [0, 0] — ND_VEC mode
        result = pl.tile.assemble(tile_x, tile_src, [0, 0])
        out_y = pl.store(result, offsets=[0, 0], output_tensor=y)
        return out_y

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        y = self.tile_assemble(x, src, y)
        return y


@pl.program
class TileAssembleRowByRowProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_assemble(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        # Load target (32×32) and source (32×16) into Vec (UB)
        tile_x = pl.load(x, offsets=[0, 0], shapes=[32, 32], target_memory=pl.MemorySpace.Vec)
        tile_src = pl.load(src, offsets=[0, 0], shapes=[32, 16], target_memory=pl.MemorySpace.Vec)
        # For each row i: slice src[i, :] with a dynamic row offset, assemble at [i, 0].
        # Models the k_group gathering pattern in real workloads (e.g. Qwen KV-head loop).
        for i in pl.range(32):
            row = pl.slice(tile_src, [1, 16], [i, 0])
            tile_x = pl.tile.assemble(tile_x, row, [i, 0])
        out_y = pl.store(tile_x, offsets=[0, 0], output_tensor=y)
        return out_y

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        y = self.tile_assemble(x, src, y)
        return y


@pl.program
class TileAssembleDoubleLoopProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_assemble(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        # Load target (32×32) and source (32×16) into Vec (UB)
        tile_x = pl.load(x, offsets=[0, 0], shapes=[32, 32], target_memory=pl.MemorySpace.Vec)
        tile_src = pl.load(src, offsets=[0, 0], shapes=[32, 16], target_memory=pl.MemorySpace.Vec)
        # Outer loop: 4 row-blocks; inner loop: 8 rows per block.
        # Row index row = b * 8 + i mirrors the batch×head two-level indexing in real workloads.
        for b in pl.range(4):
            for i in pl.range(8):
                row = b * 8 + i
                tile_row = pl.slice(tile_src, [1, 16], [row, 0])
                tile_x = pl.tile.assemble(tile_x, tile_row, [row, 0])
        out_y = pl.store(tile_x, offsets=[0, 0], output_tensor=y)
        return out_y

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        y = self.tile_assemble(x, src, y)
        return y


@pl.program
class TileAssembleLoopColBroadcastProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_assemble(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 8], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        # Load target and source into Vec (UB) with static offsets — no pl.slice needed.
        tile_x = pl.load(x, offsets=[0, 0], shapes=[32, 32], target_memory=pl.MemorySpace.Vec)
        tile_src = pl.load(src, offsets=[0, 0], shapes=[32, 8], target_memory=pl.MemorySpace.Vec)
        # Loop over 4 column-blocks (width 8 each); assemble the same tile_src at each position.
        # Dynamic offset [0, c * 8] — no pl.slice required.
        for c in pl.range(4):
            tile_x = pl.tile.assemble(tile_x, tile_src, [0, c * 8])
        out_y = pl.store(tile_x, offsets=[0, 0], output_tensor=y)
        return out_y

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 8], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        y = self.tile_assemble(x, src, y)
        return y


@pl.program
class TileAssembleDoubleLoopBroadcastProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_assemble(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[16, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        # Load target and source into Vec (UB) with static offsets — no pl.slice needed.
        tile_x = pl.load(x, offsets=[0, 0], shapes=[32, 32], target_memory=pl.MemorySpace.Vec)
        tile_src = pl.load(src, offsets=[0, 0], shapes=[16, 16], target_memory=pl.MemorySpace.Vec)
        # Outer loop: 2 row-blocks; inner loop: 2 column-blocks.
        # Assembles tile_src into all four [16,16] quadrants of tile_x.
        # Both offsets [b*16, c*16] computed from loop vars — no pl.slice required.
        for b in pl.range(2):
            for c in pl.range(2):
                tile_x = pl.tile.assemble(tile_x, tile_src, [b * 16, c * 16])
        out_y = pl.store(tile_x, offsets=[0, 0], output_tensor=y)
        return out_y

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[16, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        y = self.tile_assemble(x, src, y)
        return y
