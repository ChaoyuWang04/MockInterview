---
difficulty: 中等
topic: CUDA/矩阵转置
summary: 用 CUDA 写矩阵转置,会遇到哪些访存问题
tags: [真题, 待校对, 手撕代码, CUDA, 合并访存, bank conflict]
company:
mastered: false
highfreq: false
---

## 题目

用 CUDA 实现矩阵转置,这当中可能会遇到什么问题?请写出朴素版本、说明它慢在哪里,并给出优化后的实现。

## 要点

- 朴素版读是合并的,写不合并——每个线程写的地址跨了一整行
- 用 shared memory 中转,把不合并的全局写换成合并写
- 中转后 shared memory 出现 bank conflict,靠 padding 或 swizzle 消除
- 两次 `__syncthreads` 之间要覆盖整块,注意非整除边界
- 转置是纯访存算子,判断标准是有效带宽而不是耗时

## 答案

**朴素版的问题不是"读慢"而是"写不合并"。**

```cuda
__global__ void transpose_naive(const float* in, float* out, int W, int H) {
    int x = blockIdx.x * TILE + threadIdx.x;   // 列
    int y = blockIdx.y * TILE + threadIdx.y;   // 行
    if (x < W && y < H) out[x * H + y] = in[y * W + x];
}
```

同一 warp 内 `threadIdx.x` 连续,读 `in[y*W+x]` 地址连续,一次事务就能取回;但写 `out[x*H+y]` 中相邻线程的地址相差 `H` 个元素,每个线程各自触发一次事务,写带宽掉到几分之一。

**解法是用 shared memory 做中转**,让读和写都保持合并:

```cuda
__global__ void transpose_smem(const float* in, float* out, int W, int H) {
    __shared__ float tile[TILE][TILE + 1];     // +1 消除 bank conflict
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < W && y < H) tile[threadIdx.y][threadIdx.x] = in[y * W + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;       // 换成输出矩阵的坐标
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < H && y < W) out[y * H + x] = tile[threadIdx.x][threadIdx.y];
}
```

关键在写回时重新计算索引:线程块的 x/y 对调,使得同一 warp 写出的仍是连续地址。转置发生在读 shared memory 的 `tile[threadIdx.x][threadIdx.y]` 这一步。

**这时会撞上第二个问题:bank conflict。** `tile[threadIdx.x][threadIdx.y]` 让同一 warp 的线程访问同一列,若行长恰为 32,它们全落在同一个 bank 上,退化成 32 次串行访问。把行长改成 `TILE+1` 让它与 bank 数互质,每往下一行 bank 编号错开一位,冲突消失,代价只是每行多 4 字节。机制与 swizzle 的对比见 [访存类优化手段](../AI%20Infra/006-访存类优化手段.md)。

边界:矩阵尺寸非 TILE 整数倍时两次判断都要做,不能只判一次;转置是纯访存算子,验收看**有效带宽占峰值的比例**,而不是绝对耗时。

## 知识点

合并访存、shared memory 中转、bank conflict 与 padding、`__syncthreads` 的位置、有效带宽评估。

## 追问

- 为什么不能只在写回时判一次边界?
- `TILE+1` 会破坏 128-bit 向量化访存,这时该怎么办?
- 用 `float4` 向量化读写,转置还能怎么组织?
- 原地转置(方阵)与非原地相比,多了哪些约束?

## Note
