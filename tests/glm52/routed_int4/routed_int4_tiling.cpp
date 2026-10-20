/**
 * The cube tilings, in their own translation unit.
 *
 * `matmul_tiling.h` is host-only and pulls in the operator registry's types,
 * so it stays away from the file `bisheng` compiles. Take the
 * `AscendC::tiling::TCubeTiling` overload of `GetTiling`: the `optiling` one
 * needs `TilingDef` symbols that live in `register`.
 */
#include <cstring>

#include "lib/matmul/matmul_tiling.h"

extern "C" int routed_int4_build_tiling(int rows, int columns, int inner,
                                   int width, void* buffer)
{
    matmul_tiling::MatmulApiTiling tiling;
    // Both operands are INT4, which is the only pair the cube's `mad_s4`
    // takes and the only one the library will pass: it rejects a mixed pair
    // because it plans to widen INT4 to INT8 entering L1.
    tiling.SetAType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_INT4, false);
    tiling.SetBType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_INT4, false);
    tiling.SetCType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_INT32);
    tiling.SetShape(rows, columns, inner);
    tiling.SetOrgShape(rows, width, inner);
    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1);

    AscendC::tiling::TCubeTiling cube;
    const int64_t status = tiling.GetTiling(cube);
    if (status != 0) {
        return static_cast<int>(status);
    }
    std::memcpy(buffer, &cube, sizeof(cube));
    return 0;
}
