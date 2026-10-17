#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 VLLM_ASCEND_SOURCE NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi
: "${ASCEND_HOME_PATH:?Source the CANN/Python activation before this script}"

source_dir=$(realpath "$1")
output_dir=$(realpath -m "$2")
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
patch_file="$script_dir/patches/gmm1-swiglu-barrier.patch"
op=grouped_matmul_swiglu_quant_v2
kernel_header="gmm/$op/op_kernel/${op}_a8w4_msd_post.h"

if [[ ! -f "$source_dir/csrc/$kernel_header" ]]; then
    echo "GMM1 source not found: $source_dir/csrc/$kernel_header" >&2
    exit 1
fi
if [[ -e "$output_dir" ]]; then
    echo "Output directory must be new: $output_dir" >&2
    exit 1
fi
case "$output_dir/" in
    "$source_dir/"*)
        echo "Output directory must be outside the original source" >&2
        exit 1
        ;;
esac

build_source="$output_dir/source/csrc"
mkdir -p "$build_source"
rsync -a --exclude=/build/ --exclude=/build_cache/ \
    --exclude=/output/ --exclude=/build_out/ --exclude=.git/ \
    --exclude=__pycache__/ "$source_dir/csrc/" "$build_source/"
sha256sum "$source_dir/csrc/$kernel_header" "$patch_file" \
    "$script_dir/build_gmm1_barrier.sh" > "$output_dir/inputs.sha256"
patch --batch --fuzz=0 -p1 -d "$build_source" < "$patch_file"

(
    cd "$build_source"
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    export CPATH="$PWD/third_party/catlass/include${CPATH:+:$CPATH}"
    export MAX_JOBS=8
    bash build.sh --pkg --ops="$op" --soc=ascend910_93 \
        --vendor_name=gmm1barrier --build-type=Release -j8
) 2>&1 | tee "$output_dir/build.log"

package="$build_source/build/cann-ops-transformer-gmm1barrier_linux-aarch64.run"
bash "$package" --install-path="$output_dir/vendor" \
    2>&1 | tee "$output_dir/install.log"
vendor="$output_dir/vendor/vendors/gmm1barrier_transformer"
original_vendor="$source_dir/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
overlay="$output_dir/overlay/vendors/custom_transformer"
mkdir -p "$(dirname -- "$overlay")"
cp -a --reflink=auto "$original_vendor" "$overlay"
binary_dir="op_impl/ai_core/tbe/kernel/ascend910_93/$op"
variant=GroupedMatmulSwigluQuantV2_fa3d6d3de6e1f32e170e39d2ddd3a20e
for suffix in o json; do
    cp "$vendor/$binary_dir/$variant.$suffix" \
        "$overlay/$binary_dir/$variant.$suffix"
done
installed_header="op_impl/ai_core/tbe/custom_transformer_impl/ascendc/$op/"
installed_header+="${op}_a8w4_msd_post.h"
cp "$build_source/$kernel_header" "$overlay/$installed_header"
sha256sum "$package" "$vendor/op_api/lib/libcust_opapi.so" \
    "$build_source/$kernel_header" "$overlay/$binary_dir/$variant.o" \
    "$overlay/$binary_dir/$variant.json" \
    "$original_vendor/op_api/lib/libcust_opapi.so" \
    "$overlay/op_api/lib/libcust_opapi.so" > "$output_dir/artifacts.sha256"
printf 'ASCEND_CUSTOM_OPP_PATH=%s:%s\n' "$overlay" "$original_vendor"
