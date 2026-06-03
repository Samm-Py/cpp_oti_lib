#!/usr/bin/env bash
set -eu
set -o pipefail

CXX=${CXX:-c++}
CXXFLAGS=${CXXFLAGS:-"-std=c++17 -O2 -Wall -Wextra -pedantic"}
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-/tmp/otinum_unit_tests}
LOG_DIR=${LOG_DIR:-"$ROOT_DIR/logs"}
RUN_ID=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=${LOG_FILE:-"$LOG_DIR/unit_tests_$RUN_ID.log"}

mkdir -p "$BUILD_DIR"
mkdir -p "$LOG_DIR"

log()
{
    printf '%s\n' "$*" | tee -a "$LOG_FILE"
}

run_logged()
{
    log "+ $*"
    "$@" 2>&1 | tee -a "$LOG_FILE"
}

log "otinum focused unit test run"
log "timestamp: $RUN_ID"
log "compiler: $CXX"
log "flags: $CXXFLAGS"
log "build directory: $BUILD_DIR"
log "log file: $LOG_FILE"
log ""

for source in "$ROOT_DIR"/tests/test_*.cpp; do
    name=$(basename "$source" .cpp)
    if [ "$name" = "test_kokkos_smoke" ] && [ "${OTI_ENABLE_KOKKOS:-OFF}" != "ON" ]; then
        log "skipping $name (set OTI_ENABLE_KOKKOS=ON and use CMake to build Kokkos tests)"
        log ""
        continue
    fi
    exe="$BUILD_DIR/$name"
    log "building $name"
    # shellcheck disable=SC2086
    run_logged "$CXX" $CXXFLAGS -I "$ROOT_DIR/include" -I "$ROOT_DIR/tests" "$source" -o "$exe"
    log "running  $name"
    run_logged "$exe"
    log ""
done

log "all focused otinum unit tests passed"
