#pragma once

// Public umbrella header for the OTI library.
//
// Users that want the complete public API should include this file. It exposes
// the static OTI value type from core.hpp, the <cmath>-style analytic overloads
// from functions.hpp, the standard-library interoperability layer
// (numeric_limits, streaming, and the remaining <cmath> overloads) from
// interop.hpp, and the coefficient-major array view from soa.hpp. More
// selective users can include those headers directly.

#include "otinum/core.hpp"
#include "otinum/functions.hpp"
#include "otinum/interop.hpp"
#include "otinum/soa.hpp"
