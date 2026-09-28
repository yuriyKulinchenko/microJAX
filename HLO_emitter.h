#ifndef HLO_EMITTER_H
#define HLO_EMITTER_H

#include <string>

#include "jax_types.h"

// Emits a jaxpr as a textual StableHLO module (MLIR assembly), suitable for
// passing to PJRT_Client_Compile with format = "mlir".
std::string emit_stablehlo(const jax::expression& jaxpr);

#endif //HLO_EMITTER_H
