#pragma once

#include "slang-ir.h"

namespace Slang
{
class DiagnosticSink;
class TargetProgram;

/// Returns true if the module contains any ray tracing entry points.
bool hasRayTracingEntryPoints(IRModule* module);

/// Legalize raygen entry points for Metal by transforming them into compute shaders.
/// This pass must run before legalizeIRForMetal.
void legalizeIRForMetalRT(IRModule* module, TargetProgram* targetProgram, DiagnosticSink* sink);

} // namespace Slang
