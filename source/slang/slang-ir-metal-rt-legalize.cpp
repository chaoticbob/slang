// slang-ir-metal-rt-legalize.cpp
#include "slang-ir-metal-rt-legalize.h"

#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{

bool hasRayTracingEntryPoints(IRModule* module)
{
    for (auto inst : module->getGlobalInsts())
    {
        auto func = as<IRFunc>(inst);
        if (!func)
            continue;
        if (auto entryPointDecor = func->findDecoration<IREntryPointDecoration>())
        {
            if (isRaytracingStage(entryPointDecor->getProfile().getStage()))
                return true;
        }
    }
    return false;
}

enum class MetalRTIntrinsic
{
    None,
    DispatchRaysIndex,
    DispatchRaysDimensions,
};

static MetalRTIntrinsic getMetalRTIntrinsicFromCall(IRCall* call)
{
    auto callee = call->getOperand(0);
    auto resolved = getResolvedInstForDecorations(callee);

    if (auto nameHint = resolved->findDecoration<IRNameHintDecoration>())
    {
        auto name = nameHint->getName();
        if (name == toSlice("DispatchRaysIndex"))
            return MetalRTIntrinsic::DispatchRaysIndex;
        if (name == toSlice("DispatchRaysDimensions"))
            return MetalRTIntrinsic::DispatchRaysDimensions;
    }
    return MetalRTIntrinsic::None;
}

static void legalizeRaygenEntryPoint(IRFunc* func, IRBuilder& builder)
{
    auto firstBlock = func->getFirstBlock();
    if (!firstBlock)
        return;

    // Build uint3 type.
    auto uintType = builder.getBasicType(BaseType::UInt);
    auto uint3Type = builder.getVectorType(uintType, 3);

    // 1. Add SV_DispatchThreadID parameter (maps to [[thread_position_in_grid]]).
    builder.setInsertInto(firstBlock);
    auto threadIdParam = builder.emitParam(uint3Type);
    builder.addNameHintDecoration(threadIdParam, toSlice("_sv_dispatch_thread_id"));
    builder.addSemanticDecoration(threadIdParam, toSlice("SV_DispatchThreadID"), 0);

    // 2. Add dispatch dimensions parameter (passed as constant buffer at [[buffer(30)]]).
    auto dimsParam = builder.emitParam(uint3Type);
    builder.addNameHintDecoration(dimsParam, toSlice("_metalrt_dispatch_dimensions"));
    builder.addTargetSystemValueDecoration(dimsParam, toSlice("buffer(30)"));

    // 3. Replace intrinsic calls with the new parameters.
    List<IRCall*> callsToRemove;
    for (auto block : func->getBlocks())
    {
        for (auto inst = block->getFirstOrdinaryInst(); inst; inst = inst->getNextInst())
        {
            auto call = as<IRCall>(inst);
            if (!call)
                continue;

            auto intrinsicKind = getMetalRTIntrinsicFromCall(call);
            if (intrinsicKind == MetalRTIntrinsic::None)
                continue;

            IRInst* replacement = nullptr;
            switch (intrinsicKind)
            {
            case MetalRTIntrinsic::DispatchRaysIndex:
                replacement = threadIdParam;
                break;
            case MetalRTIntrinsic::DispatchRaysDimensions:
                replacement = dimsParam;
                break;
            default:
                break;
            }

            if (replacement)
            {
                call->replaceUsesWith(replacement);
                callsToRemove.add(call);
            }
        }
    }

    for (auto call : callsToRemove)
    {
        call->removeAndDeallocate();
    }

    // 4. Change stage from RayGeneration to Compute.
    if (auto entryPointDecor = func->findDecoration<IREntryPointDecoration>())
    {
        auto profile = entryPointDecor->getProfile();
        profile.setStage(Stage::Compute);
        entryPointDecor->setOperand(
            0,
            builder.getIntValue(builder.getIntType(), profile.raw));
    }

    // 5. Add [numthreads(8, 8, 1)] decoration.
    auto intType = builder.getIntType();
    builder.addNumThreadsDecoration(
        func,
        builder.getIntValue(intType, 8),
        builder.getIntValue(intType, 8),
        builder.getIntValue(intType, 1));

    // 6. Fix up the function type to match the new parameter list.
    fixUpFuncType(func);
}

void legalizeIRForMetalRT(IRModule* module, TargetProgram* targetProgram, DiagnosticSink* sink)
{
    SLANG_UNUSED(targetProgram);
    SLANG_UNUSED(sink);

    IRBuilder builder(module);

    for (auto inst : module->getGlobalInsts())
    {
        auto func = as<IRFunc>(inst);
        if (!func)
            continue;

        auto entryPointDecor = func->findDecoration<IREntryPointDecoration>();
        if (!entryPointDecor)
            continue;

        auto stage = entryPointDecor->getProfile().getStage();
        if (stage != Stage::RayGeneration)
            continue;

        legalizeRaygenEntryPoint(func, builder);
    }
}

} // namespace Slang
