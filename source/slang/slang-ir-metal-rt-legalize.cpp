// slang-ir-metal-rt-legalize.cpp
#include "slang-ir-metal-rt-legalize.h"

#include "slang-ir-clone.h"
#include "slang-ir-insts.h"
#include "slang-ir.h"
#include "slang-target-program.h"

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
    TraceRay,
    WorldRayOrigin,
    WorldRayDirection,
    RayTMin,
    RayTCurrent,
    RayFlags,
    InstanceIndex,
    InstanceID,
    PrimitiveIndex,
    HitKind,
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
        if (name == toSlice("__metalrt_TraceRay"))
            return MetalRTIntrinsic::TraceRay;
        if (name == toSlice("WorldRayOrigin"))
            return MetalRTIntrinsic::WorldRayOrigin;
        if (name == toSlice("WorldRayDirection"))
            return MetalRTIntrinsic::WorldRayDirection;
        if (name == toSlice("RayTMin"))
            return MetalRTIntrinsic::RayTMin;
        if (name == toSlice("RayTCurrent"))
            return MetalRTIntrinsic::RayTCurrent;
        if (name == toSlice("RayFlags"))
            return MetalRTIntrinsic::RayFlags;
        if (name == toSlice("InstanceIndex"))
            return MetalRTIntrinsic::InstanceIndex;
        if (name == toSlice("InstanceID"))
            return MetalRTIntrinsic::InstanceID;
        if (name == toSlice("PrimitiveIndex"))
            return MetalRTIntrinsic::PrimitiveIndex;
        if (name == toSlice("HitKind"))
            return MetalRTIntrinsic::HitKind;
    }
    return MetalRTIntrinsic::None;
}

static void legalizeRaygenEntryPoint(IRFunc* func, IRBuilder& builder, int dispatchDimsBufferSlot)
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
    StringBuilder bufferSlotStr;
    bufferSlotStr << "buffer(" << dispatchDimsBufferSlot << ")";
    builder.addTargetSystemValueDecoration(dimsParam, bufferSlotStr.getUnownedSlice());

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

struct RTEntryPoints
{
    IRFunc* raygen = nullptr;
    IRFunc* closestHit = nullptr;
    IRFunc* miss = nullptr;
};

// Check if a struct type has the name "BuiltInTriangleIntersectionAttributes".
static bool isTriangleIntersectionAttrsType(IRType* type)
{
    auto structType = as<IRStructType>(type);
    if (!structType)
        return false;
    if (auto nameHint = structType->findDecoration<IRNameHintDecoration>())
    {
        return nameHint->getName() == toSlice("BuiltInTriangleIntersectionAttributes");
    }
    return false;
}

// Find RT entry points by looking for IREntryPointDecoration with RT stages.
// In library mode, Slang auto-discovers [shader("...")] functions and creates
// IREntryPointDecoration for each one.
static RTEntryPoints findRTEntryPoints(IRModule* module)
{
    RTEntryPoints result;

    for (auto inst : module->getGlobalInsts())
    {
        auto func = as<IRFunc>(inst);
        if (!func)
            continue;
        auto entryPointDecor = func->findDecoration<IREntryPointDecoration>();
        if (!entryPointDecor)
            continue;

        auto stage = entryPointDecor->getProfile().getStage();
        switch (stage)
        {
        default:
            break;
        case Stage::RayGeneration:
            result.raygen = func;
            break;
        case Stage::ClosestHit:
            result.closestHit = func;
            break;
        case Stage::Miss:
            result.miss = func;
            break;
        }
    }

    return result;
}

// Find the entryPointParams global associated with a shader function.
// Returns nullptr if not found.
static IRInst* findEntryPointParamsGlobal(IRModule* module, IRFunc* func)
{
    for (auto inst : module->getGlobalInsts())
    {
        if (auto decor = inst->findDecoration<IREntryPointParamDecoration>())
        {
            if (decor->getEntryPoint() == func)
                return inst;
        }
    }
    return nullptr;
}

// Inline a closesthit or miss function body into a target block.
// The cloned body replaces ray intrinsic calls with values from the TraceRay call site.
// `payloadPtr` is the inout payload argument from TraceRay.
// `isHitBranch` controls whether hit-specific intrinsics (InstanceIndex etc.) are available.
//
// After IR linking, closesthit/miss function bodies access their parameters through
// entryPointParams globals (not through block params). The function body contains patterns
// like:
//   addr = get_field_addr(entryPointParams, payload_field)
//   ptr = load(addr)  -> BorrowInOutParam(PayloadType) = pointer to payload
// We pre-map these load results in the clone env so the cloned code uses the raygen's
// local payload variable instead.
static void inlineShaderBody(
    IRFunc* shaderFunc,
    IRBlock* targetBlock,
    IRBlock* afterBlock,
    IRBuilder& builder,
    IRInst* intersectResult,
    IRInst* origin,
    IRInst* direction,
    IRInst* tMin,
    IRInst* tMax,
    IRInst* rayFlags,
    IRInst* payloadPtr,
    bool isHitBranch)
{
    if (!shaderFunc)
    {
        builder.setInsertInto(targetBlock);
        builder.emitBranch(afterBlock);
        return;
    }

    auto firstBlock = shaderFunc->getFirstBlock();
    if (!firstBlock)
    {
        builder.setInsertInto(targetBlock);
        builder.emitBranch(afterBlock);
        return;
    }

    builder.setInsertInto(targetBlock);

    IRCloneEnv env;
    auto module = shaderFunc->getModule();

    // Find the entryPointParams global for this shader function.
    // After IR linking, shader parameters are accessed through this global struct.
    auto epParamsGlobal = findEntryPointParamsGlobal(module, shaderFunc);

    if (epParamsGlobal)
    {
        // Pre-scan the function body for get_field_addr(entryPointParams, field) -> load
        // patterns and map the load results to the correct raygen-side values.
        for (auto block : shaderFunc->getBlocks())
        {
            for (auto inst = block->getFirstOrdinaryInst(); inst; inst = inst->getNextInst())
            {
                if (inst->getOp() != kIROp_FieldAddress)
                    continue;
                auto fieldAddr = as<IRFieldAddress>(inst);
                if (fieldAddr->getBase() != epParamsGlobal)
                    continue;

                // Check if the field type indicates payload or attrs.
                auto loadedType = cast<IRPtrTypeBase>(fieldAddr->getDataType())->getValueType();
                bool isPayloadField = (loadedType->getOp() == kIROp_BorrowInOutParamType);
                bool isAttrsField = isTriangleIntersectionAttrsType(loadedType);

                // Map the get_field_addr itself so cloneInst skips it.
                env.mapOldValToNew.add(fieldAddr, fieldAddr);

                // Find all loads of this field address and map them.
                for (auto use = fieldAddr->firstUse; use; use = use->nextUse)
                {
                    auto load = as<IRLoad>(use->getUser());
                    if (!load)
                        continue;

                    if (isPayloadField)
                    {
                        // BorrowInOutParam(PayloadType) is essentially a pointer to the
                        // payload. Map the load result to the raygen's local payload
                        // variable (which is also a pointer to PayloadType).
                        env.mapOldValToNew.add(load, payloadPtr);
                    }
                    else if (isAttrsField && isHitBranch)
                    {
                        // Construct BuiltInTriangleIntersectionAttributes from intersection
                        // barycentrics. Create a local variable, store the barycentrics
                        // field into it, and load the struct value.
                        auto barycentrics = builder.emitIntrinsicInst(
                            builder.getVectorType(builder.getBasicType(BaseType::Float), 2),
                            kIROp_MetalRTIntersectionGetBarycentrics,
                            1,
                            &intersectResult);
                        auto attrsType = loadedType;
                        auto attrsVar = builder.emitVar(attrsType);
                        // Get the barycentrics field key from the struct type.
                        auto attrsStructType = as<IRStructType>(attrsType);
                        if (attrsStructType)
                        {
                            for (auto field : attrsStructType->getFields())
                            {
                                auto fieldAddr2 = builder.emitFieldAddress(
                                    builder.getPtrType(field->getFieldType()),
                                    attrsVar,
                                    field->getKey());
                                builder.emitStore(fieldAddr2, barycentrics);
                            }
                        }
                        auto attrsVal = builder.emitLoad(attrsType, attrsVar);
                        env.mapOldValToNew.add(load, attrsVal);
                    }
                }
            }
        }
    }

    // Create additional blocks for multi-block functions.
    for (auto block : shaderFunc->getBlocks())
    {
        if (block != firstBlock)
        {
            auto newBlock = builder.createBlock();
            env.mapOldValToNew.add(block, newBlock);
            auto parentFunc = as<IRFunc>(targetBlock->getParent());
            parentFunc->addBlock(newBlock);
        }
    }

    // Clone instructions from all blocks.
    for (auto block : shaderFunc->getBlocks())
    {
        IRBlock* destBlock = nullptr;
        if (block == firstBlock)
        {
            destBlock = targetBlock;
        }
        else
        {
            destBlock = as<IRBlock>(findCloneForOperand(&env, block));
        }
        builder.setInsertInto(destBlock);

        for (auto inst = block->getFirstOrdinaryInst(); inst; inst = inst->getNextInst())
        {
            // Skip return instructions — branch to afterBlock instead.
            if (as<IRReturn>(inst))
            {
                builder.emitBranch(afterBlock);
                continue;
            }

            // Check if this is a ray intrinsic call that we need to replace.
            auto call = as<IRCall>(inst);
            if (call)
            {
                auto intrinsicKind = getMetalRTIntrinsicFromCall(call);
                IRInst* replacement = nullptr;

                switch (intrinsicKind)
                {
                default:
                    break;
                case MetalRTIntrinsic::WorldRayOrigin:
                    replacement = origin;
                    break;
                case MetalRTIntrinsic::WorldRayDirection:
                    replacement = direction;
                    break;
                case MetalRTIntrinsic::RayTMin:
                    replacement = tMin;
                    break;
                case MetalRTIntrinsic::RayTCurrent:
                    if (isHitBranch)
                    {
                        replacement = builder.emitIntrinsicInst(
                            builder.getBasicType(BaseType::Float),
                            kIROp_MetalRTIntersectionGetDistance,
                            1,
                            &intersectResult);
                    }
                    else
                    {
                        replacement = tMax;
                    }
                    break;
                case MetalRTIntrinsic::RayFlags:
                    replacement = rayFlags;
                    break;
                case MetalRTIntrinsic::InstanceIndex:
                    if (isHitBranch)
                    {
                        replacement = builder.emitIntrinsicInst(
                            builder.getBasicType(BaseType::UInt),
                            kIROp_MetalRTIntersectionGetInstanceId,
                            1,
                            &intersectResult);
                    }
                    break;
                case MetalRTIntrinsic::InstanceID:
                    if (isHitBranch)
                    {
                        replacement = builder.emitIntrinsicInst(
                            builder.getBasicType(BaseType::UInt),
                            kIROp_MetalRTIntersectionGetInstanceId,
                            1,
                            &intersectResult);
                    }
                    break;
                case MetalRTIntrinsic::PrimitiveIndex:
                    if (isHitBranch)
                    {
                        replacement = builder.emitIntrinsicInst(
                            builder.getBasicType(BaseType::UInt),
                            kIROp_MetalRTIntersectionGetPrimitiveId,
                            1,
                            &intersectResult);
                    }
                    break;
                case MetalRTIntrinsic::HitKind:
                    if (isHitBranch)
                    {
                        auto frontFace = builder.emitIntrinsicInst(
                            builder.getBoolType(),
                            kIROp_MetalRTIntersectionGetFrontFace,
                            1,
                            &intersectResult);
                        auto uintType = builder.getBasicType(BaseType::UInt);
                        auto frontVal = builder.getIntValue(uintType, 254);
                        auto backVal = builder.getIntValue(uintType, 255);
                        IRInst* selectArgs[] = {frontFace, frontVal, backVal};
                        replacement =
                            builder.emitIntrinsicInst(uintType, kIROp_Select, 3, selectArgs);
                    }
                    break;
                }

                if (replacement)
                {
                    env.mapOldValToNew[inst] = replacement;
                    continue;
                }
            }

            // cloneInst returns mapped value if already in env, otherwise clones.
            cloneInst(&env, &builder, inst);
        }
    }

    // If the last instruction in targetBlock is not a terminator, add a branch.
    auto lastInst = targetBlock->getLastInst();
    if (!lastInst || !as<IRTerminatorInst>(lastInst))
    {
        builder.setInsertInto(targetBlock);
        builder.emitBranch(afterBlock);
    }
}

static void legalizeTraceRayCalls(IRFunc* raygenFunc, const RTEntryPoints& entryPoints, IRBuilder& builder)
{
    // Find all TraceRay calls in the raygen function.
    List<IRCall*> traceRayCalls;
    for (auto block : raygenFunc->getBlocks())
    {
        for (auto inst = block->getFirstOrdinaryInst(); inst; inst = inst->getNextInst())
        {
            auto call = as<IRCall>(inst);
            if (!call)
                continue;
            if (getMetalRTIntrinsicFromCall(call) == MetalRTIntrinsic::TraceRay)
            {
                traceRayCalls.add(call);
            }
        }
    }

    auto uintType = builder.getBasicType(BaseType::UInt);

    for (auto call : traceRayCalls)
    {
        // __metalrt_TraceRay args:
        // 0: accel, 1: rayFlags, 2: instanceMask, 3: origin, 4: tMin, 5: direction, 6: tMax, 7: payload
        auto accel = call->getArg(0);
        auto rayFlagsArg = call->getArg(1);
        auto instanceMask = call->getArg(2);
        auto origin = call->getArg(3);
        auto tMin = call->getArg(4);
        auto direction = call->getArg(5);
        auto tMax = call->getArg(6);
        auto payloadPtr = call->getArg(7);

        // Insert before the TraceRay call.
        builder.setInsertBefore(call);

        // Emit MetalRTIntersect instruction.
        IRInst* intersectArgs[] = {accel, origin, direction, tMin, tMax, rayFlagsArg, instanceMask};
        auto intersectResult = builder.emitIntrinsicInst(
            uintType, // placeholder type — emitter will use `auto`
            kIROp_MetalRTIntersect,
            7,
            intersectArgs);

        // Emit type check: result type != 0 means we got a hit.
        auto intersectionType = builder.emitIntrinsicInst(
            uintType,
            kIROp_MetalRTIntersectionGetType,
            1,
            &intersectResult);
        auto zero = builder.getIntValue(uintType, 0);
        auto hitCondition = builder.emitNeq(intersectionType, zero);

        // Create hit/miss/after blocks.
        auto hitBlock = builder.createBlock();
        auto missBlock = builder.createBlock();
        auto afterBlock = builder.createBlock();
        raygenFunc->addBlock(hitBlock);
        raygenFunc->addBlock(missBlock);
        raygenFunc->addBlock(afterBlock);

        // Move all instructions after the TraceRay call to afterBlock.
        {
            List<IRInst*> instsToMove;
            for (auto inst = call->getNextInst(); inst; inst = inst->getNextInst())
            {
                instsToMove.add(inst);
            }
            for (auto inst : instsToMove)
            {
                inst->insertAtEnd(afterBlock);
            }
        }

        // Emit the if/else branch (replaces the TraceRay call as terminator).
        builder.emitIfElse(hitCondition, hitBlock, missBlock, afterBlock);

        // Remove the TraceRay call.
        call->removeAndDeallocate();

        // Inline closesthit body into hitBlock.
        inlineShaderBody(
            entryPoints.closestHit,
            hitBlock,
            afterBlock,
            builder,
            intersectResult,
            origin,
            direction,
            tMin,
            tMax,
            rayFlagsArg,
            payloadPtr,
            true);

        // Inline miss body into missBlock.
        inlineShaderBody(
            entryPoints.miss,
            missBlock,
            afterBlock,
            builder,
            intersectResult,
            origin,
            direction,
            tMin,
            tMax,
            rayFlagsArg,
            payloadPtr,
            false);
    }
}

static void removeNonRaygenEntryPoints(IRModule* module)
{
    List<IRInst*> instsToRemove;

    for (auto inst : module->getGlobalInsts())
    {
        auto func = as<IRFunc>(inst);
        if (!func)
            continue;
        auto entryPointDecor = func->findDecoration<IREntryPointDecoration>();
        if (!entryPointDecor)
            continue;

        auto stage = entryPointDecor->getProfile().getStage();
        if (stage == Stage::ClosestHit || stage == Stage::Miss)
        {
            instsToRemove.add(func);
        }
    }

    // Also remove entryPointParams globals associated with the removed functions.
    for (auto inst : module->getGlobalInsts())
    {
        if (auto decor = inst->findDecoration<IREntryPointParamDecoration>())
        {
            for (auto func : instsToRemove)
            {
                if (decor->getEntryPoint() == func)
                {
                    instsToRemove.add(inst);
                    break;
                }
            }
        }
    }

    for (auto inst : instsToRemove)
    {
        inst->removeAndDeallocate();
    }
}

void legalizeIRForMetalRT(IRModule* module, TargetProgram* targetProgram, DiagnosticSink* sink)
{
    SLANG_UNUSED(sink);

    int dispatchDimsBufferSlot =
        targetProgram->getOptionSet().getIntOption(CompilerOptionName::MetalRTDispatchDimsBufferSlot);

    IRBuilder builder(module);

    auto entryPoints = findRTEntryPoints(module);

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

        legalizeRaygenEntryPoint(func, builder, dispatchDimsBufferSlot);

        if (entryPoints.closestHit || entryPoints.miss)
        {
            legalizeTraceRayCalls(func, entryPoints, builder);
        }
    }

    removeNonRaygenEntryPoints(module);
}

} // namespace Slang
