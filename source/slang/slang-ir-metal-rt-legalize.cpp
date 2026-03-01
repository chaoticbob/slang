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
    IgnoreHit,
    AcceptHitAndEndSearch,
    ObjectToWorld3x4,
    ObjectToWorld4x3,
    WorldToObject3x4,
    WorldToObject4x3,
    ReportHit,
    ObjectRayOrigin,
    ObjectRayDirection,
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
        if (name == toSlice("IgnoreHit"))
            return MetalRTIntrinsic::IgnoreHit;
        if (name == toSlice("AcceptHitAndEndSearch"))
            return MetalRTIntrinsic::AcceptHitAndEndSearch;
        if (name == toSlice("ObjectToWorld3x4"))
            return MetalRTIntrinsic::ObjectToWorld3x4;
        if (name == toSlice("ObjectToWorld4x3"))
            return MetalRTIntrinsic::ObjectToWorld4x3;
        if (name == toSlice("WorldToObject3x4"))
            return MetalRTIntrinsic::WorldToObject3x4;
        if (name == toSlice("WorldToObject4x3"))
            return MetalRTIntrinsic::WorldToObject4x3;
        if (name == toSlice("ReportHit"))
            return MetalRTIntrinsic::ReportHit;
        if (name == toSlice("ObjectRayOrigin"))
            return MetalRTIntrinsic::ObjectRayOrigin;
        if (name == toSlice("ObjectRayDirection"))
            return MetalRTIntrinsic::ObjectRayDirection;
    }
    return MetalRTIntrinsic::None;
}

static IRInst* legalizeRaygenEntryPoint(IRFunc* func, IRBuilder& builder, int dispatchDimsBufferSlot)
{
    auto firstBlock = func->getFirstBlock();
    if (!firstBlock)
        return nullptr;

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

    // 3. Add intersection function table parameter at [[buffer(29)]].
    auto funcTableParam = builder.emitParam(uintType);
    builder.addNameHintDecoration(funcTableParam, toSlice("_metalrt_func_table"));
    builder.addTargetSystemValueDecoration(funcTableParam, toSlice("buffer(29)"));

    // 4. Replace intrinsic calls with the new parameters.
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

    // 5. Change stage from RayGeneration to Compute.
    if (auto entryPointDecor = func->findDecoration<IREntryPointDecoration>())
    {
        auto profile = entryPointDecor->getProfile();
        profile.setStage(Stage::Compute);
        entryPointDecor->setOperand(
            0,
            builder.getIntValue(builder.getIntType(), profile.raw));
    }

    // 6. Add [numthreads(8, 8, 1)] decoration.
    auto intType = builder.getIntType();
    builder.addNumThreadsDecoration(
        func,
        builder.getIntValue(intType, 8),
        builder.getIntValue(intType, 8),
        builder.getIntValue(intType, 1));

    // 7. Fix up the function type to match the new parameter list.
    fixUpFuncType(func);

    return funcTableParam;
}

struct RTEntryPoints
{
    IRFunc* raygen = nullptr;
    IRFunc* closestHit = nullptr;
    List<IRFunc*> missShaders;  // Ordered by appearance (corresponds to MissShaderIndex)
    IRFunc* anyHit = nullptr;
    IRFunc* intersection = nullptr;
    IRInst* raygenFuncTable = nullptr;
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
            if (!result.closestHit)
                result.closestHit = func;
            break;
        case Stage::Miss:
            result.missShaders.add(func);
            break;
        case Stage::AnyHit:
            if (!result.anyHit)
                result.anyHit = func;
            break;
        case Stage::Intersection:
            if (!result.intersection)
                result.intersection = func;
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
                case MetalRTIntrinsic::ObjectToWorld4x3:
                    if (isHitBranch)
                    {
                        replacement = builder.emitIntrinsicInst(
                            call->getDataType(),
                            kIROp_MetalRTIntersectionGetObjectToWorld4x3,
                            1,
                            &intersectResult);
                    }
                    break;
                case MetalRTIntrinsic::ObjectToWorld3x4:
                    if (isHitBranch)
                    {
                        replacement = builder.emitIntrinsicInst(
                            call->getDataType(),
                            kIROp_MetalRTIntersectionGetObjectToWorld3x4,
                            1,
                            &intersectResult);
                    }
                    break;
                case MetalRTIntrinsic::WorldToObject4x3:
                    if (isHitBranch)
                    {
                        replacement = builder.emitIntrinsicInst(
                            call->getDataType(),
                            kIROp_MetalRTIntersectionGetWorldToObject4x3,
                            1,
                            &intersectResult);
                    }
                    break;
                case MetalRTIntrinsic::WorldToObject3x4:
                    if (isHitBranch)
                    {
                        replacement = builder.emitIntrinsicInst(
                            call->getDataType(),
                            kIROp_MetalRTIntersectionGetWorldToObject3x4,
                            1,
                            &intersectResult);
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
        // 0: accel, 1: rayFlags, 2: instanceMask, 3: missIndex,
        // 4: origin, 5: tMin, 6: direction, 7: tMax, 8: payload
        auto accel = call->getArg(0);
        auto rayFlagsArg = call->getArg(1);
        auto instanceMask = call->getArg(2);
        auto origin = call->getArg(4);
        auto tMin = call->getArg(5);
        auto direction = call->getArg(6);
        auto tMax = call->getArg(7);
        auto payloadPtr = call->getArg(8);

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
            entryPoints.missShaders.getCount() > 0 ? entryPoints.missShaders[0] : nullptr,
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

// Phase 3: Transform a closesthit or miss function into a [[visible]] function
// with explicit parameters for payload and intersection data.
//
// Before: function accesses payload/attrs through entryPointParams global,
//         uses RT intrinsic calls for intersection data.
// After:  function takes device PayloadType* and intersection data as plain params.
static void legalizeVisibleFunction(
    IRFunc* func,
    IRBuilder& builder,
    IRModule* module,
    bool isClosestHit,
    bool isProcedural = false,
    IRType* proceduralAttrsType = nullptr)
{
    auto firstBlock = func->getFirstBlock();
    if (!firstBlock)
    {
        return;
    }

    auto epParamsGlobal = findEntryPointParamsGlobal(module, func);

    // Scan entryPointParams field accesses to determine payload/attrs types.
    IRType* payloadStructType = nullptr;
    IRType* attrsType = nullptr;

    struct LoadReplacement
    {
        IRInst* load;
        bool isPayload;
        bool isAttrs;
    };
    List<LoadReplacement> loadReplacements;
    List<IRInst*> fieldAddrsToRemove;
    List<IRInst*> loadsToRemove;

    if (epParamsGlobal)
    {
        for (auto block : func->getBlocks())
        {
            for (auto inst = block->getFirstOrdinaryInst(); inst; inst = inst->getNextInst())
            {
                if (inst->getOp() != kIROp_FieldAddress)
                {
                    continue;
                }
                auto fieldAddr = as<IRFieldAddress>(inst);
                if (fieldAddr->getBase() != epParamsGlobal)
                {
                    continue;
                }

                auto fieldPtrType = cast<IRPtrTypeBase>(fieldAddr->getDataType());
                auto fieldType = fieldPtrType->getValueType();
                bool isPayloadField = (fieldType->getOp() == kIROp_BorrowInOutParamType);
                // In procedural mode, custom attribute types (not just
                // BuiltInTriangleIntersectionAttributes) can appear as the attrs field.
                // Any non-payload field is treated as attrs.
                bool isAttrsField = isClosestHit
                    ? !isPayloadField
                    : isTriangleIntersectionAttrsType(fieldType);

                if (isPayloadField)
                {
                    payloadStructType = cast<IRPtrTypeBase>(fieldType)->getValueType();
                }
                if (isAttrsField)
                {
                    attrsType = fieldType;
                }

                fieldAddrsToRemove.add(fieldAddr);

                for (auto use = fieldAddr->firstUse; use; use = use->nextUse)
                {
                    auto load = as<IRLoad>(use->getUser());
                    if (!load)
                    {
                        continue;
                    }
                    LoadReplacement rep;
                    rep.load = load;
                    rep.isPayload = isPayloadField;
                    rep.isAttrs = isAttrsField;
                    loadReplacements.add(rep);
                    loadsToRemove.add(load);
                }
            }
        }
    }

    // Add new function parameters to the first block.
    builder.setInsertInto(firstBlock);

    // Payload parameter: device PayloadType*.
    IRInst* payloadParam = nullptr;
    if (payloadStructType)
    {
        auto payloadPtrType = builder.getPtrType(payloadStructType, AddressSpace::Global);
        payloadParam = builder.emitParam(payloadPtrType);
        builder.addNameHintDecoration(payloadParam, toSlice("payload"));
    }

    // Ray parameters (both closesthit and miss).
    auto float3Type = builder.getVectorType(builder.getBasicType(BaseType::Float), 3);
    auto floatType = builder.getBasicType(BaseType::Float);
    auto uintType = builder.getBasicType(BaseType::UInt);

    auto worldRayOriginParam = builder.emitParam(float3Type);
    builder.addNameHintDecoration(worldRayOriginParam, toSlice("worldRayOrigin"));

    auto worldRayDirectionParam = builder.emitParam(float3Type);
    builder.addNameHintDecoration(worldRayDirectionParam, toSlice("worldRayDirection"));

    auto rayTMinParam = builder.emitParam(floatType);
    builder.addNameHintDecoration(rayTMinParam, toSlice("rayTMin"));

    auto rayFlagsParam = builder.emitParam(uintType);
    builder.addNameHintDecoration(rayFlagsParam, toSlice("rayFlags"));

    // Attrs parameter for procedural closesthit (set below if applicable).
    IRInst* attrsParam = nullptr;

    // Intersection data parameters (closesthit only).
    IRInst* baryParam = nullptr;
    IRInst* distParam = nullptr;
    IRInst* primIdParam = nullptr;
    IRInst* instIdParam = nullptr;
    IRInst* frontFacingParam = nullptr;
    IRInst* objectToWorld4x3Param = nullptr;
    IRInst* objectToWorld3x4Param = nullptr;
    IRInst* worldToObject4x3Param = nullptr;
    IRInst* worldToObject3x4Param = nullptr;

    if (isClosestHit)
    {
        // For triangle mode, closestHit receives barycentrics and frontFacing.
        // For procedural mode, these are not available (no triangle intersection data).
        if (!isProcedural)
        {
            auto float2Type = builder.getVectorType(builder.getBasicType(BaseType::Float), 2);

            baryParam = builder.emitParam(float2Type);
            builder.addNameHintDecoration(baryParam, toSlice("barycentrics"));
        }

        distParam = builder.emitParam(floatType);
        builder.addNameHintDecoration(distParam, toSlice("distance"));

        primIdParam = builder.emitParam(uintType);
        builder.addNameHintDecoration(primIdParam, toSlice("primitiveId"));

        instIdParam = builder.emitParam(uintType);
        builder.addNameHintDecoration(instIdParam, toSlice("instanceId"));

        if (!isProcedural)
        {
            auto boolType = builder.getBoolType();

            frontFacingParam = builder.emitParam(boolType);
            builder.addNameHintDecoration(frontFacingParam, toSlice("frontFacing"));
        }

        // Transform matrix parameters.
        auto intType = builder.getIntType();
        auto intVal4 = builder.getIntValue(intType, 4);
        auto intVal3 = builder.getIntValue(intType, 3);
        auto noLayout = builder.getIntValue(intType, 0);
        auto float4x3Type = builder.getMatrixType(floatType, intVal4, intVal3, noLayout);
        auto float3x4Type = builder.getMatrixType(floatType, intVal3, intVal4, noLayout);

        objectToWorld4x3Param = builder.emitParam(float4x3Type);
        builder.addNameHintDecoration(objectToWorld4x3Param, toSlice("objectToWorld4x3"));

        objectToWorld3x4Param = builder.emitParam(float3x4Type);
        builder.addNameHintDecoration(objectToWorld3x4Param, toSlice("objectToWorld3x4"));

        worldToObject4x3Param = builder.emitParam(float4x3Type);
        builder.addNameHintDecoration(worldToObject4x3Param, toSlice("worldToObject4x3"));

        worldToObject3x4Param = builder.emitParam(float3x4Type);
        builder.addNameHintDecoration(worldToObject3x4Param, toSlice("worldToObject3x4"));

        // Intersection function table parameter (passed from raygen).
        auto funcTableParam = builder.emitParam(uintType);
        builder.addNameHintDecoration(funcTableParam, toSlice("_metalrt_func_table"));

        // For procedural mode, add attrs as the last parameter (passed from raygen
        // after the intersection function writes to it via ray_data payload).
        if (isProcedural && proceduralAttrsType)
        {
            attrsParam = builder.emitParam(proceduralAttrsType);
            builder.addNameHintDecoration(attrsParam, toSlice("attrs"));
        }
    }

    // Replace entryPointParams load results with new parameter values.
    for (auto& rep : loadReplacements)
    {
        if (rep.isPayload && payloadParam)
        {
            rep.load->replaceUsesWith(payloadParam);
        }
        else if (rep.isAttrs && isClosestHit && baryParam)
        {
            // Triangle mode: construct BuiltInTriangleIntersectionAttributes from barycentrics.
            auto insertPoint = firstBlock->getFirstOrdinaryInst();
            if (insertPoint)
            {
                builder.setInsertBefore(insertPoint);
            }
            else
            {
                builder.setInsertInto(firstBlock);
            }

            auto attrsVar = builder.emitVar(attrsType);
            auto attrsStructType = as<IRStructType>(attrsType);
            if (attrsStructType)
            {
                for (auto field : attrsStructType->getFields())
                {
                    auto fieldAddr2 = builder.emitFieldAddress(
                        builder.getPtrType(field->getFieldType()),
                        attrsVar,
                        field->getKey());
                    builder.emitStore(fieldAddr2, baryParam);
                }
            }
            auto attrsVal = builder.emitLoad(attrsType, attrsVar);
            rep.load->replaceUsesWith(attrsVal);
        }
        else if (rep.isAttrs && isClosestHit && isProcedural && attrsParam)
        {
            // Procedural mode: custom hit attributes are passed as a function parameter
            // from raygen, which received them from the intersection function via ray_data payload.
            rep.load->replaceUsesWith(attrsParam);
        }
    }

    // Replace RT intrinsic calls with new function parameters.
    List<IRCall*> intrinsicCallsToRemove;
    for (auto block : func->getBlocks())
    {
        for (auto inst = block->getFirstOrdinaryInst(); inst; inst = inst->getNextInst())
        {
            auto call = as<IRCall>(inst);
            if (!call)
            {
                continue;
            }

            auto intrinsicKind = getMetalRTIntrinsicFromCall(call);
            if (intrinsicKind == MetalRTIntrinsic::None)
            {
                continue;
            }

            IRInst* replacement = nullptr;
            switch (intrinsicKind)
            {
            default:
                {
                }
                break;
            case MetalRTIntrinsic::TraceRay:
                {
                    // TraceRay calls inside visible functions are handled later by
                    // legalizeTraceRayCallsInVisibleFunctions. Skip without removing.
                    continue;
                }
            case MetalRTIntrinsic::WorldRayOrigin:
                {
                    replacement = worldRayOriginParam;
                }
                break;
            case MetalRTIntrinsic::WorldRayDirection:
                {
                    replacement = worldRayDirectionParam;
                }
                break;
            case MetalRTIntrinsic::RayTMin:
                {
                    replacement = rayTMinParam;
                }
                break;
            case MetalRTIntrinsic::RayFlags:
                {
                    replacement = rayFlagsParam;
                }
                break;
            case MetalRTIntrinsic::RayTCurrent:
                {
                    if (isClosestHit && distParam)
                    {
                        replacement = distParam;
                    }
                }
                break;
            case MetalRTIntrinsic::PrimitiveIndex:
                {
                    if (isClosestHit && primIdParam)
                    {
                        replacement = primIdParam;
                    }
                }
                break;
            case MetalRTIntrinsic::InstanceIndex:
            case MetalRTIntrinsic::InstanceID:
                {
                    if (isClosestHit && instIdParam)
                    {
                        replacement = instIdParam;
                    }
                }
                break;
            case MetalRTIntrinsic::HitKind:
                {
                    if (isClosestHit && frontFacingParam)
                    {
                        builder.setInsertBefore(call);
                        auto uintType = builder.getBasicType(BaseType::UInt);
                        auto frontVal = builder.getIntValue(uintType, 254);
                        auto backVal = builder.getIntValue(uintType, 255);
                        IRInst* selectArgs[] = {frontFacingParam, frontVal, backVal};
                        replacement =
                            builder.emitIntrinsicInst(uintType, kIROp_Select, 3, selectArgs);
                    }
                }
                break;
            case MetalRTIntrinsic::ObjectToWorld4x3:
                {
                    if (isClosestHit && objectToWorld4x3Param)
                    {
                        replacement = objectToWorld4x3Param;
                    }
                }
                break;
            case MetalRTIntrinsic::ObjectToWorld3x4:
                {
                    if (isClosestHit && objectToWorld3x4Param)
                    {
                        replacement = objectToWorld3x4Param;
                    }
                }
                break;
            case MetalRTIntrinsic::WorldToObject4x3:
                {
                    if (isClosestHit && worldToObject4x3Param)
                    {
                        replacement = worldToObject4x3Param;
                    }
                }
                break;
            case MetalRTIntrinsic::WorldToObject3x4:
                {
                    if (isClosestHit && worldToObject3x4Param)
                    {
                        replacement = worldToObject3x4Param;
                    }
                }
                break;
            }

            if (replacement)
            {
                call->replaceUsesWith(replacement);
                intrinsicCallsToRemove.add(call);
            }
        }
    }

    // Remove replaced instructions.
    for (auto call : intrinsicCallsToRemove)
    {
        call->removeAndDeallocate();
    }
    for (auto load : loadsToRemove)
    {
        load->removeAndDeallocate();
    }
    for (auto fieldAddr : fieldAddrsToRemove)
    {
        fieldAddr->removeAndDeallocate();
    }

    // Remove the entryPointParams global for this function.
    if (epParamsGlobal)
    {
        epParamsGlobal->removeAndDeallocate();
    }

    // Fix up the function type to match the new parameter list.
    fixUpFuncType(func);
}

// Phase 4: Transform an AnyHit function into a Metal [[intersection(triangle, instancing)]]
// function with bool return type.
//
// Before: function takes payload (inout) and attrs through entryPointParams global,
//         uses IgnoreHit()/AcceptHitAndEndSearch() intrinsics, returns void.
// After:  function takes barycentrics/primitiveId/instanceId as system-value params,
//         returns bool (false = ignore hit, true = accept hit).
//         No payload parameter (Metal intersection functions can't access payload).
static void legalizeAnyHitFunction(
    IRFunc* func,
    IRBuilder& builder,
    IRModule* module)
{
    auto firstBlock = func->getFirstBlock();
    if (!firstBlock)
    {
        return;
    }

    auto epParamsGlobal = findEntryPointParamsGlobal(module, func);

    // Scan entryPointParams field accesses to find payload and attrs fields.
    IRType* attrsType = nullptr;

    struct LoadReplacement
    {
        IRInst* load;
        bool isPayload;
        bool isAttrs;
    };
    List<LoadReplacement> loadReplacements;
    List<IRInst*> fieldAddrsToRemove;
    List<IRInst*> loadsToRemove;

    if (epParamsGlobal)
    {
        for (auto block : func->getBlocks())
        {
            for (auto inst = block->getFirstOrdinaryInst(); inst; inst = inst->getNextInst())
            {
                if (inst->getOp() != kIROp_FieldAddress)
                {
                    continue;
                }
                auto fieldAddr = as<IRFieldAddress>(inst);
                if (fieldAddr->getBase() != epParamsGlobal)
                {
                    continue;
                }

                auto fieldPtrType = cast<IRPtrTypeBase>(fieldAddr->getDataType());
                auto fieldType = fieldPtrType->getValueType();
                bool isPayloadField = (fieldType->getOp() == kIROp_BorrowInOutParamType);
                bool isAttrsField = isTriangleIntersectionAttrsType(fieldType);

                if (isAttrsField)
                {
                    attrsType = fieldType;
                }

                fieldAddrsToRemove.add(fieldAddr);

                for (auto use = fieldAddr->firstUse; use; use = use->nextUse)
                {
                    auto load = as<IRLoad>(use->getUser());
                    if (!load)
                    {
                        continue;
                    }
                    LoadReplacement rep;
                    rep.load = load;
                    rep.isPayload = isPayloadField;
                    rep.isAttrs = isAttrsField;
                    loadReplacements.add(rep);
                    loadsToRemove.add(load);
                }
            }
        }
    }

    // Add new function parameters to the first block.
    builder.setInsertInto(firstBlock);

    auto float2Type = builder.getVectorType(builder.getBasicType(BaseType::Float), 2);
    auto uintType = builder.getBasicType(BaseType::UInt);

    // Barycentrics parameter with [[barycentric_coord]].
    auto baryParam = builder.emitParam(float2Type);
    builder.addNameHintDecoration(baryParam, toSlice("barycentrics"));
    builder.addTargetSystemValueDecoration(baryParam, toSlice("barycentric_coord"));

    // PrimitiveId parameter with [[primitive_id]].
    auto primIdParam = builder.emitParam(uintType);
    builder.addNameHintDecoration(primIdParam, toSlice("primitiveId"));
    builder.addTargetSystemValueDecoration(primIdParam, toSlice("primitive_id"));

    // InstanceId parameter with [[instance_id]].
    auto instIdParam = builder.emitParam(uintType);
    builder.addNameHintDecoration(instIdParam, toSlice("instanceId"));
    builder.addTargetSystemValueDecoration(instIdParam, toSlice("instance_id"));

    // Replace entryPointParams load results.
    for (auto& rep : loadReplacements)
    {
        if (rep.isPayload)
        {
            // Create ray_data payload parameter with [[payload]] attribute.
            // The payload type is extracted from BorrowInOutParam -> underlying struct type.
            // In MSL, the payload param is a reference (ray_data T&), not a pointer.
            // The existing IR code accesses payload through pointer ops (get_field_addr -> store).
            // To bridge the reference/pointer mismatch:
            //   1. Create the ray_data param (emitted as ray_data T& [[payload]])
            //   2. Allocate a thread-local copy and load the param into it
            //   3. Replace rep.load with &localCopy (thread T* — pointer ops work)
            //   4. At every return, write localCopy back to the ray_data param
            auto payloadPtrType = cast<IRPtrTypeBase>(rep.load->getDataType());
            auto payloadStructType = payloadPtrType->getValueType();
            auto rayDataPtrType = builder.getPtrType(payloadStructType, AddressSpace::MetalRayData);
            auto payloadParam = builder.emitParam(rayDataPtrType);
            builder.addNameHintDecoration(payloadParam, toSlice("_metalrt_payload"));
            builder.addTargetSystemValueDecoration(payloadParam, toSlice("payload"));

            // Allocate thread-local copy and load from ray_data param.
            auto insertPoint = firstBlock->getFirstOrdinaryInst();
            if (insertPoint)
                builder.setInsertBefore(insertPoint);
            else
                builder.setInsertInto(firstBlock);

            auto localVar = builder.emitVar(payloadStructType);
            auto payloadVal = builder.emitLoad(payloadStructType, payloadParam);
            builder.emitStore(localVar, payloadVal);

            // Replace payload pointer uses with address of local copy.
            rep.load->replaceUsesWith(localVar);

            // Before every return, write the local copy back to the ray_data param.
            for (auto block : func->getBlocks())
            {
                for (auto inst = block->getFirstOrdinaryInst(); inst; inst = inst->getNextInst())
                {
                    if (as<IRReturn>(inst))
                    {
                        builder.setInsertBefore(inst);
                        auto updatedVal = builder.emitLoad(payloadStructType, localVar);
                        builder.emitStore(payloadParam, updatedVal);
                    }
                }
            }
        }
        else if (rep.isAttrs && baryParam)
        {
            // Construct BuiltInTriangleIntersectionAttributes struct from barycentrics param.
            auto insertPoint = firstBlock->getFirstOrdinaryInst();
            if (insertPoint)
            {
                builder.setInsertBefore(insertPoint);
            }
            else
            {
                builder.setInsertInto(firstBlock);
            }

            auto attrsVar = builder.emitVar(attrsType);
            auto attrsStructType = as<IRStructType>(attrsType);
            if (attrsStructType)
            {
                for (auto field : attrsStructType->getFields())
                {
                    auto fieldAddr2 = builder.emitFieldAddress(
                        builder.getPtrType(field->getFieldType()),
                        attrsVar,
                        field->getKey());
                    builder.emitStore(fieldAddr2, baryParam);
                }
            }
            auto attrsVal = builder.emitLoad(attrsType, attrsVar);
            rep.load->replaceUsesWith(attrsVal);
        }
    }

    // Replace RT intrinsic calls and IgnoreHit/AcceptHitAndEndSearch.
    auto boolType = builder.getBoolType();
    auto boolTrue = builder.getBoolValue(true);
    auto boolFalse = builder.getBoolValue(false);

    List<IRInst*> instsToRemove;
    for (auto block : func->getBlocks())
    {
        for (auto inst = block->getFirstOrdinaryInst(); inst; inst = inst->getNextInst())
        {
            // Replace void returns with `return true` (accept hit by default).
            if (auto retInst = as<IRReturn>(inst))
            {
                builder.setInsertBefore(retInst);
                builder.emitReturn(boolTrue);
                instsToRemove.add(retInst);
                continue;
            }

            auto call = as<IRCall>(inst);
            if (!call)
            {
                continue;
            }

            auto intrinsicKind = getMetalRTIntrinsicFromCall(call);
            if (intrinsicKind == MetalRTIntrinsic::None)
            {
                continue;
            }

            IRInst* replacement = nullptr;
            switch (intrinsicKind)
            {
            default:
                {
                }
                break;
            case MetalRTIntrinsic::IgnoreHit:
                {
                    // IgnoreHit() -> return false
                    builder.setInsertBefore(call);
                    builder.emitReturn(boolFalse);
                    instsToRemove.add(call);
                }
                break;
            case MetalRTIntrinsic::AcceptHitAndEndSearch:
                {
                    // AcceptHitAndEndSearch() -> return true
                    builder.setInsertBefore(call);
                    builder.emitReturn(boolTrue);
                    instsToRemove.add(call);
                }
                break;
            case MetalRTIntrinsic::PrimitiveIndex:
                {
                    replacement = primIdParam;
                }
                break;
            case MetalRTIntrinsic::InstanceIndex:
            case MetalRTIntrinsic::InstanceID:
                {
                    replacement = instIdParam;
                }
                break;
            case MetalRTIntrinsic::HitKind:
                {
                    // No frontFacing param available — use constant for front face.
                    replacement = builder.getIntValue(uintType, 254);
                }
                break;
            }

            if (replacement)
            {
                call->replaceUsesWith(replacement);
                instsToRemove.add(call);
            }
        }
    }

    // Remove replaced instructions.
    for (auto inst : instsToRemove)
    {
        inst->removeAndDeallocate();
    }
    for (auto load : loadsToRemove)
    {
        load->removeAndDeallocate();
    }
    for (auto fieldAddr : fieldAddrsToRemove)
    {
        fieldAddr->removeAndDeallocate();
    }

    // Remove the entryPointParams global for this function.
    if (epParamsGlobal)
    {
        epParamsGlobal->removeAndDeallocate();
    }

    // Change the function return type to bool.
    auto funcType = as<IRFuncType>(func->getDataType());
    if (funcType)
    {
        List<IRType*> paramTypes;
        for (UInt i = 0; i < funcType->getParamCount(); i++)
        {
            paramTypes.add(funcType->getParamType(i));
        }
        auto newFuncType = builder.getFuncType(paramTypes, boolType);
        func->setFullType(newFuncType);
    }

    // Fix up the function type to match the new parameter list.
    fixUpFuncType(func);
}

// Phase 5: Transform an Intersection function into a Metal
// [[intersection(bounding_box, instancing, world_space_data)]] function with bool return type.
//
// Before: function accesses payload and attrs through entryPointParams global,
//         uses ObjectRayOrigin()/ObjectRayDirection()/RayTMin()/RayTCurrent() intrinsics,
//         calls ReportHit(t, kind, attrs) to report intersections.
// After:  function takes origin/direction/min_distance/max_distance/primitive_id/instance_id
//         as system-value params, has a float& distance output param,
//         returns bool (true = hit reported, false = no hit).
static void legalizeIntersectionFunction(
    IRFunc* func,
    IRBuilder& builder,
    IRModule* module)
{
    auto firstBlock = func->getFirstBlock();
    if (!firstBlock)
    {
        return;
    }

    auto epParamsGlobal = findEntryPointParamsGlobal(module, func);

    // Scan entryPointParams field accesses to find payload fields.
    struct LoadReplacement
    {
        IRInst* load;
        bool isPayload;
    };
    List<LoadReplacement> loadReplacements;
    List<IRInst*> fieldAddrsToRemove;
    List<IRInst*> loadsToRemove;

    if (epParamsGlobal)
    {
        for (auto block : func->getBlocks())
        {
            for (auto inst = block->getFirstOrdinaryInst(); inst; inst = inst->getNextInst())
            {
                if (inst->getOp() != kIROp_FieldAddress)
                {
                    continue;
                }
                auto fieldAddr = as<IRFieldAddress>(inst);
                if (fieldAddr->getBase() != epParamsGlobal)
                {
                    continue;
                }

                auto fieldPtrType = cast<IRPtrTypeBase>(fieldAddr->getDataType());
                auto fieldType = fieldPtrType->getValueType();
                bool isPayloadField = (fieldType->getOp() == kIROp_BorrowInOutParamType);

                fieldAddrsToRemove.add(fieldAddr);

                for (auto use = fieldAddr->firstUse; use; use = use->nextUse)
                {
                    auto load = as<IRLoad>(use->getUser());
                    if (!load)
                    {
                        continue;
                    }
                    LoadReplacement rep;
                    rep.load = load;
                    rep.isPayload = isPayloadField;
                    loadReplacements.add(rep);
                    loadsToRemove.add(load);
                }
            }
        }
    }

    // Add new function parameters to the first block.
    builder.setInsertInto(firstBlock);

    auto float3Type = builder.getVectorType(builder.getBasicType(BaseType::Float), 3);
    auto floatType = builder.getBasicType(BaseType::Float);
    auto uintType = builder.getBasicType(BaseType::UInt);
    auto boolType = builder.getBoolType();

    // Create the return struct: { bool accept [[accept_intersection]]; float distance [[distance]]; }
    builder.setInsertBefore(func);
    auto resultStructType = builder.createStructType();
    builder.addNameHintDecoration(resultStructType, toSlice("BoundingBoxResult"));

    auto acceptKey = builder.createStructKey();
    builder.addNameHintDecoration(acceptKey, toSlice("accept"));
    builder.addTargetSystemValueDecoration(acceptKey, toSlice("accept_intersection"));
    builder.createStructField(resultStructType, acceptKey, boolType);

    auto distanceKey = builder.createStructKey();
    builder.addNameHintDecoration(distanceKey, toSlice("distance"));
    builder.addTargetSystemValueDecoration(distanceKey, toSlice("distance"));
    builder.createStructField(resultStructType, distanceKey, floatType);

    // Origin parameter with [[origin]].
    builder.setInsertInto(firstBlock);
    auto originParam = builder.emitParam(float3Type);
    builder.addNameHintDecoration(originParam, toSlice("origin"));
    builder.addTargetSystemValueDecoration(originParam, toSlice("origin"));

    // Direction parameter with [[direction]].
    auto directionParam = builder.emitParam(float3Type);
    builder.addNameHintDecoration(directionParam, toSlice("direction"));
    builder.addTargetSystemValueDecoration(directionParam, toSlice("direction"));

    // Min distance parameter with [[min_distance]].
    auto minDistParam = builder.emitParam(floatType);
    builder.addNameHintDecoration(minDistParam, toSlice("min_dist"));
    builder.addTargetSystemValueDecoration(minDistParam, toSlice("min_distance"));

    // Max distance parameter with [[max_distance]].
    auto maxDistParam = builder.emitParam(floatType);
    builder.addNameHintDecoration(maxDistParam, toSlice("max_distance"));
    builder.addTargetSystemValueDecoration(maxDistParam, toSlice("max_distance"));

    // PrimitiveId parameter with [[primitive_id]].
    auto primIdParam = builder.emitParam(uintType);
    builder.addNameHintDecoration(primIdParam, toSlice("primitiveId"));
    builder.addTargetSystemValueDecoration(primIdParam, toSlice("primitive_id"));

    // InstanceId parameter with [[instance_id]].
    auto instIdParam = builder.emitParam(uintType);
    builder.addNameHintDecoration(instIdParam, toSlice("instanceId"));
    builder.addTargetSystemValueDecoration(instIdParam, toSlice("instance_id"));

    // Scan for ReportHit calls to determine attrs type, then add ray_data payload param.
    IRType* attrsType = nullptr;
    for (auto block : func->getBlocks())
    {
        for (auto inst = block->getFirstOrdinaryInst(); inst; inst = inst->getNextInst())
        {
            auto call = as<IRCall>(inst);
            if (!call)
                continue;
            if (getMetalRTIntrinsicFromCall(call) == MetalRTIntrinsic::ReportHit)
            {
                attrsType = call->getArg(2)->getDataType();
                break;
            }
        }
        if (attrsType)
            break;
    }

    // Add ray_data AttrsType& [[payload]] parameter for passing custom hit attributes
    // back to the caller via Metal's payload mechanism.
    IRInst* attrsPayloadParam = nullptr;
    if (attrsType)
    {
        auto rayDataPtrType = builder.getPtrType(attrsType, AddressSpace::MetalRayData);
        attrsPayloadParam = builder.emitParam(rayDataPtrType);
        builder.addNameHintDecoration(attrsPayloadParam, toSlice("_metalrt_payload"));
        builder.addTargetSystemValueDecoration(attrsPayloadParam, toSlice("payload"));
    }

    // Replace resource loads from epParamsGlobal with direct function parameters.
    // Metal intersection functions can only receive device/constant [[buffer(N)]] parameters,
    // not KernelContext. For each non-payload field that is a buffer type, add a parameter.
    // Payload fields are dropped (not accessible from intersection functions on Metal).
    for (auto& rep : loadReplacements)
    {
        if (rep.isPayload)
            continue;

        auto loadInst = rep.load;
        auto resourceType = loadInst->getDataType();

        // Only add parameters for pointer/buffer types Metal allows in intersection functions.
        auto ptrType = as<IRPtrTypeBase>(resourceType);
        if (!ptrType)
            continue;

        builder.setInsertInto(firstBlock);
        auto newParam = builder.emitParam(resourceType);

        // Copy name and layout decorations from the field key.
        auto fieldAddr = as<IRFieldAddress>(as<IRLoad>(loadInst)->getPtr());
        if (fieldAddr)
        {
            auto fieldKey = fieldAddr->getField();
            if (auto nameHint = fieldKey->findDecoration<IRNameHintDecoration>())
                builder.addNameHintDecoration(newParam, nameHint->getName());
            if (auto layoutDecor = fieldKey->findDecoration<IRLayoutDecoration>())
                builder.addLayoutDecoration(newParam, layoutDecor->getLayout());
        }

        loadInst->replaceUsesWith(newParam);
    }

    // Replace ALL references to IRGlobalParam/IRGlobalVar within the function body
    // with function parameters. This prevents the explicit global context pass from
    // adding a KernelContext parameter (which Metal intersection functions cannot receive).
    // Handles direct operand references like structuredBufferLoad(%spheres, ...).
    //
    // We iterate over the use chain of each global param in the module and check if
    // any use site is inside this function.
    {
        Dictionary<IRInst*, IRParam*> globalMap;

        // Collect all global params from the module.
        List<IRGlobalParam*> allGlobalParams;
        for (auto inst : module->getGlobalInsts())
        {
            if (auto gp = as<IRGlobalParam>(inst))
                allGlobalParams.add(gp);
        }

        // For each global param, check if it has uses in this function.
        for (auto globalParam : allGlobalParams)
        {
            bool usedInFunc = false;
            for (auto use = globalParam->firstUse; use; use = use->nextUse)
            {
                auto user = use->getUser();
                // Walk up the parent chain to see if this use is inside our function.
                for (IRInst* parent = user; parent; parent = parent->getParent())
                {
                    if (parent == func)
                    {
                        usedInFunc = true;
                        break;
                    }
                }
                if (usedInFunc)
                    break;
            }

            if (!usedInFunc)
                continue;

            // Add a function parameter with the same type.
            builder.setInsertInto(firstBlock);
            auto newParam = builder.emitParam(globalParam->getFullType());
            if (auto nameHint = globalParam->findDecoration<IRNameHintDecoration>())
                builder.addNameHintDecoration(newParam, nameHint->getName());
            if (auto layoutDecor = globalParam->findDecoration<IRLayoutDecoration>())
                builder.addLayoutDecoration(newParam, layoutDecor->getLayout());

            globalMap.add(globalParam, newParam);
        }

        // Replace uses within this function.
        for (auto& pair : globalMap)
        {
            auto globalParam = pair.first;
            auto newParam = pair.second;

            // Iterate the use chain and replace uses inside this function.
            IRUse* nextUse = nullptr;
            for (auto use = globalParam->firstUse; use; use = nextUse)
            {
                nextUse = use->nextUse;
                auto user = use->getUser();

                // Check if this use is inside our function.
                bool inFunc = false;
                for (IRInst* parent = user; parent; parent = parent->getParent())
                {
                    if (parent == func)
                    {
                        inFunc = true;
                        break;
                    }
                }
                if (inFunc)
                {
                    use->set(newParam);
                }
            }
        }
    }

    // Helper: build a return value of { accept, distance }.
    auto boolTrue = builder.getBoolValue(true);
    auto boolFalse = builder.getBoolValue(false);
    auto floatZero = builder.getFloatValue(floatType, 0.0);

    // Replace RT intrinsic calls and ReportHit.
    List<IRInst*> instsToRemove;
    List<IRCall*> reportHitCalls;
    for (auto block : func->getBlocks())
    {
        for (auto inst = block->getFirstOrdinaryInst(); inst; inst = inst->getNextInst())
        {
            // Replace void returns with `return {false, 0.0f}` (no hit reported).
            if (auto retInst = as<IRReturn>(inst))
            {
                builder.setInsertBefore(retInst);
                IRInst* rejectArgs[] = {boolFalse, floatZero};
                auto rejectVal = builder.emitMakeStruct(resultStructType, 2, rejectArgs);
                builder.emitReturn(rejectVal);
                instsToRemove.add(retInst);
                continue;
            }

            auto call = as<IRCall>(inst);
            if (!call)
            {
                continue;
            }

            auto intrinsicKind = getMetalRTIntrinsicFromCall(call);
            if (intrinsicKind == MetalRTIntrinsic::None)
            {
                continue;
            }

            IRInst* replacement = nullptr;
            switch (intrinsicKind)
            {
            default:
                {
                }
                break;
            case MetalRTIntrinsic::ReportHit:
                {
                    // Defer ReportHit processing to a second pass below.
                    // Block structure changes (splitting) can't safely happen
                    // while iterating the instruction list.
                    reportHitCalls.add(call);
                }
                break;
            case MetalRTIntrinsic::ObjectRayOrigin:
                {
                    replacement = originParam;
                }
                break;
            case MetalRTIntrinsic::ObjectRayDirection:
                {
                    replacement = directionParam;
                }
                break;
            case MetalRTIntrinsic::RayTMin:
                {
                    replacement = minDistParam;
                }
                break;
            case MetalRTIntrinsic::RayTCurrent:
                {
                    replacement = maxDistParam;
                }
                break;
            case MetalRTIntrinsic::IgnoreHit:
                {
                    // IgnoreHit() -> return {false, 0.0f}
                    builder.setInsertBefore(call);
                    IRInst* rejectArgs[] = {boolFalse, floatZero};
                    auto rejectVal = builder.emitMakeStruct(resultStructType, 2, rejectArgs);
                    builder.emitReturn(rejectVal);
                    instsToRemove.add(call);
                }
                break;
            case MetalRTIntrinsic::AcceptHitAndEndSearch:
                {
                    // AcceptHitAndEndSearch() -> return {true, 0.0f}
                    builder.setInsertBefore(call);
                    IRInst* acceptArgs[] = {boolTrue, floatZero};
                    auto acceptVal = builder.emitMakeStruct(resultStructType, 2, acceptArgs);
                    builder.emitReturn(acceptVal);
                    instsToRemove.add(call);
                }
                break;
            case MetalRTIntrinsic::PrimitiveIndex:
                {
                    replacement = primIdParam;
                }
                break;
            case MetalRTIntrinsic::InstanceIndex:
            case MetalRTIntrinsic::InstanceID:
                {
                    replacement = instIdParam;
                }
                break;
            }

            if (replacement)
            {
                call->replaceUsesWith(replacement);
                instsToRemove.add(call);
            }
        }
    }

    // Process ReportHit calls: add distance range check per Metal requirements.
    //
    // Metal's intersection function must return accept=true ONLY when the
    // distance is within [min_distance, max_distance]. Returning a distance
    // outside this range is undefined behavior (Apple WWDC20 "Discover Ray
    // Tracing with Metal"). DXR's ReportHit validates this internally.
    //
    // We emit:
    //   if (t >= minDist && t <= maxDist) { store(payload, attrs); return {true, t}; }
    // If the check fails, execution falls through to the next ReportHit or
    // the default return {false, 0.0f}, correctly modeling DXR's semantics
    // where ReportHit does not terminate the intersection function.
    for (auto rhCall : reportHitCalls)
    {
        auto tHit = rhCall->getArg(0);
        auto attrs = rhCall->getArg(2);

        builder.setInsertBefore(rhCall);

        // Range check: tHit >= minDistParam && tHit <= maxDistParam
        auto geqMinDist = builder.emitGeq(tHit, minDistParam);
        auto leqMaxDist = builder.emitGeq(maxDistParam, tHit);
        auto inRange = builder.emitAnd(boolType, geqMinDist, leqMaxDist);

        // Split the block: create accept and continue blocks.
        auto acceptBlock = builder.createBlock();
        auto continueBlock = builder.createBlock();
        func->addBlock(acceptBlock);
        func->addBlock(continueBlock);

        // Move all instructions after the ReportHit call to continueBlock.
        // This preserves the original block's terminator (e.g. branch to the
        // next if-check for t.y), so execution falls through when out of range.
        {
            List<IRInst*> instsToMoveToContBlock;
            for (auto moveInst = rhCall->getNextInst(); moveInst;
                 moveInst = moveInst->getNextInst())
            {
                instsToMoveToContBlock.add(moveInst);
            }
            for (auto moveInst : instsToMoveToContBlock)
            {
                moveInst->insertAtEnd(continueBlock);
            }
        }

        // Emit conditional branch: in range -> accept, out of range -> continue.
        builder.emitIfElse(inRange, acceptBlock, continueBlock, continueBlock);

        // Populate accept block: store attrs to payload, return {true, t}.
        builder.setInsertInto(acceptBlock);
        if (attrsPayloadParam)
            builder.emitStore(attrsPayloadParam, attrs);
        IRInst* acceptArgs[] = {boolTrue, tHit};
        auto acceptVal = builder.emitMakeStruct(resultStructType, 2, acceptArgs);
        builder.emitReturn(acceptVal);

        // Remove the ReportHit call from the continue block.
        rhCall->removeAndDeallocate();
    }

    // Remove replaced instructions.
    for (auto inst : instsToRemove)
    {
        inst->removeAndDeallocate();
    }
    for (auto load : loadsToRemove)
    {
        load->removeAndDeallocate();
    }
    for (auto fieldAddr : fieldAddrsToRemove)
    {
        fieldAddr->removeAndDeallocate();
    }

    // Remove the entryPointParams global for this function.
    if (epParamsGlobal)
    {
        epParamsGlobal->removeAndDeallocate();
    }

    // Change the function return type to BoundingBoxResult struct.
    auto funcType = as<IRFuncType>(func->getDataType());
    if (funcType)
    {
        List<IRType*> paramTypes;
        for (UInt i = 0; i < funcType->getParamCount(); i++)
        {
            paramTypes.add(funcType->getParamType(i));
        }
        auto newFuncType = builder.getFuncType(paramTypes, resultStructType);
        func->setFullType(newFuncType);
    }

    // Fix up the function type to match the new parameter list.
    fixUpFuncType(func);
}

// Phase 3: Replace TraceRay calls in the raygen function with intersector +
// calls to visible closesthit/miss functions (instead of inlining their bodies).
static void legalizeTraceRayCallsWithVisibleFunctions(
    IRFunc* raygenFunc,
    const RTEntryPoints& entryPoints,
    IRBuilder& builder,
    bool isProcedural = false,
    IRType* proceduralAttrsType = nullptr)
{
    // Find all TraceRay calls in the raygen function.
    List<IRCall*> traceRayCalls;
    for (auto block : raygenFunc->getBlocks())
    {
        for (auto inst = block->getFirstOrdinaryInst(); inst; inst = inst->getNextInst())
        {
            auto call = as<IRCall>(inst);
            if (!call)
            {
                continue;
            }
            if (getMetalRTIntrinsicFromCall(call) == MetalRTIntrinsic::TraceRay)
            {
                traceRayCalls.add(call);
            }
        }
    }

    auto uintType = builder.getBasicType(BaseType::UInt);

    for (auto call : traceRayCalls)
    {
        // __metalrt_TraceRay args (updated with MissShaderIndex):
        // 0: accel, 1: rayFlags, 2: instanceMask, 3: missIndex,
        // 4: origin, 5: tMin, 6: direction, 7: tMax, 8: payload
        auto accel = call->getArg(0);
        auto rayFlagsArg = call->getArg(1);
        auto instanceMask = call->getArg(2);
        auto missIndexArg = call->getArg(3);
        auto origin = call->getArg(4);
        auto tMin = call->getArg(5);
        auto direction = call->getArg(6);
        auto tMax = call->getArg(7);
        auto payloadPtr = call->getArg(8);

        // Check for RAY_FLAG_SKIP_CLOSEST_HIT_SHADER (0x08).
        // If rayFlags is a compile-time constant, check statically.
        bool skipClosestHit = false;
        bool forceNonOpaque = false;
        if (auto flagsLit = as<IRIntLit>(rayFlagsArg))
        {
            skipClosestHit = (flagsLit->getValue() & 0x08) != 0;
            forceNonOpaque = (flagsLit->getValue() & 0x02) != 0;
        }

        // Determine which miss shader to call.
        // If missIndex is a constant, use it directly; otherwise fall back to index 0.
        Index missShaderIdx = 0;
        if (auto missLit = as<IRIntLit>(missIndexArg))
        {
            missShaderIdx = (Index)missLit->getValue();
        }
        IRFunc* targetMissFunc = nullptr;
        if (missShaderIdx < entryPoints.missShaders.getCount())
        {
            targetMissFunc = entryPoints.missShaders[missShaderIdx];
        }
        else if (entryPoints.missShaders.getCount() > 0)
        {
            targetMissFunc = entryPoints.missShaders[0];
        }

        // Insert before the TraceRay call.
        builder.setInsertBefore(call);

        // Emit MetalRTIntersect instruction.
        // Use 9-operand form (with IFT + payload) when:
        //   - Procedural mode: intersection function writes attrs via ray_data [[payload]]
        //   - FORCE_NON_OPAQUE with anyhit: anyhit (triangle intersection function) needs
        //     the payload via ray_data to read/write shadow hit results
        // Use 7-operand form (no IFT) otherwise: no intersection functions need to run.
        IRInst* attrsVar = nullptr;
        IRInst* intersectResult = nullptr;
        bool needsAnyhitPayload = forceNonOpaque && entryPoints.anyHit;
        if (isProcedural && proceduralAttrsType)
        {
            attrsVar = builder.emitVar(proceduralAttrsType);
            IRInst* intersectArgs[] = {
                accel, origin, direction, tMin, tMax,
                rayFlagsArg, instanceMask, entryPoints.raygenFuncTable, attrsVar};
            intersectResult = builder.emitIntrinsicInst(
                uintType, kIROp_MetalRTIntersect, 9, intersectArgs);
        }
        else if (needsAnyhitPayload)
        {
            // Shadow ray pattern: pass the DXR payload to intersect() so the
            // anyhit [[intersection(triangle)]] function can access it via ray_data.
            IRInst* intersectArgs[] = {
                accel, origin, direction, tMin, tMax,
                rayFlagsArg, instanceMask, entryPoints.raygenFuncTable, payloadPtr};
            intersectResult = builder.emitIntrinsicInst(
                uintType, kIROp_MetalRTIntersect, 9, intersectArgs);
        }
        else
        {
            IRInst* intersectArgs[] = {
                accel, origin, direction, tMin, tMax, rayFlagsArg, instanceMask};
            intersectResult = builder.emitIntrinsicInst(
                uintType, kIROp_MetalRTIntersect, 7, intersectArgs);
        }

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

        // Hit branch: call closestHitFunc with intersection result fields as arguments.
        // If RAY_FLAG_SKIP_CLOSEST_HIT_SHADER is set, skip the closestHit call.
        builder.setInsertInto(hitBlock);
        if (entryPoints.closestHit && !skipClosestHit)
        {
            auto floatType = builder.getBasicType(BaseType::Float);

            auto distance = builder.emitIntrinsicInst(
                floatType,
                kIROp_MetalRTIntersectionGetDistance,
                1,
                &intersectResult);
            auto primitiveId = builder.emitIntrinsicInst(
                uintType,
                kIROp_MetalRTIntersectionGetPrimitiveId,
                1,
                &intersectResult);
            auto instanceId = builder.emitIntrinsicInst(
                uintType,
                kIROp_MetalRTIntersectionGetInstanceId,
                1,
                &intersectResult);

            // Transform matrix data.
            auto intType = builder.getIntType();
            auto intVal4 = builder.getIntValue(intType, 4);
            auto intVal3 = builder.getIntValue(intType, 3);
            auto noLayout = builder.getIntValue(intType, 0);
            auto float4x3Type = builder.getMatrixType(floatType, intVal4, intVal3, noLayout);
            auto float3x4Type = builder.getMatrixType(floatType, intVal3, intVal4, noLayout);

            auto objectToWorld4x3 = builder.emitIntrinsicInst(
                float4x3Type,
                kIROp_MetalRTIntersectionGetObjectToWorld4x3,
                1,
                &intersectResult);
            auto objectToWorld3x4 = builder.emitIntrinsicInst(
                float3x4Type,
                kIROp_MetalRTIntersectionGetObjectToWorld3x4,
                1,
                &intersectResult);
            auto worldToObject4x3 = builder.emitIntrinsicInst(
                float4x3Type,
                kIROp_MetalRTIntersectionGetWorldToObject4x3,
                1,
                &intersectResult);
            auto worldToObject3x4 = builder.emitIntrinsicInst(
                float3x4Type,
                kIROp_MetalRTIntersectionGetWorldToObject3x4,
                1,
                &intersectResult);

            if (isProcedural)
            {
                if (proceduralAttrsType && attrsVar)
                {
                    // Procedural mode with attrs: load attrs written by intersection
                    // function via ray_data payload, pass as last arg (14 args).
                    auto attrsVal = builder.emitLoad(proceduralAttrsType, attrsVar);
                    IRInst* hitArgs[] = {
                        payloadPtr, origin, direction, tMin, rayFlagsArg,
                        distance, primitiveId, instanceId,
                        objectToWorld4x3, objectToWorld3x4, worldToObject4x3, worldToObject3x4,
                        entryPoints.raygenFuncTable, attrsVal};
                    builder.emitCallInst(
                        builder.getVoidType(), entryPoints.closestHit, 14, hitArgs);
                }
                else
                {
                    // Procedural mode without attrs (13 args).
                    IRInst* hitArgs[] = {
                        payloadPtr, origin, direction, tMin, rayFlagsArg,
                        distance, primitiveId, instanceId,
                        objectToWorld4x3, objectToWorld3x4, worldToObject4x3, worldToObject3x4,
                        entryPoints.raygenFuncTable};
                    builder.emitCallInst(
                        builder.getVoidType(), entryPoints.closestHit, 13, hitArgs);
                }
            }
            else
            {
                // Triangle mode: include barycentrics and frontFacing (15 args).
                auto float2Type = builder.getVectorType(builder.getBasicType(BaseType::Float), 2);
                auto boolType = builder.getBoolType();

                auto barycentrics = builder.emitIntrinsicInst(
                    float2Type,
                    kIROp_MetalRTIntersectionGetBarycentrics,
                    1,
                    &intersectResult);
                auto frontFacing = builder.emitIntrinsicInst(
                    boolType,
                    kIROp_MetalRTIntersectionGetFrontFace,
                    1,
                    &intersectResult);

                IRInst* hitArgs[] = {
                    payloadPtr, origin, direction, tMin, rayFlagsArg,
                    barycentrics, distance, primitiveId, instanceId, frontFacing,
                    objectToWorld4x3, objectToWorld3x4, worldToObject4x3, worldToObject3x4,
                    entryPoints.raygenFuncTable};
                builder.emitCallInst(
                    builder.getVoidType(), entryPoints.closestHit, 15, hitArgs);
            }
        }
        builder.emitBranch(afterBlock);

        // Miss branch: call the appropriate miss function based on MissShaderIndex.
        builder.setInsertInto(missBlock);
        if (targetMissFunc)
        {
            IRInst* missArgs[] = {payloadPtr, origin, direction, tMin, rayFlagsArg};
            builder.emitCallInst(
                builder.getVoidType(), targetMissFunc, 5, missArgs);
        }
        builder.emitBranch(afterBlock);
    }
}

// Lower TraceRay calls inside visible functions (e.g. closestHit) to
// MetalRTIntersect calls that pass the intersection function table and payload.
// The any-hit function runs automatically during intersect() via the function table.
static void legalizeTraceRayCallsInVisibleFunctions(
    IRFunc* visibleFunc,
    IRBuilder& builder,
    IRType* proceduralAttrsType = nullptr)
{
    // Find all TraceRay calls in the visible function.
    List<IRCall*> traceRayCalls;
    for (auto block : visibleFunc->getBlocks())
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

    // Find the function table parameter by scanning for _metalrt_func_table name hint.
    IRInst* funcTableParam = nullptr;
    for (auto param : visibleFunc->getFirstBlock()->getParams())
    {
        if (auto hint = param->findDecoration<IRNameHintDecoration>())
        {
            if (hint->getName() == toSlice("_metalrt_func_table"))
            {
                funcTableParam = param;
                break;
            }
        }
    }

    auto uintType = builder.getBasicType(BaseType::UInt);

    for (auto call : traceRayCalls)
    {
        // __metalrt_TraceRay args:
        // 0: accel, 1: rayFlags, 2: instanceMask, 3: missIndex,
        // 4: origin, 5: tMin, 6: direction, 7: tMax, 8: payload
        auto accel = call->getArg(0);
        auto rayFlagsArg = call->getArg(1);
        auto instanceMask = call->getArg(2);
        // missIndex (arg 3) not used for visible function TraceRay lowering.
        auto origin = call->getArg(4);
        auto tMin = call->getArg(5);
        auto direction = call->getArg(6);
        auto tMax = call->getArg(7);
        auto payloadPtr = call->getArg(8);

        builder.setInsertBefore(call);

        if (proceduralAttrsType)
        {
            // Procedural: intersection function expects ray_data AttrsType& [[payload]],
            // not ray_data DXRPayload&. Pass an attrs variable instead.
            auto attrsVar = builder.emitVar(proceduralAttrsType);
            IRInst* intersectArgs[] = {
                accel, origin, direction, tMin, tMax,
                rayFlagsArg, instanceMask, funcTableParam, attrsVar};
            builder.emitIntrinsicInst(uintType, kIROp_MetalRTIntersect, 9, intersectArgs);
        }
        else
        {
            // Triangle: pass DXR payload (existing behavior).
            IRInst* intersectArgs[] = {
                accel, origin, direction, tMin, tMax,
                rayFlagsArg, instanceMask, funcTableParam, payloadPtr};
            builder.emitIntrinsicInst(uintType, kIROp_MetalRTIntersect, 9, intersectArgs);
        }

        // Just remove the TraceRay — no block splitting needed.
        // The any-hit/intersection ran during intersect() and modified the payload.
        call->removeAndDeallocate();
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

    // Phase 1: Legalize raygen entry point (compute kernel).
    IRFunc* raygenFunc = nullptr;
    for (auto inst : module->getGlobalInsts())
    {
        auto func = as<IRFunc>(inst);
        if (!func)
        {
            continue;
        }

        auto entryPointDecor = func->findDecoration<IREntryPointDecoration>();
        if (!entryPointDecor)
        {
            continue;
        }

        auto stage = entryPointDecor->getProfile().getStage();
        if (stage != Stage::RayGeneration)
        {
            continue;
        }

        entryPoints.raygenFuncTable = legalizeRaygenEntryPoint(func, builder, dispatchDimsBufferSlot);
        raygenFunc = func;
    }

    bool isProcedural = (entryPoints.intersection != nullptr);

    // Extract custom hit attributes type from intersection function's ReportHit calls
    // BEFORE legalization removes them. This type is needed to set up the ray_data payload
    // mechanism for passing attrs from intersection → raygen → closesthit.
    IRType* proceduralAttrsType = nullptr;
    if (entryPoints.intersection)
    {
        for (auto block : entryPoints.intersection->getBlocks())
        {
            for (auto inst = block->getFirstOrdinaryInst(); inst; inst = inst->getNextInst())
            {
                auto call = as<IRCall>(inst);
                if (!call)
                    continue;
                if (getMetalRTIntrinsicFromCall(call) == MetalRTIntrinsic::ReportHit)
                {
                    proceduralAttrsType = call->getArg(2)->getDataType();
                    break;
                }
            }
            if (proceduralAttrsType)
                break;
        }
    }

    if (entryPoints.closestHit || entryPoints.missShaders.getCount() > 0)
    {
        // Phase 3: Transform closesthit/miss into [[visible]] functions.
        if (entryPoints.closestHit)
        {
            legalizeVisibleFunction(
                entryPoints.closestHit, builder, module, true, isProcedural, proceduralAttrsType);
        }
        for (auto missFunc : entryPoints.missShaders)
        {
            legalizeVisibleFunction(missFunc, builder, module, false);
        }

        // Replace TraceRay calls with intersector + calls to visible functions.
        if (raygenFunc)
        {
            legalizeTraceRayCallsWithVisibleFunctions(
                raygenFunc, entryPoints, builder, isProcedural, proceduralAttrsType);
        }

        // Lower TraceRay calls inside visible functions (e.g. shadow rays in closestHit)
        // to MetalRTIntersect calls with intersection function table and payload.
        if (entryPoints.closestHit)
        {
            legalizeTraceRayCallsInVisibleFunctions(
                entryPoints.closestHit, builder, proceduralAttrsType);
        }
    }

    // Phase 4: Transform AnyHit into [[intersection(triangle, instancing)]] function.
    if (entryPoints.anyHit)
    {
        legalizeAnyHitFunction(entryPoints.anyHit, builder, module);
    }

    // Phase 5: Transform Intersection into [[intersection(bounding_box, ...)]] function.
    if (entryPoints.intersection)
    {
        legalizeIntersectionFunction(entryPoints.intersection, builder, module);
    }

    // Do NOT call removeNonRaygenEntryPoints — closesthit/miss/anyhit/intersection stay as
    // entry points for [[visible]]/[[intersection()]] function emission.
}

} // namespace Slang
