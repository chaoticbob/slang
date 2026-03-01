// slang-emit-metal-rt.cpp
#include "slang-emit-metal-rt.h"

#include "slang-ir-insts.h"

namespace Slang
{

void MetalRTSourceEmitter::beforeComputeEmitActions(IRModule* module)
{
    Super::beforeComputeEmitActions(module);

    // Scan for intersection stage to determine geometry type.
    // Must happen before emit actions so m_isProcedural is set
    // when intersector/function_table templates are emitted.
    for (auto inst : module->getGlobalInsts())
    {
        if (auto func = as<IRFunc>(inst))
        {
            if (auto ep = func->findDecoration<IREntryPointDecoration>())
            {
                if (ep->getProfile().getStage() == Stage::Intersection)
                {
                    m_isProcedural = true;
                    break;
                }
            }
        }
    }
}

void MetalRTSourceEmitter::emitFrontMatterImpl(TargetRequest* targetReq)
{
    Super::emitFrontMatterImpl(targetReq);
    m_writer->emit("#include <metal_raytracing>\n");
    m_writer->emit("using namespace metal::raytracing;\n");
}

void MetalRTSourceEmitter::emitEntryPointAttributesImpl(
    IRFunc* irFunc,
    IREntryPointDecoration* entryPointDecor)
{
    auto stage = entryPointDecor->getProfile().getStage();
    switch (stage)
    {
    case Stage::ClosestHit:
    case Stage::Miss:
        {
            m_writer->emit("[[visible]] ");
        }
        break;
    case Stage::AnyHit:
        {
            m_writer->emit("[[intersection(triangle, triangle_data, instancing, world_space_data)]] ");
        }
        break;
    case Stage::Intersection:
        {
            m_writer->emit("[[intersection(bounding_box, instancing, world_space_data)]] ");
        }
        break;
    default:
        {
            Super::emitEntryPointAttributesImpl(irFunc, entryPointDecor);
        }
        break;
    }
}

void MetalRTSourceEmitter::emitSimpleFuncParamImpl(IRParam* param)
{
    // Check if this is a _metalrt_* param with special handling.
    if (auto nameHint = param->findDecoration<IRNameHintDecoration>())
    {
        if (nameHint->getName() == toSlice("_metalrt_func_table"))
        {
            // Emit intersection_function_table with correct template tags for geometry type.
            if (m_isProcedural)
                m_writer->emit("intersection_function_table<instancing, world_space_data> ");
            else
                m_writer->emit("intersection_function_table<triangle_data, instancing, world_space_data> ");
            m_writer->emit(getName(param));
            if (auto sysVal = param->findDecoration<IRTargetSystemValueDecoration>())
            {
                m_writer->emit(" [[");
                m_writer->emit(sysVal->getSemantic());
                m_writer->emit("]]");
            }
            return;
        }
        if (nameHint->getName() == toSlice("_metalrt_payload"))
        {
            // Emit: ray_data PayloadType& name [[payload]]
            auto ptrType = as<IRPtrTypeBase>(param->getDataType());
            m_writer->emit("ray_data ");
            emitSimpleType(ptrType->getValueType());
            m_writer->emit("& ");
            m_writer->emit(getName(param));
            m_writer->emit(" [[payload]]");
            return;
        }
        if (nameHint->getName().startsWith(toSlice("_metalrt_")))
        {
            if (auto sysVal = param->findDecoration<IRTargetSystemValueDecoration>())
            {
                // Emit: constant uint3& name [[buffer(30)]]
                m_writer->emit("constant ");
                emitSimpleType(param->getDataType());
                m_writer->emit("& ");
                m_writer->emit(getName(param));
                m_writer->emit(" [[");
                m_writer->emit(sysVal->getSemantic());
                m_writer->emit("]]");
                return;
            }
        }
    }

    // Emit the base param type and name.
    Super::emitSimpleFuncParamImpl(param);

    // If the parent didn't emit a system value semantic (because no IRLayoutDecoration
    // was present), emit it now for params we injected during legalization.
    if (!param->findDecoration<IRLayoutDecoration>())
    {
        maybeEmitSystemSemantic(param);
    }
}

void MetalRTSourceEmitter::_emitStoreImpl(IRStore* store)
{
    // ray_data T* in the IR is emitted as ray_data T& (reference) in MSL.
    // Storing to a reference is just ref = value (no * dereference).
    auto dstPtr = store->getPtr();
    auto ptrType = as<IRPtrTypeBase>(dstPtr->getDataType());
    if (ptrType && ptrType->getAddressSpace() == AddressSpace::MetalRayData)
    {
        emitOperand(dstPtr, getInfo(EmitOp::General));
        m_writer->emit(" = ");
        emitOperand(store->getVal(), getInfo(EmitOp::General));
        m_writer->emit(";\n");
        return;
    }
    Super::_emitStoreImpl(store);
}

bool MetalRTSourceEmitter::tryEmitInstStmtImpl(IRInst* inst)
{
    switch (inst->getOp())
    {
    case kIROp_MetalRTIntersect:
        {
            int idx = m_intersectorCounter++;
            auto resultName = getName(inst);

            // Emit intersector template with correct tags for geometry type.
            if (m_isProcedural)
                m_writer->emit("intersector<instancing, world_space_data> _i_");
            else
                m_writer->emit("intersector<triangle_data, instancing, world_space_data> _i_");
            m_writer->emit(idx);
            m_writer->emit(";\n");

            // Emit: ray _r_N;
            m_writer->emit("ray _r_");
            m_writer->emit(idx);
            m_writer->emit(";\n");

            // _r_N.origin = <origin>;
            m_writer->emit("_r_");
            m_writer->emit(idx);
            m_writer->emit(".origin = ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(";\n");

            // _r_N.direction = <direction>;
            m_writer->emit("_r_");
            m_writer->emit(idx);
            m_writer->emit(".direction = ");
            emitOperand(inst->getOperand(2), getInfo(EmitOp::General));
            m_writer->emit(";\n");

            // _r_N.min_distance = <tmin>;
            m_writer->emit("_r_");
            m_writer->emit(idx);
            m_writer->emit(".min_distance = ");
            emitOperand(inst->getOperand(3), getInfo(EmitOp::General));
            m_writer->emit(";\n");

            // _r_N.max_distance = <tmax>;
            m_writer->emit("_r_");
            m_writer->emit(idx);
            m_writer->emit(".max_distance = ");
            emitOperand(inst->getOperand(4), getInfo(EmitOp::General));
            m_writer->emit(";\n");

            // Configure intersector from HLSL ray flags (operand 5).
            // RAY_FLAG_FORCE_OPAQUE = 0x01, RAY_FLAG_FORCE_NON_OPAQUE = 0x02,
            // RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH = 0x04
            {
                auto rayFlags = inst->getOperand(5);
                bool hasIFT = inst->getOperandCount() >= 9;

                // In procedural (bounding_box) mode, force all geometry to
                // non-opaque so intersection functions are always called.
                // Metal skips [[intersection(...)]] functions for opaque geometry,
                // but bounding_box intersection functions MUST always run
                // (there is no built-in AABB intersection).
                // DXR's RAY_FLAG_FORCE_OPAQUE only skips any-hit shaders, not
                // intersection shaders.
                if (m_isProcedural)
                {
                    m_writer->emit("_i_");
                    m_writer->emit(idx);
                    m_writer->emit(".force_opacity(forced_opacity::non_opaque);\n");
                }
                else
                {
                    m_writer->emit("if (");
                    emitOperand(rayFlags, getInfo(EmitOp::General));
                    m_writer->emit(" & 0x01u) _i_");
                    m_writer->emit(idx);
                    m_writer->emit(".force_opacity(forced_opacity::opaque);\n");

                    m_writer->emit("if (");
                    emitOperand(rayFlags, getInfo(EmitOp::General));
                    m_writer->emit(" & 0x02u) _i_");
                    m_writer->emit(idx);
                    m_writer->emit(".force_opacity(forced_opacity::non_opaque);\n");
                }

                // Metal's accept_any_intersection(true) skips intersection functions
                // entirely. In DXR, RAY_FLAG_ACCEPT_FIRST_HIT still runs any-hit
                // shaders before accepting. When an intersection function table is
                // present, only enable accept_any_intersection when FORCE_NON_OPAQUE
                // is not set, so that intersection functions are still called for
                // non-opaque geometry.
                if (hasIFT)
                {
                    m_writer->emit("if ((");
                    emitOperand(rayFlags, getInfo(EmitOp::General));
                    m_writer->emit(" & 0x04u) && !(");
                    emitOperand(rayFlags, getInfo(EmitOp::General));
                    m_writer->emit(" & 0x02u)) _i_");
                    m_writer->emit(idx);
                    m_writer->emit(".accept_any_intersection(true);\n");
                }
                else
                {
                    m_writer->emit("if (");
                    emitOperand(rayFlags, getInfo(EmitOp::General));
                    m_writer->emit(" & 0x04u) _i_");
                    m_writer->emit(idx);
                    m_writer->emit(".accept_any_intersection(true);\n");
                }
            }

            // 7-operand: intersect(ray, accel, mask)
            // 9-operand: intersect(ray, accel, mask, funcTable, *payload)
            if (inst->getOperandCount() >= 9)
            {
                m_writer->emit("auto ");
                m_writer->emit(resultName);
                m_writer->emit(" = _i_");
                m_writer->emit(idx);
                m_writer->emit(".intersect(_r_");
                m_writer->emit(idx);
                m_writer->emit(", ");
                emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
                m_writer->emit(", ");
                emitOperand(inst->getOperand(6), getInfo(EmitOp::General));
                m_writer->emit(", ");
                emitOperand(inst->getOperand(7), getInfo(EmitOp::General));
                // Payload operand: local variables (IRVar) must be emitted as
                // a direct variable name for Metal's ray_data write-back to work.
                // Device pointer params need * to dereference.
                auto payloadOperand = inst->getOperand(8);
                m_writer->emit(", ");
                if (as<IRVar>(payloadOperand))
                {
                    m_writer->emit(getName(payloadOperand));
                }
                else
                {
                    m_writer->emit("*");
                    emitOperand(payloadOperand, getInfo(EmitOp::General));
                }
                m_writer->emit(");\n");
            }
            else
            {
                m_writer->emit("auto ");
                m_writer->emit(resultName);
                m_writer->emit(" = _i_");
                m_writer->emit(idx);
                m_writer->emit(".intersect(_r_");
                m_writer->emit(idx);
                m_writer->emit(", ");
                emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
                m_writer->emit(", ");
                emitOperand(inst->getOperand(6), getInfo(EmitOp::General));
                m_writer->emit(");\n");
            }
            return true;
        }
    default:
        break;
    }

    return Super::tryEmitInstStmtImpl(inst);
}

bool MetalRTSourceEmitter::tryEmitInstExprImpl(IRInst* inst, const EmitOpInfo& inOuterPrec)
{
    switch (inst->getOp())
    {
    case kIROp_Load:
        {
            // ray_data T* in the IR is emitted as ray_data T& (reference) in MSL.
            // Loading from a reference is just the reference itself (no * dereference).
            auto base = inst->getOperand(0);
            auto basePtrType = as<IRPtrTypeBase>(base->getDataType());
            if (basePtrType && basePtrType->getAddressSpace() == AddressSpace::MetalRayData)
            {
                emitOperand(base, inOuterPrec);
                return true;
            }
            break;
        }
    case kIROp_MetalRTIntersectionGetType:
        {
            EmitOpInfo outerPrec = inOuterPrec;
            auto prec = getInfo(EmitOp::Prefix);
            bool needClose = maybeEmitParens(outerPrec, prec);
            m_writer->emit("(uint)");
            emitOperand(inst->getOperand(0), rightSide(outerPrec, prec));
            m_writer->emit(".type");
            maybeCloseParens(needClose);
            return true;
        }
    case kIROp_MetalRTIntersectionGetDistance:
        {
            EmitOpInfo outerPrec = inOuterPrec;
            auto prec = getInfo(EmitOp::Postfix);
            bool needClose = maybeEmitParens(outerPrec, prec);
            emitOperand(inst->getOperand(0), leftSide(outerPrec, prec));
            m_writer->emit(".distance");
            maybeCloseParens(needClose);
            return true;
        }
    case kIROp_MetalRTIntersectionGetPrimitiveId:
        {
            EmitOpInfo outerPrec = inOuterPrec;
            auto prec = getInfo(EmitOp::Postfix);
            bool needClose = maybeEmitParens(outerPrec, prec);
            emitOperand(inst->getOperand(0), leftSide(outerPrec, prec));
            m_writer->emit(".primitive_id");
            maybeCloseParens(needClose);
            return true;
        }
    case kIROp_MetalRTIntersectionGetInstanceId:
        {
            EmitOpInfo outerPrec = inOuterPrec;
            auto prec = getInfo(EmitOp::Postfix);
            bool needClose = maybeEmitParens(outerPrec, prec);
            emitOperand(inst->getOperand(0), leftSide(outerPrec, prec));
            m_writer->emit(".instance_id");
            maybeCloseParens(needClose);
            return true;
        }
    case kIROp_MetalRTIntersectionGetBarycentrics:
        {
            EmitOpInfo outerPrec = inOuterPrec;
            auto prec = getInfo(EmitOp::Postfix);
            bool needClose = maybeEmitParens(outerPrec, prec);
            emitOperand(inst->getOperand(0), leftSide(outerPrec, prec));
            m_writer->emit(".triangle_barycentric_coord");
            maybeCloseParens(needClose);
            return true;
        }
    case kIROp_MetalRTIntersectionGetFrontFace:
        {
            EmitOpInfo outerPrec = inOuterPrec;
            auto prec = getInfo(EmitOp::Postfix);
            bool needClose = maybeEmitParens(outerPrec, prec);
            emitOperand(inst->getOperand(0), leftSide(outerPrec, prec));
            m_writer->emit(".triangle_front_facing");
            maybeCloseParens(needClose);
            return true;
        }
    case kIROp_MetalRTIntersectionGetObjectToWorld4x3:
        {
            EmitOpInfo outerPrec = inOuterPrec;
            auto prec = getInfo(EmitOp::Postfix);
            bool needClose = maybeEmitParens(outerPrec, prec);
            emitOperand(inst->getOperand(0), leftSide(outerPrec, prec));
            m_writer->emit(".object_to_world_transform");
            maybeCloseParens(needClose);
            return true;
        }
    case kIROp_MetalRTIntersectionGetObjectToWorld3x4:
        {
            m_writer->emit("transpose(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(".object_to_world_transform)");
            return true;
        }
    case kIROp_MetalRTIntersectionGetWorldToObject4x3:
        {
            EmitOpInfo outerPrec = inOuterPrec;
            auto prec = getInfo(EmitOp::Postfix);
            bool needClose = maybeEmitParens(outerPrec, prec);
            emitOperand(inst->getOperand(0), leftSide(outerPrec, prec));
            m_writer->emit(".world_to_object_transform");
            maybeCloseParens(needClose);
            return true;
        }
    case kIROp_MetalRTIntersectionGetWorldToObject3x4:
        {
            m_writer->emit("transpose(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(".world_to_object_transform)");
            return true;
        }
    default:
        break;
    }

    return Super::tryEmitInstExprImpl(inst, inOuterPrec);
}

} // namespace Slang
