// slang-emit-metal-rt.cpp
#include "slang-emit-metal-rt.h"

#include "slang-ir-insts.h"

namespace Slang
{

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
            m_writer->emit("[[intersection(triangle, instancing)]] ");
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
    // Check if this is a _metalrt_* param with a buffer binding.
    if (auto nameHint = param->findDecoration<IRNameHintDecoration>())
    {
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

bool MetalRTSourceEmitter::tryEmitInstStmtImpl(IRInst* inst)
{
    switch (inst->getOp())
    {
    case kIROp_MetalRTIntersect:
        {
            int idx = m_intersectorCounter++;
            auto resultName = getName(inst);

            // Emit: intersector<triangle_data, instancing> _i_N;
            m_writer->emit("intersector<triangle_data, instancing> _i_");
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

            // auto <result> = _i_N.intersect(_r_N, <accel>, <mask>);
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
    default:
        break;
    }

    return Super::tryEmitInstExprImpl(inst, inOuterPrec);
}

} // namespace Slang
