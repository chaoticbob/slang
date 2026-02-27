// slang-emit-metal-rt.cpp
#include "slang-emit-metal-rt.h"

#include "slang-ir-insts.h"

namespace Slang
{

void MetalRTSourceEmitter::emitFrontMatterImpl(TargetRequest* targetReq)
{
    Super::emitFrontMatterImpl(targetReq);
    m_writer->emit("#include <metal_raytracing>\n");
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

} // namespace Slang
