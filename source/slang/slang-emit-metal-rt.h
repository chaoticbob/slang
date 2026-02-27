// slang-emit-metal-rt.h
#ifndef SLANG_EMIT_METAL_RT_H
#define SLANG_EMIT_METAL_RT_H

#include "slang-emit-metal.h"

namespace Slang
{

class MetalRTSourceEmitter : public MetalSourceEmitter
{
public:
    typedef MetalSourceEmitter Super;

    MetalRTSourceEmitter(const Desc& desc)
        : Super(desc)
    {
    }

protected:
    virtual void emitFrontMatterImpl(TargetRequest* targetReq) SLANG_OVERRIDE;
    virtual void emitSimpleFuncParamImpl(IRParam* param) SLANG_OVERRIDE;
    virtual bool tryEmitInstExprImpl(IRInst* inst, const EmitOpInfo& inOuterPrec) SLANG_OVERRIDE;
    virtual bool tryEmitInstStmtImpl(IRInst* inst) SLANG_OVERRIDE;

private:
    int m_intersectorCounter = 0;
};

} // namespace Slang
#endif
