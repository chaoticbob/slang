Metal Ray Tracing Translation — Current Limitations
====================================================

This document covers the current limitations of the DXR-to-Metal ray tracing
translation in Slang's Metal backend.

## Supported Shader Stages

| DXR Stage | Metal Mapping | Status |
|---|---|---|
| RayGeneration | `[[kernel]]` compute | Supported |
| ClosestHit | `[[visible]]` function | Supported |
| Miss | `[[visible]]` function | Supported |
| AnyHit | `[[intersection(triangle, instancing)]]` | Supported |
| Intersection | `[[intersection(bounding_box, instancing)]]` | Not supported |
| Callable | — | Not supported |

## General Limitations

- **Single entry point per stage.** Only one raygen, one closesthit, one miss,
  and one anyhit function are discovered per compilation. Multiple shaders of
  the same stage type are not supported.

- **No recursive TraceRay.** TraceRay is lowered to an inline
  `intersector.intersect()` call with closesthit/miss dispatched as visible
  function calls from the raygen kernel. Recursive trace calls from closesthit
  or miss shaders are not handled.

- **Triangle geometry only.** The intersector is always configured as
  `intersector<triangle_data, instancing>`. Procedural (AABB) geometry is not
  supported.

- **Fixed thread group size.** The raygen compute kernel is emitted with
  `[numthreads(8, 8, 1)]`. This is not configurable.

- **Dispatch dimensions passed via constant buffer.** `DispatchRaysDimensions()`
  reads from a constant buffer at a fixed binding slot (default `buffer(30)`,
  configurable via `MetalRTDispatchDimsBufferSlot`). The host application must
  bind a `uint3` buffer at that slot containing the dispatch dimensions.

## AnyHit Limitations

- **No payload access.** Metal intersection functions cannot access the ray
  payload. AnyHit shaders that read or modify the payload will have those
  accesses silently dropped. This is acceptable for common use cases like alpha
  testing based on geometry/texture data, but shaders that depend on payload
  modifications in AnyHit will not behave correctly.

- **HitKind is a constant.** Since the AnyHit function does not receive a
  front-facing parameter, `HitKind()` returns a constant value (254,
  `HIT_KIND_TRIANGLE_FRONT_FACE`) rather than distinguishing front/back faces.

- **No WorldRayOrigin/WorldRayDirection/RayTMin/RayTCurrent/RayFlags.** These
  ray query intrinsics are not currently wired up in AnyHit. They require
  additional Metal system value parameters that are not yet added.

## ClosestHit / Miss Limitations

- **Payload passed as thread-address pointer.** The closesthit and miss visible
  functions receive the payload as a `thread T*` pointer. This works for the
  raygen-calls-visible-function pattern but would not work if these functions
  were called from a different address space context.

## Intersection Shader (Not Supported)

DXR intersection shaders (`[shader("intersection")]`) for procedural/AABB
geometry are not implemented. Metal uses the same intersection function concept
for both DXR's AnyHit (triangle) and DXR's Intersection (procedural) — the
difference is the `triangle` vs `bounding_box` tag. The basic DXR-to-Metal
mapping would be:

| DXR | Metal |
|---|---|
| `[shader("intersection")]` | `[[intersection(bounding_box, instancing)]]` |
| `ObjectRayOrigin()` | `float3 origin [[origin]]` parameter |
| `ObjectRayDirection()` | `float3 direction [[direction]]` parameter |
| `RayTMin()` | `float min_dist [[min_distance]]` parameter |
| `ReportHit(t, kind, attrs)` | Set distance in result, return `true` |
| No hit reported | Return `false` |

### What would work

The simple case — a single `ReportHit()` call with no custom hit attributes
beyond the hit distance — maps cleanly to a `bounding_box` intersection
function using the same pattern as the AnyHit translation. The function returns
`bool`, and the hit distance is communicated through the result.

### Challenges

- **Multiple `ReportHit()` calls.** DXR allows calling `ReportHit()` multiple
  times within one intersection shader invocation to report multiple candidate
  hits. Metal's bounding box intersection function returns a single result. A
  shader with multiple `ReportHit()` calls would need to be transformed to
  track the closest `t` value across all calls and return only that one.

- **Custom hit attributes.** DXR passes an arbitrary struct through
  `ReportHit(t, kind, attrs)` that is later available in the ClosestHit and
  AnyHit shaders via the attributes parameter. Metal does not have a built-in
  mechanism to pass custom data from a bounding box intersection function to
  the rest of the pipeline. The workaround would be to stash attributes in a
  device buffer keyed by primitive ID during intersection, then read them back
  in ClosestHit/AnyHit — but this requires host-side buffer allocation and
  introduces synchronization concerns.

- **Intersector tag change.** The raygen kernel's intersector would need to use
  a different template configuration for AABB geometry. Supporting mixed
  triangle and procedural geometry in the same scene would require additional
  complexity.

## Callable Shaders (Not Supported)

`CallShader()` and `[shader("callable")]` are not implemented. Metal does not
have a direct equivalent — these would need to be lowered to visible function
calls with manual dispatch.
