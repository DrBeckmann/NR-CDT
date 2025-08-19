module NormalizedRadonCDT

using Interpolations: LinearInterpolation
using Statistics

export TestImages
export RadonCDT, RadonTransform
export NormRadonCDT, MaxNormRadonCDT, MeanNormRadonCDT
export normalization, filter_angles, max_normalization, mean_normalization
export DataTransformations
export Classify


include("TestImages.jl")
include("RadonTransform.jl")
include("RadonCDT.jl")
include("DataTransformations.jl")
include("Classify.jl")

end
