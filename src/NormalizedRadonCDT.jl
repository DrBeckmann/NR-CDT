module NormalizedRadonCDT

using Interpolations: LinearInterpolation
using Statistics

export TestImages
export RadonCDT, RadonTransform
export NormRadonCDT, MaxNormRadonCDT, MeanNormRadonCDT
export RandomAffineTransformation

include("TestImages.jl")
include("RadonTransform.jl")
include("RadonCDT.jl")
include("DataTransformations.jl")
include("Classify.jl")

end
