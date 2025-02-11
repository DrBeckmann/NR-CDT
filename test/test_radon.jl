@testset "square phantom" begin 
    using NormalizedRadonCDT.RadonTransform
    I = ones(10, 10)
    S = radon(I; opt=RadonOpt(101, 4, 0.0))
    @test true == all(isapprox.(S[1:15, 1], 0.0, atol=1e-6))
    @test true == all(isapprox.(S[16:51, 1], sqrt(2), atol=1e-6))
    @test true == all(isapprox.(S[1:51, 2], [2t / 50 for t=0:50], atol=1e-6))
    @test true == all(isapprox.(S[1:15, 3], 0.0, atol=1e-6))
    @test true == all(isapprox.(S[16:51, 3], sqrt(2), atol=1e-6))
    @test true == all(isapprox.(S[1:51, 4], [2t / 50 for t=0:50], atol=1e-6))
end
