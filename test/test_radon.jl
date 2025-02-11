@testset "square phantom" begin
    I = ones(10, 10)
    S = RadonTransform.radon(I, 101, 5, 0.0)
    @test true == all(isapprox.(S[1:14, 1], 0.0, atol=1e-6))
    @test true == all(isapprox.(S[15:51, 1], sqrt(2), atol=1e-6))
    @test true == all(isapprox.(S[1:51, 2], [2t / 50 for t=0:50], atol=1e-6))
    @test true == all(isapprox.(S[1:14, 3], 0.0, atol=1e-6))
    @test true == all(isapprox.(S[15:51, 3], sqrt(2), atol=1e-6))
    @test true == all(isapprox.(S[1:51, 4], [2t / 50 for t=0:50], atol=1e-6))
    @test true == all(isapprox.(S[1:14, 5], 0.0, atol=1e-6))
    @test true == all(isapprox.(S[15:51, 5], sqrt(2), atol=1e-6))
end
