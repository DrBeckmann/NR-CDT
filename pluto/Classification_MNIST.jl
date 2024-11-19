### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 03f9300e-692e-11ef-145c-85ecce1e4c7f
begin
	import Pkg
	Pkg.activate("..")
	using Revise
	using NormalizedRadonCDT.TestImages
	using NormalizedRadonCDT
end

# ╔═╡ 75fa041d-f4a0-4cb0-8af7-b09ff2249874
using MLDatasets, Plots, Random;

# ╔═╡ 0eb4cfdc-c875-4d0f-b46e-531e72fc06c3
trainset = MNIST(:train)

# ╔═╡ 907c64eb-30d8-4cbe-af47-3d4b6d72e3ca
Null = trainset[2].features;

# ╔═╡ 997d8a2c-b11b-4d9e-9812-927a75454479
heatmap(Null)

# ╔═╡ 161664b7-c90c-4ff3-b7d4-1f5ccd536459
heatmap(NormalizedRadonCDT.RadonTransform.radon(Float64.(Null), 40, 1060, 0.0))

# ╔═╡ a71828ea-c868-4a39-abf9-4f0a63b10844
number_mnist_1 = [1,7]

# ╔═╡ 4b0a3bdb-b0f7-4b84-b886-7b81639fbe2e
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 10000, 42, 32, 128, 5, 2, 0, 0)

# ╔═╡ 98b11c4e-f04a-4bfe-b22a-0ea44610430c
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 1000, 42, 32, 128, 5, 2, 0, 0)

# ╔═╡ 38f7a6ef-f1fb-43b2-8d45-af141a8f4f17
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 500, 42, 32, 128, 5, 2, 0, 0)

# ╔═╡ 30909840-1fec-4554-b75c-6331bb483be6
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 100, 42, 32, 128, 5, 2, 0, 0)

# ╔═╡ ecb9f9ea-f988-498e-add0-cc4850a8fac9
number_mnist_2 = [1,3]

# ╔═╡ 88b0e3f5-6b75-4634-bd2f-ff7035b3d7f5
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_2, 10000, 42, 32, 128, 5, 2, 0, 0)

# ╔═╡ a2d8c900-33c8-434e-99b8-1fe0531e7bc6
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_2, 1000, 42, 32, 128, 5, 2, 0, 0)

# ╔═╡ 9b458f45-10f8-48ae-bbbd-8d707d3082cd
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_2, 500, 42, 32, 128, 5, 2, 0, 0)

# ╔═╡ 7690a2dd-6ac5-4aea-9046-c42a9571cd79
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_2, 100, 42, 32, 128, 5, 2, 0, 0)

# ╔═╡ a33eb82f-7a45-4ca2-8f3a-23b457dd46d2
number_mnist_3 = [1,3,7]

# ╔═╡ d3726cf8-76cd-48b7-b05b-197eccac6de5
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_3, 10000, 42, 32, 128, 5, 2, 0, 0)

# ╔═╡ 2b254abc-eecd-4c19-b19a-aae412596087
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_3, 1000, 42, 32, 128, 5, 2, 0, 0)

# ╔═╡ 33cc1d97-cc66-4bb5-9a9b-338256c621d3
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_3, 500, 42, 32, 128, 5, 2, 0, 0)

# ╔═╡ 26a838f9-b4a0-4b41-ab2c-b7cf573e46cb
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_3, 100, 42, 32, 128, 5, 2, 0, 0)

# ╔═╡ Cell order:
# ╠═03f9300e-692e-11ef-145c-85ecce1e4c7f
# ╠═75fa041d-f4a0-4cb0-8af7-b09ff2249874
# ╠═0eb4cfdc-c875-4d0f-b46e-531e72fc06c3
# ╠═907c64eb-30d8-4cbe-af47-3d4b6d72e3ca
# ╠═997d8a2c-b11b-4d9e-9812-927a75454479
# ╠═161664b7-c90c-4ff3-b7d4-1f5ccd536459
# ╠═a71828ea-c868-4a39-abf9-4f0a63b10844
# ╠═4b0a3bdb-b0f7-4b84-b886-7b81639fbe2e
# ╠═98b11c4e-f04a-4bfe-b22a-0ea44610430c
# ╠═38f7a6ef-f1fb-43b2-8d45-af141a8f4f17
# ╠═30909840-1fec-4554-b75c-6331bb483be6
# ╠═ecb9f9ea-f988-498e-add0-cc4850a8fac9
# ╠═88b0e3f5-6b75-4634-bd2f-ff7035b3d7f5
# ╠═a2d8c900-33c8-434e-99b8-1fe0531e7bc6
# ╠═9b458f45-10f8-48ae-bbbd-8d707d3082cd
# ╠═7690a2dd-6ac5-4aea-9046-c42a9571cd79
# ╠═a33eb82f-7a45-4ca2-8f3a-23b457dd46d2
# ╠═d3726cf8-76cd-48b7-b05b-197eccac6de5
# ╠═2b254abc-eecd-4c19-b19a-aae412596087
# ╠═33cc1d97-cc66-4bb5-9a9b-338256c621d3
# ╠═26a838f9-b4a0-4b41-ab2c-b7cf573e46cb
