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

# ╔═╡ a71828ea-c868-4a39-abf9-4f0a63b10844
number_mnist_1 = [1,5,7]

# ╔═╡ 0ea6e146-6f2d-40e2-8fea-c0e56942b3af
@time NormalizedRadonCDT.nearest_cross_mnist_NRCDT(number_mnist_1, 50, 42, 2, 2, 0, 0)

# ╔═╡ 721d6367-e101-4b90-bd47-72f6d216aeba
@time NormalizedRadonCDT.nearest_cross_mnist_NRCDT(number_mnist_1, 50, 42, 4, 2, 0, 0)

# ╔═╡ fe5f48bb-492d-41dc-860b-51ac78c89bf0
@time NormalizedRadonCDT.nearest_cross_mnist_NRCDT(number_mnist_1, 50, 42, 8, 2, 0, 0)

# ╔═╡ 87c58c53-d24e-44de-b88d-b1bca48ba070
@time NormalizedRadonCDT.nearest_cross_mnist_NRCDT(number_mnist_1, 50, 42, 16, 2, 0, 0)

# ╔═╡ ec62abf0-b2be-487d-8e4f-7961ad87e40c
@time NormalizedRadonCDT.nearest_cross_mnist_NRCDT(number_mnist_1, 50, 42, 32, 2, 0, 0)

# ╔═╡ 9084c008-ee24-41a3-b464-7cc47d911954
@time NormalizedRadonCDT.nearest_cross_mnist_NRCDT(number_mnist_1, 50, 42, 64, 2, 0, 0)

# ╔═╡ a850afe5-9eae-42ea-bc17-76375cf7d439
@time NormalizedRadonCDT.nearest_cross_mnist_NRCDT(number_mnist_1, 50, 42, 128, 2, 0, 0)

# ╔═╡ Cell order:
# ╠═03f9300e-692e-11ef-145c-85ecce1e4c7f
# ╠═a71828ea-c868-4a39-abf9-4f0a63b10844
# ╠═0ea6e146-6f2d-40e2-8fea-c0e56942b3af
# ╠═721d6367-e101-4b90-bd47-72f6d216aeba
# ╠═fe5f48bb-492d-41dc-860b-51ac78c89bf0
# ╠═87c58c53-d24e-44de-b88d-b1bca48ba070
# ╠═ec62abf0-b2be-487d-8e4f-7961ad87e40c
# ╠═9084c008-ee24-41a3-b464-7cc47d911954
# ╠═a850afe5-9eae-42ea-bc17-76375cf7d439
