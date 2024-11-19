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

# ╔═╡ 7fab2e41-9342-4c96-8aaf-43258aab4f66
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 10, 42, 1, 1, 1, 0, 0)

# ╔═╡ 3f7c0e09-98ce-4e3c-8002-2e253c0aa281
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 10, 42, 2, 2, 1, 0, 0)

# ╔═╡ 6e43e0fa-1ef8-4807-b80d-90c5fa5a72ec
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 10, 42, 4, 4, 1, 0, 0)

# ╔═╡ 2e5c469a-8001-4a7b-8905-c017617e7f80
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 10, 42, 8, 8, 1, 0, 0)

# ╔═╡ 2303943c-8db9-43e9-aa1c-d7e31b7bdf59
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 10, 42, 16, 16, 1, 0, 0)

# ╔═╡ 72ff2bf5-2525-416f-a812-2b45a79dacd9
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 10, 42, 32, 32, 1, 0, 0)

# ╔═╡ cf47b43e-0e2e-4a0d-8613-6a855b969a58
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 30, 42, 1, 1, 1, 0, 0)

# ╔═╡ 318272bb-09fc-4221-8936-d900f8481365
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 30, 42, 2, 2, 1, 0, 0)

# ╔═╡ 6fc0fb42-1ae3-4ca8-8783-47373d78ef68
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 30, 42, 32, 4, 1, 0, 0)

# ╔═╡ 23734fa4-bdfc-42f4-98a7-6dadf9df416a
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 30, 42, 32, 8, 1, 0, 0)

# ╔═╡ 0bd15fa0-24da-4f8a-a696-e7084102babd
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 30, 42, 16, 16, 1, 0, 0)

# ╔═╡ 921653b4-ea6a-4d73-8a6d-23fee688478a
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 30, 42, 32, 32, 1, 0, 0)

# ╔═╡ 27670a58-ab73-49c4-987f-20130c1cae7d
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 90, 42, 1, 1, 1, 0, 0)

# ╔═╡ a6e7350e-bbe0-4389-a1dd-432cc1aaa3ec
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 90, 42, 2, 2, 1, 0, 0)

# ╔═╡ 51459344-23aa-4be0-83e5-8d6ed8306cb6
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 90, 42, 4, 4, 1, 0, 0)

# ╔═╡ 032403d3-b22a-45e6-94f2-a00ab532702e
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 90, 42, 8, 8, 1, 0, 0)

# ╔═╡ 7a875314-dec1-4e20-b2c5-35024c1f98e0
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 90, 42, 16, 16, 1, 0, 0)

# ╔═╡ 981a6723-6f70-4347-a9d7-2dcd303f7d2f
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 270, 42, 2, 2, 1, 0, 0)

# ╔═╡ 98802b7b-6855-43c6-9cb3-6124fe4b2479
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 270, 42, 4, 4, 1, 0, 0)

# ╔═╡ 55ff621d-2936-4f9b-95d1-043e8d641183
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 270, 42, 8, 8, 1, 0, 0)

# ╔═╡ f1b3707c-806f-48f4-a9d7-48548cce711c
@time NormalizedRadonCDT.classify_data_NRCDT([1,6], 128, 270, 42, 16, 16, 1, 0, 0)

# ╔═╡ Cell order:
# ╠═03f9300e-692e-11ef-145c-85ecce1e4c7f
# ╠═7fab2e41-9342-4c96-8aaf-43258aab4f66
# ╠═3f7c0e09-98ce-4e3c-8002-2e253c0aa281
# ╠═6e43e0fa-1ef8-4807-b80d-90c5fa5a72ec
# ╠═2e5c469a-8001-4a7b-8905-c017617e7f80
# ╠═2303943c-8db9-43e9-aa1c-d7e31b7bdf59
# ╠═72ff2bf5-2525-416f-a812-2b45a79dacd9
# ╠═cf47b43e-0e2e-4a0d-8613-6a855b969a58
# ╠═318272bb-09fc-4221-8936-d900f8481365
# ╠═6fc0fb42-1ae3-4ca8-8783-47373d78ef68
# ╠═23734fa4-bdfc-42f4-98a7-6dadf9df416a
# ╠═0bd15fa0-24da-4f8a-a696-e7084102babd
# ╠═921653b4-ea6a-4d73-8a6d-23fee688478a
# ╠═27670a58-ab73-49c4-987f-20130c1cae7d
# ╠═a6e7350e-bbe0-4389-a1dd-432cc1aaa3ec
# ╠═51459344-23aa-4be0-83e5-8d6ed8306cb6
# ╠═032403d3-b22a-45e6-94f2-a00ab532702e
# ╠═7a875314-dec1-4e20-b2c5-35024c1f98e0
# ╠═981a6723-6f70-4347-a9d7-2dcd303f7d2f
# ╠═98802b7b-6855-43c6-9cb3-6124fe4b2479
# ╠═55ff621d-2936-4f9b-95d1-043e8d641183
# ╠═f1b3707c-806f-48f4-a9d7-48548cce711c
