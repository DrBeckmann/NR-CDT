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

# ╔═╡ 124e44b3-35a7-4269-a111-f1d0c6bdd40c
temp_data = [1,12]
# temp_data = [1,6]

# ╔═╡ 47ffa7c2-dc44-4335-a314-840bcb743f54
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 10, 42, 1, 1, 1, 3.3, 0)

# ╔═╡ d4b1707b-2459-412e-9352-3ad75d8c0e91
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 10, 42, 2, 2, 1, 3.3, 0)

# ╔═╡ e5e3e712-8a02-4b8c-9a14-2cb6a2e7f0f8
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 10, 42, 4, 4, 1, 3.3, 0)

# ╔═╡ 481e76c0-4a19-4516-a962-7f9d508bcf0e
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 10, 42, 8, 8, 1, 3.3, 0)

# ╔═╡ 46f4270c-6f4b-4a07-83f1-0de077247d46
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 10, 42, 16, 16, 1, 3.3, 0)

# ╔═╡ 30445f56-dcc2-421f-b163-7bd4eec372d8
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 10, 42, 32, 32, 1, 3.3, 0)

# ╔═╡ 7d022f84-955b-4ab0-85e0-9141457fcb4c
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 10, 42, 64, 64, 1, 3.3, 0)

# ╔═╡ 95dc4ee4-c519-4ef3-ace9-eb8b35dc33c8
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 10, 42, 128, 128, 1, 3.3, 0)

# ╔═╡ 7fab2e41-9342-4c96-8aaf-43258aab4f66
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 10, 42, 1, 1, 1, 2, 0)

# ╔═╡ 3f7c0e09-98ce-4e3c-8002-2e253c0aa281
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 10, 42, 2, 2, 1, 2, 0)

# ╔═╡ 6e43e0fa-1ef8-4807-b80d-90c5fa5a72ec
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 10, 42, 4, 4, 1, 2, 0)

# ╔═╡ 2e5c469a-8001-4a7b-8905-c017617e7f80
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 10, 42, 8, 8, 1, 2, 0)

# ╔═╡ 2303943c-8db9-43e9-aa1c-d7e31b7bdf59
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 10, 42, 16, 16, 1, 2, 0)

# ╔═╡ 72ff2bf5-2525-416f-a812-2b45a79dacd9
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 10, 42, 32, 32, 1, 2, 0)

# ╔═╡ c04943e8-9e68-42ba-8914-04b7a787a9d9
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 10, 42, 64, 64, 1, 2, 0)

# ╔═╡ 5b8408c3-5f9e-4055-b14c-a1c1f1c31835
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 10, 42, 128, 128, 1, 2, 0)

# ╔═╡ cf47b43e-0e2e-4a0d-8613-6a855b969a58
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 30, 42, 1, 1, 1, 3.3, 0)

# ╔═╡ 318272bb-09fc-4221-8936-d900f8481365
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 30, 42, 2, 2, 1, 3.3, 0)

# ╔═╡ 6fc0fb42-1ae3-4ca8-8783-47373d78ef68
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 30, 42, 4, 4, 1, 3.3, 0)

# ╔═╡ 23734fa4-bdfc-42f4-98a7-6dadf9df416a
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 30, 42, 8, 8, 1, 3.3, 0)

# ╔═╡ 0bd15fa0-24da-4f8a-a696-e7084102babd
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 30, 42, 16, 16, 1, 3.3, 0)

# ╔═╡ 921653b4-ea6a-4d73-8a6d-23fee688478a
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 30, 42, 32, 32, 1, 3.3, 0)

# ╔═╡ 1787f68f-794c-4989-a816-a3ec4e1f795f
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 30, 42, 64, 64, 1, 3.3, 0)

# ╔═╡ 5e896950-b5ce-46d4-bff9-0017bc5c10c0
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 30, 42, 128, 128, 1, 3.3, 0)

# ╔═╡ f7c141e1-f289-4eea-b7e9-7f16cbf3e7d6
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 30, 42, 1, 1, 1, 2, 0)

# ╔═╡ 63515003-e04b-4428-9204-de962e6a322d
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 30, 42, 2, 2, 1, 2, 0)

# ╔═╡ 5b4afaec-0f82-41a8-b344-39672c616a13
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 30, 42, 4, 4, 1, 2, 0)

# ╔═╡ 3c0fd8a3-d97f-4c3b-a34b-339526cd4ff6
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 30, 42, 8, 8, 1, 2, 0)

# ╔═╡ 80e717ce-95ed-4d38-b2a4-4f3f73890689
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 30, 42, 16, 16, 1, 2, 0)

# ╔═╡ 2e2dc9cf-2158-478c-9545-6173ab545f7d
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 30, 42, 32, 32, 1, 2, 0)

# ╔═╡ 228e2ca2-f599-40e1-98f6-2316066d1294
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 30, 42, 64, 64, 1, 2, 0)

# ╔═╡ 9f4a55fe-8d9f-4675-a049-389d81b56465
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 30, 42, 128, 128, 1, 2, 0)

# ╔═╡ 27670a58-ab73-49c4-987f-20130c1cae7d
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 90, 42, 1, 1, 1, 3.3, 0)

# ╔═╡ a6e7350e-bbe0-4389-a1dd-432cc1aaa3ec
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 90, 42, 2, 2, 1, 3.3, 0)

# ╔═╡ 51459344-23aa-4be0-83e5-8d6ed8306cb6
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 90, 42, 4, 4, 1, 3.3, 0)

# ╔═╡ 032403d3-b22a-45e6-94f2-a00ab532702e
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 90, 42, 8, 8, 1, 3.3, 0)

# ╔═╡ 7a875314-dec1-4e20-b2c5-35024c1f98e0
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 90, 42, 16, 16, 1, 3.3, 0)

# ╔═╡ edcb6161-b8e1-4abf-b476-3eac0fa8c191
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 90, 42, 32, 32, 1, 3.3, 0)

# ╔═╡ a5d67cfe-d317-4fdf-b9b8-5acd4fcb4bbe
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 90, 42, 64, 64, 1, 3.3, 0)

# ╔═╡ 3146a645-8730-468b-9403-98bed7dac435
#=╠═╡
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 90, 42, 128, 128, 1, 3.3, 0)
  ╠═╡ =#

# ╔═╡ ac130496-ea6c-497c-93d9-888703cfb38a
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 90, 42, 1, 1, 1, 2, 0)

# ╔═╡ 981a6723-6f70-4347-a9d7-2dcd303f7d2f
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 90, 42, 2, 2, 1, 2, 0)

# ╔═╡ 98802b7b-6855-43c6-9cb3-6124fe4b2479
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 90, 42, 4, 4, 1, 2, 0)

# ╔═╡ 55ff621d-2936-4f9b-95d1-043e8d641183
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 90, 42, 8, 8, 1, 2, 0)

# ╔═╡ f1b3707c-806f-48f4-a9d7-48548cce711c
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 90, 42, 16, 16, 1, 2, 0)

# ╔═╡ 4a8725da-e6a6-45bd-8910-9d89898bc13e
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 90, 42, 32, 32, 1, 2, 0)

# ╔═╡ fa754e4b-954a-48e9-8183-a9816acb545a
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 90, 42, 64, 64, 1, 2, 0)

# ╔═╡ fc3ff682-3f58-44cb-b97a-144e16886533
@time NormalizedRadonCDT.classify_data_NRCDT(temp_data, 128, 90, 42, 128, 128, 1, 2, 0)

# ╔═╡ Cell order:
# ╠═03f9300e-692e-11ef-145c-85ecce1e4c7f
# ╠═124e44b3-35a7-4269-a111-f1d0c6bdd40c
# ╠═47ffa7c2-dc44-4335-a314-840bcb743f54
# ╠═d4b1707b-2459-412e-9352-3ad75d8c0e91
# ╠═e5e3e712-8a02-4b8c-9a14-2cb6a2e7f0f8
# ╠═481e76c0-4a19-4516-a962-7f9d508bcf0e
# ╠═46f4270c-6f4b-4a07-83f1-0de077247d46
# ╠═30445f56-dcc2-421f-b163-7bd4eec372d8
# ╠═7d022f84-955b-4ab0-85e0-9141457fcb4c
# ╠═95dc4ee4-c519-4ef3-ace9-eb8b35dc33c8
# ╠═7fab2e41-9342-4c96-8aaf-43258aab4f66
# ╠═3f7c0e09-98ce-4e3c-8002-2e253c0aa281
# ╠═6e43e0fa-1ef8-4807-b80d-90c5fa5a72ec
# ╠═2e5c469a-8001-4a7b-8905-c017617e7f80
# ╠═2303943c-8db9-43e9-aa1c-d7e31b7bdf59
# ╠═72ff2bf5-2525-416f-a812-2b45a79dacd9
# ╠═c04943e8-9e68-42ba-8914-04b7a787a9d9
# ╠═5b8408c3-5f9e-4055-b14c-a1c1f1c31835
# ╠═cf47b43e-0e2e-4a0d-8613-6a855b969a58
# ╠═318272bb-09fc-4221-8936-d900f8481365
# ╠═6fc0fb42-1ae3-4ca8-8783-47373d78ef68
# ╠═23734fa4-bdfc-42f4-98a7-6dadf9df416a
# ╠═0bd15fa0-24da-4f8a-a696-e7084102babd
# ╠═921653b4-ea6a-4d73-8a6d-23fee688478a
# ╠═1787f68f-794c-4989-a816-a3ec4e1f795f
# ╠═5e896950-b5ce-46d4-bff9-0017bc5c10c0
# ╠═f7c141e1-f289-4eea-b7e9-7f16cbf3e7d6
# ╠═63515003-e04b-4428-9204-de962e6a322d
# ╠═5b4afaec-0f82-41a8-b344-39672c616a13
# ╠═3c0fd8a3-d97f-4c3b-a34b-339526cd4ff6
# ╠═80e717ce-95ed-4d38-b2a4-4f3f73890689
# ╠═2e2dc9cf-2158-478c-9545-6173ab545f7d
# ╠═228e2ca2-f599-40e1-98f6-2316066d1294
# ╠═9f4a55fe-8d9f-4675-a049-389d81b56465
# ╠═27670a58-ab73-49c4-987f-20130c1cae7d
# ╠═a6e7350e-bbe0-4389-a1dd-432cc1aaa3ec
# ╠═51459344-23aa-4be0-83e5-8d6ed8306cb6
# ╠═032403d3-b22a-45e6-94f2-a00ab532702e
# ╠═7a875314-dec1-4e20-b2c5-35024c1f98e0
# ╠═edcb6161-b8e1-4abf-b476-3eac0fa8c191
# ╠═a5d67cfe-d317-4fdf-b9b8-5acd4fcb4bbe
# ╠═3146a645-8730-468b-9403-98bed7dac435
# ╠═ac130496-ea6c-497c-93d9-888703cfb38a
# ╠═981a6723-6f70-4347-a9d7-2dcd303f7d2f
# ╠═98802b7b-6855-43c6-9cb3-6124fe4b2479
# ╠═55ff621d-2936-4f9b-95d1-043e8d641183
# ╠═f1b3707c-806f-48f4-a9d7-48548cce711c
# ╠═4a8725da-e6a6-45bd-8910-9d89898bc13e
# ╠═fa754e4b-954a-48e9-8183-a9816acb545a
# ╠═fc3ff682-3f58-44cb-b97a-144e16886533
