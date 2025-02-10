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

# ╔═╡ c2e723a2-fa6d-4884-8eb5-7713dd8fc3de
#temp_data = [1,12]
temp_data = [1,6]

# ╔═╡ c3b7783d-ce49-49d8-8eb4-77dfda088fa0
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 2, 1, 2.8, 0)

# ╔═╡ d7725e6d-ac09-4667-83d7-ea0f0133d024
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 4, 1, 2.8, 0)

# ╔═╡ d66ca2ab-96f9-4394-8667-d94fd37a09f4
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 8, 1, 2.8, 0)

# ╔═╡ e3a35fa8-dc42-45bf-9dcc-5732c20641b4
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 12, 1, 2.8, 0)

# ╔═╡ 3dd41ee6-7ca6-4603-ba03-eeb7300de767
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 24, 1, 2.8, 0)

# ╔═╡ f1bd1798-4bdd-4f0e-b49d-c2571b7f2aa4
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 2, 1, 2.3, 0)

# ╔═╡ 428e0fbb-1f0c-48e9-a3fe-7eb066210312
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 4, 1, 2.3, 0)

# ╔═╡ 9a043622-4a0f-4e10-912a-508296387568
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 8, 1, 2.3, 0)

# ╔═╡ 390c3026-b37e-44b7-91e7-c5a31b61f6c5
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 12, 1, 2.3, 0)

# ╔═╡ 56fe7096-6fe0-4c6e-a0fe-dc26f44cbfa2
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 24, 1, 2.3, 0)

# ╔═╡ dd369a52-37b3-4ef7-969a-135bee1041de
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 2, 1, 3.3, 0)

# ╔═╡ 59258ff4-39a8-4eb5-bb21-7f557eef777b
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 4, 1, 3.3, 0)

# ╔═╡ a04c23af-fa05-4107-9306-6b34a6488c58
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 8, 1, 3.3, 0)

# ╔═╡ e1ebe1d0-94c5-49d9-ac5b-6738dddf76b5
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 12, 1, 3.3, 0)

# ╔═╡ f98e6ade-7aae-4394-a868-a62fd0cb98e3
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 24, 1, 3.3, 0)

# ╔═╡ 9df3d920-a637-4eee-87b8-153c4674bfb6
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 2, 1, 2.3, 0)

# ╔═╡ 1e22e48c-ad82-4bd8-802b-3aa9c4f71999
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 4, 1, 2.3, 0)

# ╔═╡ a0efac01-1e43-406d-8a88-55dceea9ff21
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 8, 1, 2.3, 0)

# ╔═╡ d1b0ed2a-9de2-4afe-90b3-c75cd9551753
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 90, 42, 16, 1, 2.3, 0)

# ╔═╡ 8077e4b6-4019-4555-843e-e60aa9212c34
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 32, 1, 2.3, 0)

# ╔═╡ 2ce5228f-dfd3-4c4a-abb9-d00e792404fd
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 2, 1, 2, 0)

# ╔═╡ 8d6d15d3-09b8-4a93-b152-b948892e9041
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 4, 1, 2, 0)

# ╔═╡ 68c83926-365e-4ce7-a40a-8792de681976
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 8, 1, 2, 0)

# ╔═╡ 1c96e369-85d6-4b53-8bd5-b619a3f989fa
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 16, 1, 2, 0)

# ╔═╡ b7aab03a-7ebb-4989-b18d-d91d4d813659
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 32, 1, 2, 0)

# ╔═╡ d8a9f4af-d1a0-478e-af56-84d3a6b32488
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 128, 1, 2, 0)

# ╔═╡ 28d6645f-e944-45ff-bce9-874ad2442c34
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 2, 1, 3.3, 0)

# ╔═╡ 9fe2e42f-533f-4c06-b7bc-72748c314eb6
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 4, 1, 3.3, 0)

# ╔═╡ 7d525c2b-1051-4d10-b5cd-011fb4b5a7c3
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 8, 1, 3.3, 0)

# ╔═╡ 8fd1fd22-460d-4cc3-80dd-3dd66e1ad944
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 16, 1, 3.3, 0)

# ╔═╡ 43621b5f-90d1-41d4-8e0c-3e4245e3526d
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 32, 1, 3.3, 0)

# ╔═╡ dedac775-c369-42c7-bcd3-c35cc4610fc1
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 2, 1, 3, 0)

# ╔═╡ 2c9f1fb7-588a-4eda-b517-deffa0963e1d
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 4, 1, 3, 0)

# ╔═╡ f909b134-1d70-40b4-a748-4bc88a2f0b05
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 8, 1, 3, 0)

# ╔═╡ 73cd698b-5fbf-44cd-8716-91724ded1a88
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 128, 1, 3, 0)

# ╔═╡ f57b86e7-8ecc-494e-bdd9-dc9792786212
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 256, 1, 3, 0)

# ╔═╡ caa888a5-24a2-4ddd-a501-25d52bd31570
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 512, 1, 3, 0)

# ╔═╡ c829bcdb-d920-4ffc-874b-b7b4565db346
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 1024, 1, 3, 0)

# ╔═╡ Cell order:
# ╠═03f9300e-692e-11ef-145c-85ecce1e4c7f
# ╠═c2e723a2-fa6d-4884-8eb5-7713dd8fc3de
# ╠═c3b7783d-ce49-49d8-8eb4-77dfda088fa0
# ╠═d7725e6d-ac09-4667-83d7-ea0f0133d024
# ╠═d66ca2ab-96f9-4394-8667-d94fd37a09f4
# ╠═e3a35fa8-dc42-45bf-9dcc-5732c20641b4
# ╠═3dd41ee6-7ca6-4603-ba03-eeb7300de767
# ╠═f1bd1798-4bdd-4f0e-b49d-c2571b7f2aa4
# ╠═428e0fbb-1f0c-48e9-a3fe-7eb066210312
# ╠═9a043622-4a0f-4e10-912a-508296387568
# ╠═390c3026-b37e-44b7-91e7-c5a31b61f6c5
# ╠═56fe7096-6fe0-4c6e-a0fe-dc26f44cbfa2
# ╠═dd369a52-37b3-4ef7-969a-135bee1041de
# ╠═59258ff4-39a8-4eb5-bb21-7f557eef777b
# ╠═a04c23af-fa05-4107-9306-6b34a6488c58
# ╠═e1ebe1d0-94c5-49d9-ac5b-6738dddf76b5
# ╠═f98e6ade-7aae-4394-a868-a62fd0cb98e3
# ╠═9df3d920-a637-4eee-87b8-153c4674bfb6
# ╠═1e22e48c-ad82-4bd8-802b-3aa9c4f71999
# ╠═a0efac01-1e43-406d-8a88-55dceea9ff21
# ╠═d1b0ed2a-9de2-4afe-90b3-c75cd9551753
# ╠═8077e4b6-4019-4555-843e-e60aa9212c34
# ╠═2ce5228f-dfd3-4c4a-abb9-d00e792404fd
# ╠═8d6d15d3-09b8-4a93-b152-b948892e9041
# ╠═68c83926-365e-4ce7-a40a-8792de681976
# ╠═1c96e369-85d6-4b53-8bd5-b619a3f989fa
# ╠═b7aab03a-7ebb-4989-b18d-d91d4d813659
# ╠═d8a9f4af-d1a0-478e-af56-84d3a6b32488
# ╠═28d6645f-e944-45ff-bce9-874ad2442c34
# ╠═9fe2e42f-533f-4c06-b7bc-72748c314eb6
# ╠═7d525c2b-1051-4d10-b5cd-011fb4b5a7c3
# ╠═8fd1fd22-460d-4cc3-80dd-3dd66e1ad944
# ╠═43621b5f-90d1-41d4-8e0c-3e4245e3526d
# ╠═dedac775-c369-42c7-bcd3-c35cc4610fc1
# ╠═2c9f1fb7-588a-4eda-b517-deffa0963e1d
# ╠═f909b134-1d70-40b4-a748-4bc88a2f0b05
# ╠═73cd698b-5fbf-44cd-8716-91724ded1a88
# ╠═f57b86e7-8ecc-494e-bdd9-dc9792786212
# ╠═caa888a5-24a2-4ddd-a501-25d52bd31570
# ╠═c829bcdb-d920-4ffc-874b-b7b4565db346
