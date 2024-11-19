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
temp_data = [1,6]

# ╔═╡ c3b7783d-ce49-49d8-8eb4-77dfda088fa0
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 2, 1, 0, 0)

# ╔═╡ 572c96a1-bc8e-467d-a051-2051db6e4080
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 2, 1, 0.5, 0)

# ╔═╡ 932daed7-3788-463b-8d00-39f6983c7c7b
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 4, 1, 0, 0)

# ╔═╡ 7dbf1d48-a9c9-476c-a2db-1d42f0340c69
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 4, 1, 0.5, 0)

# ╔═╡ e0a85ddd-2876-438b-9d90-1a19a3f352a5
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 8, 1, 0, 0)

# ╔═╡ 51093cb7-ca31-4754-9d3a-e97bf755b4a2
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 8, 1, 0.5, 0)

# ╔═╡ 576b6bae-b49c-45bd-99a8-a1faa7fdc581
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 16, 1, 0, 0)

# ╔═╡ b1deaa37-ba9a-49f9-9b17-7cfe6011759b
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 16, 1, 0.5, 0)

# ╔═╡ 08946800-ff02-4353-989f-2c2f2597892f
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 32, 1, 0, 0)

# ╔═╡ b608a98a-35cc-4380-8c25-ec9fab907cc2
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 32, 1, 0.5, 0)

# ╔═╡ 20d2706c-9a62-486e-8b9d-4484e1c5f09e
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 64, 1, 0, 0)

# ╔═╡ 8c2186b4-9fe0-49c7-a53d-f4e358d7fea3
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 64, 1, 0.5, 0)

# ╔═╡ 2cc6b472-7070-46cd-b160-364d184f4de6
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 128, 1, 0, 0)

# ╔═╡ cf47b43e-0e2e-4a0d-8613-6a855b969a58
NormalizedRadonCDT.nearest_data_NRCDT(temp_data, 128, 10, 42, 128, 1, 0.5, 0)

# ╔═╡ Cell order:
# ╠═03f9300e-692e-11ef-145c-85ecce1e4c7f
# ╠═c2e723a2-fa6d-4884-8eb5-7713dd8fc3de
# ╠═c3b7783d-ce49-49d8-8eb4-77dfda088fa0
# ╠═572c96a1-bc8e-467d-a051-2051db6e4080
# ╠═932daed7-3788-463b-8d00-39f6983c7c7b
# ╠═7dbf1d48-a9c9-476c-a2db-1d42f0340c69
# ╠═e0a85ddd-2876-438b-9d90-1a19a3f352a5
# ╠═51093cb7-ca31-4754-9d3a-e97bf755b4a2
# ╠═576b6bae-b49c-45bd-99a8-a1faa7fdc581
# ╠═b1deaa37-ba9a-49f9-9b17-7cfe6011759b
# ╠═08946800-ff02-4353-989f-2c2f2597892f
# ╠═b608a98a-35cc-4380-8c25-ec9fab907cc2
# ╠═20d2706c-9a62-486e-8b9d-4484e1c5f09e
# ╠═8c2186b4-9fe0-49c7-a53d-f4e358d7fea3
# ╠═2cc6b472-7070-46cd-b160-364d184f4de6
# ╠═cf47b43e-0e2e-4a0d-8613-6a855b969a58
