### A Pluto.jl notebook ###
# v0.19.45

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

# ╔═╡ cf47b43e-0e2e-4a0d-8613-6a855b969a58
NormalizedRadonCDT.classify_data_NRCDT([1,5], 128, 100, 42, 32, 2, 5, 0.0)

# ╔═╡ 40184a37-9244-49ab-8139-00439864f976
NormalizedRadonCDT.classify_data_NRCDT([1,5], 128, 200, 42, 32, 32, 5, 0.001)

# ╔═╡ Cell order:
# ╠═03f9300e-692e-11ef-145c-85ecce1e4c7f
# ╠═cf47b43e-0e2e-4a0d-8613-6a855b969a58
# ╠═40184a37-9244-49ab-8139-00439864f976
