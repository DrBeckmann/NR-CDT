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

# ╔═╡ fef061d3-cd6e-45e4-afa8-8e2bd8a17395
NormalizedRadonCDT.generator_temp(1, 1, 1, 1, [2, 4, 7])

# ╔═╡ 4d9195e5-5e6e-4932-8b3b-992403cef7b2
NormalizedRadonCDT.view_temp(1, 1, 1, 1, [2, 4, 7])

# ╔═╡ 21c0724b-ed8c-468f-a45e-6453f6b8cc9f
data_size = 128; samp_size = 100; random_seed = 42; choice_of_temp_images = [2, 5, 8, 11];

# ╔═╡ 8c20527b-ca66-4333-82eb-0b3db48523f6
NormalizedRadonCDT.create_data(choice_of_temp_images, data_size, samp_size, random_seed)

# ╔═╡ cf53aba0-b047-4291-b9e4-9972c522035c
NormalizedRadonCDT.view_data()

# ╔═╡ Cell order:
# ╠═03f9300e-692e-11ef-145c-85ecce1e4c7f
# ╠═fef061d3-cd6e-45e4-afa8-8e2bd8a17395
# ╠═4d9195e5-5e6e-4932-8b3b-992403cef7b2
# ╠═21c0724b-ed8c-468f-a45e-6453f6b8cc9f
# ╠═8c20527b-ca66-4333-82eb-0b3db48523f6
# ╠═cf53aba0-b047-4291-b9e4-9972c522035c
