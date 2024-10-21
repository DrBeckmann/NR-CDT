### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 03f9300e-692e-11ef-145c-85ecce1e4c7f
begin
	import Pkg
	Pkg.activate("..")
	using Revise
	using FFTW
	using NormalizedRadonCDT.TestImages
	using NormalizedRadonCDT
end

# ╔═╡ b552ec65-4b9a-4a2e-bc7c-3673a0b53c7f
using Plots

# ╔═╡ 58dd3650-4530-49ac-92de-4c8e5b3e7c2d


# ╔═╡ 15bf2e2e-e5b7-40a8-ac0d-647bec88922e


# ╔═╡ 9307f2c6-5777-4530-a0b5-6a075a0dc2bb
image = render(OrbAndCross(Triangle(), Star(7)))

# ╔═╡ 027a67d2-b8ae-49e3-b485-7b70691888bf
sino = NormalizedRadonCDT.radon(image, 200, 200);

# ╔═╡ 7261ffe0-963b-4cd7-9661-a47c7742874e
heatmap(sino)

# ╔═╡ 9faaf967-491f-4f4a-9b74-b12d2a1913b7
adj_sino = NormalizedRadonCDT.adj_radon(sino, 1000, 1000);

# ╔═╡ 9af64079-e7b1-46a8-8404-4f2a6b20d41c
heatmap(adj_sino)

# ╔═╡ 047c80be-3f2e-424c-ac83-4ec3726efd97
inv_sino = NormalizedRadonCDT.iradon(sino, 150, 150);

# ╔═╡ ad1dba23-58f6-42bb-97a6-50bf11d372e8
heatmap(inv_sino)

# ╔═╡ 2e6e97ed-0be0-413e-a180-0b96340e4b9d
iinv_sino = inv_sino[inv_sino > 0]

# ╔═╡ Cell order:
# ╠═58dd3650-4530-49ac-92de-4c8e5b3e7c2d
# ╠═03f9300e-692e-11ef-145c-85ecce1e4c7f
# ╠═15bf2e2e-e5b7-40a8-ac0d-647bec88922e
# ╠═9307f2c6-5777-4530-a0b5-6a075a0dc2bb
# ╠═027a67d2-b8ae-49e3-b485-7b70691888bf
# ╠═b552ec65-4b9a-4a2e-bc7c-3673a0b53c7f
# ╠═7261ffe0-963b-4cd7-9661-a47c7742874e
# ╠═9faaf967-491f-4f4a-9b74-b12d2a1913b7
# ╠═9af64079-e7b1-46a8-8404-4f2a6b20d41c
# ╠═047c80be-3f2e-424c-ac83-4ec3726efd97
# ╟─ad1dba23-58f6-42bb-97a6-50bf11d372e8
# ╠═2e6e97ed-0be0-413e-a180-0b96340e4b9d
