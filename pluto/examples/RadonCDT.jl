### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 2c63867e-638a-11ef-3da0-a91c57c21253
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.activate("../..")
	using NormalizedRadonCDT
	using NormalizedRadonCDT.TestImages
	using Plots
end

# ╔═╡ bde12216-84b7-4029-808d-b184b6aae11e
md"""
# Radon-based Transformations
The package `NormalizedRadonCDT` deals with the computation of
- the Radon transform
- the Radon cumulative distribution transform (RCDT)
- the normalized RCDT
- the max-normalized RCDT
- the mean-normalized RCDT
for gray-valued images.
All Transformations are implemented as callable `struct`.
"""

# ╔═╡ 85d57137-6655-44ca-afa8-248512d12182
md"""
## Test Image
As test image,
we use the orb-and-cross shape 
from the submodule `TestImages`. 
"""

# ╔═╡ 1aabbfed-2b4f-4acb-a287-eddb9f8497e5
I = render(OrbAndCross(Circle(), Star(6)))

# ╔═╡ 812fa343-4a31-4f2c-a844-3f242960cfb7
md"""
## Radon Transform
For angle ``\vartheta \in [0, \pi]`` 
and radius ``t \in [-1, 1]``,
the *Radon transform* 
of a function ``f \colon \mathbb R^2 \to \mathbb R``
is defined as the line integral 
```math
	\mathcal R f (t, \vartheta) 
    \coloneqq 
    \int_{\ell_{\theta,\vartheta}} f(s) \; \mathrm d s,
	\qquad
	\ell_{t,\vartheta}
	\coloneqq
	\{x \in \mathbb R^2 
	\mid \langle x, (\cos \vartheta, -\sin \vartheta) \rangle = t\},
```
where ``\mathrm d s`` denotes the arg element of the line.
The Radon transform 
for a fixed number of equispaced angles and radii
is initialized via `RadonTransform`.
"""

# ╔═╡ b8e3a3ff-253e-4cad-94b4-a8d89aab97ee
R = RadonTransform(100, 100)

# ╔═╡ e2cccf7d-5a15-4c84-977c-4b968688e9d3
md"""
Since `RadonTransform` is a callable `struct`,
it can be applied to the test image.
"""

# ╔═╡ c5fe52fe-46f3-4fa2-8077-cd0c8ee99907
heatmap(R(I))

# ╔═╡ 28f16420-6b87-43cb-b807-b4dd8ac908ff
md"""
## RCDT
The RCDT is a combination of the Radon transform
and the cumulative distribution transform,
which is applied to each column
of the sinogram.
For a fixed number of samples of the quantile functions,
and for a given Radon transform,
the RCDT is initialized using `RadonCDT`.
"""

# ╔═╡ 4e0ac30a-2ca4-483e-b2ea-b1548a8b9e3f
RCDT = RadonCDT(100, R)

# ╔═╡ 6afd8345-f6a1-41e4-a50e-179049e30a84
md"""
The RCDT is implemented as callable `struct`
and can be applied to an image.
"""

# ╔═╡ 58989d05-1fd6-4866-95e4-e4ab5a51ca68
heatmap(RCDT(I))

# ╔═╡ fc548d1c-0d46-433d-b0d6-5a82e3400f74
md"""
## Normalized RCDT
The normalized RCDT applies an additional affine transformation
to the computed RCDT
such that
each quantile has mean zero and standard deviation.
For a given RCDT,
the normalized RCDT is initialized using `NormRadonCDT`.
"""

# ╔═╡ adef3075-6201-4da8-b4a4-7a3f37e5a2e4
NRCDT = NormRadonCDT(RCDT)

# ╔═╡ 42ff5bee-dc93-47b4-9529-f7a98d21bb4f
md"""
The normalized RCDT is implemented as callable `struct`
and can be applied to an image.
"""

# ╔═╡ decb055d-83ba-4618-8760-7dabd9a82386
heatmap(NRCDT(I))

# ╔═╡ 242ec439-1111-4e07-9022-61f2a05dbda2
md"""
## Max-Normalized RCDT
The max-normalized RCDT computes the maximum 
over all normalized quantile functions.
The implemantation and usage is simular to the normalized RCDT.
"""

# ╔═╡ 0ab1cff1-2445-49f6-a52c-b0ac889667e2
mNRCDT = MaxNormRadonCDT(RCDT)

# ╔═╡ 2c303a08-aa30-4579-91e8-ba526ab25667
plot(mNRCDT(I))

# ╔═╡ 1d6bc808-0178-4936-8150-7faea0f6d411
md"""
## Mean-Normalized RCDT
The mean-normalized RCDT computes the mean 
over all normalized quantile functions.
The implemantation and usage is simular to the normalized RCDT.
"""

# ╔═╡ 56ce610a-ae83-4c96-b7c3-ec9691ae110a
aNRCDT = MeanNormRadonCDT(RCDT)

# ╔═╡ 6125f9c3-5c65-475a-98f8-40d409400f4d
plot(aNRCDT(I))

# ╔═╡ Cell order:
# ╟─bde12216-84b7-4029-808d-b184b6aae11e
# ╠═2c63867e-638a-11ef-3da0-a91c57c21253
# ╟─85d57137-6655-44ca-afa8-248512d12182
# ╠═1aabbfed-2b4f-4acb-a287-eddb9f8497e5
# ╟─812fa343-4a31-4f2c-a844-3f242960cfb7
# ╠═b8e3a3ff-253e-4cad-94b4-a8d89aab97ee
# ╟─e2cccf7d-5a15-4c84-977c-4b968688e9d3
# ╠═c5fe52fe-46f3-4fa2-8077-cd0c8ee99907
# ╟─28f16420-6b87-43cb-b807-b4dd8ac908ff
# ╠═4e0ac30a-2ca4-483e-b2ea-b1548a8b9e3f
# ╟─6afd8345-f6a1-41e4-a50e-179049e30a84
# ╠═58989d05-1fd6-4866-95e4-e4ab5a51ca68
# ╟─fc548d1c-0d46-433d-b0d6-5a82e3400f74
# ╠═adef3075-6201-4da8-b4a4-7a3f37e5a2e4
# ╟─42ff5bee-dc93-47b4-9529-f7a98d21bb4f
# ╠═decb055d-83ba-4618-8760-7dabd9a82386
# ╟─242ec439-1111-4e07-9022-61f2a05dbda2
# ╠═0ab1cff1-2445-49f6-a52c-b0ac889667e2
# ╠═2c303a08-aa30-4579-91e8-ba526ab25667
# ╟─1d6bc808-0178-4936-8150-7faea0f6d411
# ╠═56ce610a-ae83-4c96-b7c3-ec9691ae110a
# ╠═6125f9c3-5c65-475a-98f8-40d409400f4d
