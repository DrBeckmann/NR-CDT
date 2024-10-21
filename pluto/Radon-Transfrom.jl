### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 2c63867e-638a-11ef-3da0-a91c57c21253
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.activate("..")
	using NormalizedRadonCDT.TestImages
	using NormalizedRadonCDT
	using NormalizedRadonCDT.transformation
	using Plots
end

# ╔═╡ bde12216-84b7-4029-808d-b184b6aae11e
md"""
# Radon Transform
The file *transfroms* allows to compute the a *exact* Radon transformation 
for images, or densities. 
There are two ways to compute such a transormation:
- on the line,
- on a tube around the line with a width.
"""

# ╔═╡ 812fa343-4a31-4f2c-a844-3f242960cfb7
md"""
The Radon transform w.r.t. to
- an angle ``\theta \in \mathbb S_1`` paranetrized by at least ``\theta \in [0, \pi]`` and 
- a radius ``t \in [-1, 1] := \mathbb I``
is defined for ``L^p`` functions 
``\mathcal R : L^p(\mathbb B_2) \to L^p(\mathbb I \times \mathbb S_1),``
where 
``\mathcal R f : \mathbb I \times \mathbb S_1 \to \mathbb R`` 
is defined as the line integral 

 ``\mathcal R f (t, \theta) 
    \coloneqq 
    \int_{C_\theta^t} f(s) \; d s,``

where 
- ``d s`` denotes the arg length of the line 
- ``C_\theta^t := \{x \in \mathbb R^2 : S_\theta(x) = t\}`` and 
- ``S_\theta(x) := \langle x, \theta\rangle.``
"""

# ╔═╡ 28f16420-6b87-43cb-b807-b4dd8ac908ff
md"""
## Examples
We use the *basic* shapes from our module *TestImages* to perform 
one exmaple for each implementation:
- the line and
- the tube 
based computation of the Radon transformation.
Notably,
for the limit width ``\rightarrow 0``,
we obtain the line based Radon transformation.
"""

# ╔═╡ 1aabbfed-2b4f-4acb-a287-eddb9f8497e5
image = render(OrbAndCross(Circle(), Star(10)));

# ╔═╡ 15185324-7336-4c5b-860c-37895249ff5c
heatmap(image)

# ╔═╡ c5fe52fe-46f3-4fa2-8077-cd0c8ee99907
sino_width = NormalizedRadonCDT.transformation.radon(image, 100, 100, 0.05);

# ╔═╡ 4e0ac30a-2ca4-483e-b2ea-b1548a8b9e3f
sino = NormalizedRadonCDT.transformation.radon(image,100,100,0);

# ╔═╡ 6afd8345-f6a1-41e4-a50e-179049e30a84
md"""
The Radon transformation of the image from above
with 100 angles and 100 radii, both equidistant, and width = 0.1.
"""

# ╔═╡ 58989d05-1fd6-4866-95e4-e4ab5a51ca68
heatmap(sino_width)

# ╔═╡ fc548d1c-0d46-433d-b0d6-5a82e3400f74
md"""
The *exact* Radon transformation of the image from above
with 100 angles and 100 radii, both equidistant.
"""

# ╔═╡ adef3075-6201-4da8-b4a4-7a3f37e5a2e4
heatmap(sino)

# ╔═╡ Cell order:
# ╟─bde12216-84b7-4029-808d-b184b6aae11e
# ╟─812fa343-4a31-4f2c-a844-3f242960cfb7
# ╠═2c63867e-638a-11ef-3da0-a91c57c21253
# ╟─28f16420-6b87-43cb-b807-b4dd8ac908ff
# ╠═1aabbfed-2b4f-4acb-a287-eddb9f8497e5
# ╟─15185324-7336-4c5b-860c-37895249ff5c
# ╠═c5fe52fe-46f3-4fa2-8077-cd0c8ee99907
# ╠═4e0ac30a-2ca4-483e-b2ea-b1548a8b9e3f
# ╟─6afd8345-f6a1-41e4-a50e-179049e30a84
# ╟─58989d05-1fd6-4866-95e4-e4ab5a51ca68
# ╟─fc548d1c-0d46-433d-b0d6-5a82e3400f74
# ╟─adef3075-6201-4da8-b4a4-7a3f37e5a2e4
