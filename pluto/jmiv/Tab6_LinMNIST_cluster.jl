### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 8cbe0300-edff-11ef-2fad-d3b8cca171a9
begin
	import Pkg
	Pkg.activate("../..")
	using Revise
	using NormalizedRadonCDT
	using NormalizedRadonCDT.TestImages
	using NormalizedRadonCDT.DataTransformations
	using NormalizedRadonCDT.Classify
	using Images
	using Plots
	using MLDatasets
	using JLD2
	using Random
	Random.seed!(42)
end

# ╔═╡ b3eac197-9352-4d18-b90d-675ba000adc1
using Clustering, MultivariateStats, Distances, Statistics, TSne

# ╔═╡ a8d8409d-72cb-4e42-832f-0cc142aa021f
md"""
# XXXX 2025 -- Table 6
This pluto notebook reproduces the numerical experiment
for Table 6 from

- Matthias Beckmann, Robert Beinert, Jonas Bresch, 
  'Normalized Radon Cummulative Distribution Transforms for Invariance and Robustness in Optimal Transport Based Image Classification',
  XXXX 2025.
"""

# ╔═╡ 0380ed85-ee40-47f8-9a12-689a0ad857f2
md"""
## Dataset
Load MNIST 
using `MLDatasets`.
Further information about MNIST can be found in

- L. Deng, 
  '[The MNIST database of handwritten digit images 
  for machine learning research]
  (https://doi.org/10.1109/MSP.2012.2211477)',
  *IEEE Signal Processing Magazine* **29**(6),
  141--142 (2012). 

!!! warning "Download MNIST"
	In order to load MNIST
	using `MLDatasets`,
	MNIST has to be downloaded first!
	For this,
	one can activate the enivironment of the project
	and load the dataset one time explicitly.
	Starting a Julia REPL 
	in the main directory of the project,
	this can be done by
	```
	import Pkg
	Pkg.activate()
	using MLDatasets
	MNIST(:train)
	```
	Julia then asks to download the corresponding dataset.
"""

# ╔═╡ 81783bfb-d7a2-4c18-a4f8-b634f3bbc59b
trainset = MNIST(:train)

# ╔═╡ 98131234-5ab0-4954-bd68-0646241ed22a
md"""
Generate LinMNIST
by selecting a subset of samples
and applying random affine transformations
using the submodule `DataTransformations`.
Further information about LinMNIST can be found in

- M. Beckmann, N. Heilenkötter,
  '[Equivariant neural networks 
  for indirect measurements]
  (https://doi.org/10.1137/23M1582862)',
  *SIAM Journal on Mathematics of Data Science* **6**(3),
  579--601 (2024).
"""

# ╔═╡ 81170d86-6140-41ce-a1e4-24e70c0530ff
MLClass, MLLabel = generate_ml_classes(trainset, [1, 5, 7], 100, 0.666);

# ╔═╡ bf8448db-7cb1-42ba-9f1e-03b775b31cb8
MLClass

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.5, 1.25), 
	scale_y = (0.5, 1.25),
	rotate=(-180.0, 180.0), 
	#shear_x=(-5.0, 5.0),
	#shear_y=(-5.0, 5.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
Random.seed!(42); TMLClass = A.(MLClass)

# ╔═╡ eff42224-7bdb-4d52-be3c-0fe19911cf2f
md"
# Setting the RCDT 
with 300 radii, 128 Radon angles, and 64 interpolation points.
The max- and mean-normalized RCDT are applied on the generated dataset.
"

# ╔═╡ 8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
R = RadonTransform(301,256,0.0) # (301,128), top

# ╔═╡ bbbcd04c-8b4f-4c44-958d-9e4089ada051
RCDT = RadonCDT(64, R)

# ╔═╡ 81fd55d8-24df-4047-b235-20468b2c111c
NRCDT = NormRadonCDT(RCDT)

# ╔═╡ a4860242-45bb-4f48-816c-3ed1bc82a9c6
rcdt = RCDT.(TMLClass);

# ╔═╡ 81e32395-78d9-4a5f-b6f0-ba2d6f01c8ee
mqClass = max_normalization.(rcdt)

# ╔═╡ 3a83ba1e-1565-49a0-94e1-f433a4762760
miqClass = maxmin_normalization.(rcdt)

# ╔═╡ 13849a92-c3d3-42e9-a959-ca456452aaeb
tvqClass = tv_normalization.(rcdt)

# ╔═╡ 0366e109-8b69-4b5e-b805-eb52bcc50aa2
maqClass = maxabs_normalization.(rcdt)

# ╔═╡ 2a0147ee-a92c-494a-94b8-bec39fd31a58
iaqClass = minabs_normalization.(rcdt)

# ╔═╡ a35feac1-880e-4a03-ac4a-f89904c83233
miaqClass = maxminabs_normalization.(rcdt)

# ╔═╡ c0499ac9-97b0-4ec9-bc0c-a6031b7fa223
md"""
## k-means Clustering -- Figure X
Use the k-means clustering algorithm from Clustering.jl
with respect to 2,...,6 fixed clusters 
on the (transformed) LinMNIST dataset.
The max-, max-min-normalized, and tv-normalized RCDT is applied.
"""

# ╔═╡ 570bf90e-0258-470c-90ef-a0ae53144a5f
md"
- using PCA reduction.
"

# ╔═╡ 2fa2178b-07c1-4617-90c2-8631ae99974f
MClass = reshape(vcat(Array{Float64}.(MLClass)...), :, length(MLClass));

# ╔═╡ 9ea0eb6a-9ce9-4863-adda-af47780cf2c7
MqQClass = reshape(vcat(rcdt...), :, length(rcdt));

# ╔═╡ 11dac364-a57a-4f0f-b534-2e730248223e
MtvQClass = reshape(vcat(tvqClass...), :, length(tvqClass));

# ╔═╡ 553a3e8d-ce84-4b82-8821-6927917b43e9
MmQClass = reshape(vcat(mqClass...), :, length(mqClass));

# ╔═╡ 74d09be3-2f2d-40ae-b8d5-c0cbdbd334ba
MmiQClass = reshape(vcat(miqClass...), :, length(miqClass));

# ╔═╡ 4650de34-d084-42f8-b74e-1ff2673fffdf
MmaQClass = reshape(vcat(maqClass...), :, length(maqClass));

# ╔═╡ 0eaf0f0a-8685-4b29-b666-69b1f747f223
MClass_pca = fit(PCA, MClass; maxoutdim=2);  # PCA expects n x d

# ╔═╡ 196a93d4-530d-4ce3-b881-9b57a27613d2
MqQClass_pca = fit(PCA, MqQClass; maxoutdim=2);  # PCA expects n x d

# ╔═╡ 18baf544-bb3f-40e4-9a21-7661aae925fa
MtvQClass_pca = fit(PCA, MtvQClass; maxoutdim=2);  # PCA expects n x d

# ╔═╡ 5a0a14bb-58d5-4c6b-a9ae-243eba459c60
MmQClass_pca = fit(PCA, MmQClass; maxoutdim=2);  # PCA expects n x d

# ╔═╡ d4e0e65e-322c-4b8d-95f8-6fa8e310d0cd
MmiQClass_pca = fit(PCA, MmiQClass; maxoutdim=2);  # PCA expects n x d

# ╔═╡ 727fbe9d-2ebf-4e17-9f37-175afa78b399
MmaQClass_pca = fit(PCA, MmaQClass; maxoutdim=2);  # PCA expects n x d

# ╔═╡ fda1a44f-1259-40bc-a16d-a205418bd0f6
MClass_reduced = transform(MClass_pca, MqQClass)'  # Now 2 x n

# ╔═╡ 82f04dcd-b40f-4e95-ae03-1d0ef962673e
MqQClass_reduced = transform(MqQClass_pca, MqQClass)'  # Now 2 x n

# ╔═╡ d187834a-5652-4761-b6ec-6dfb245195be
MtvQClass_reduced = transform(MtvQClass_pca, MtvQClass)'  # Now 2 x n

# ╔═╡ d49f0576-e17c-4aae-8086-59fbc89ea078
MmQClass_reduced = transform(MmQClass_pca, MmQClass)'  # Now 2 x n

# ╔═╡ ce620b29-5a21-4cd9-bc14-124e8031a2c7
MmiQClass_reduced = transform(MmiQClass_pca, MmiQClass)'  # Now 2 x n

# ╔═╡ 588571f4-f51a-4864-89a8-76d54f6719b4
MmaQClass_reduced = transform(MmaQClass_pca, MmaQClass)'  # Now 2 x n

# ╔═╡ 0b1cec64-e07b-404d-8b09-67ee616032ab
index=vcat(1:50,101:150,201:250)

# ╔═╡ 934b504c-b6b8-433e-90fc-6af35cd7053f
offindex=vcat(51:100,151:200,251:300)

# ╔═╡ e0ade989-78b0-40aa-aada-8791f688f164
RK = kmeans(MClass[:,index], 3, weights=ones(150))

# ╔═╡ e3691904-1ea1-423c-bfd9-0442d39290a3
RKq = kmeans(MqQClass[:,index], 3, weights=ones(150))

# ╔═╡ bc469388-9c04-48f9-89a4-76add5f37ce3
RKtv = kmeans(MtvQClass[:,index], 3, weights=ones(150))

# ╔═╡ 96176a20-6537-418a-b0a1-99d1485c7cf4
RKm = kmeans(MmQClass[:,index], 3, weights=ones(150))

# ╔═╡ f1997ae9-3d0d-4103-ae27-8bae7b0f357d
RKmi = kmeans(MmiQClass[:,index], 3, weights=ones(150))

# ╔═╡ b889bb2b-f924-48ef-9de5-0a8f3be0aef9
RKma = kmeans(MmaQClass[:,index], 3, weights=ones(150))

# ╔═╡ cb11c33c-3ae0-47a8-8903-3bce5aea8af4
rename_map = Dict(1 => 1, 2 => 5, 3 => 7)

# ╔═╡ b737a14f-62ba-4b77-9e85-7120913ede05
label_marker = Dict(1 => :circle, 5 => :star5, 7 => :utriangle)

# ╔═╡ 86e981a1-3c7b-4bd1-ad42-fbc9b025bfd6
label_color = Dict(1 => :blue, 5 => :red, 7 => :green)

# ╔═╡ 3d4c1de2-1ee6-4e6d-a2d0-acf80676d384
RK_re = [rename_map[x] for x in RK.assignments]

# ╔═╡ 04e5d165-5c91-4adb-b64d-869d48605cb1
RK_label_re = [label_marker[x] for x in MLLabel]

# ╔═╡ 75830189-02aa-43ea-8f6d-90b94b5e88f9
RK_color_re = [label_color[x] for x in RK_re]

# ╔═╡ 1de81b0f-934d-413f-ae77-0bb036b69027
ann_unique = [1,101,201]; ann_MLLabel = MLLabel[ann_unique]; ann_color = [:blue, :red, :green];

# ╔═╡ 08ab4693-f7ba-45ae-95a6-1cd56398a1c3
function dis_cluster(Class::AbstractArray, Center::AbstractArray)
	ss = size(Class)[2]
	qi = zeros(size(Class)[2])
	for i in 1:ss
		sc = size(Center)[2]
		qc = zeros(size(Center)[2])
		for j in 1:sc
			qc[j] = sqrt(sum(abs.(Class[:,i] .- Center[:,j]).^2))
		end
		qi[i] = argmin(qc)
	end
	return qi
end

# ╔═╡ 04f0f05c-e0d1-44b4-836a-cce9895aa67b
label_color_off = Dict(3.0 => :red, 1.0 => :blue, 2.0 => :green)

# ╔═╡ ab5c4caf-97aa-4a1b-9b9a-c95c8ff89d6b
label_off = Dict(3.0 => 3, 1.0 => 1, 2.0 => 2)

# ╔═╡ b8911df4-8b5c-4f66-9c86-64d322b099d6
RKmic = RKmi.centers;

# ╔═╡ 3cbed232-dfe5-45ca-a684-4ea7d6f5ee12
MmiQCenter_reduced = transform(MmiQClass_pca, RKmic)';

# ╔═╡ c70cc796-3870-434f-bf2e-9299ced4d60d
RKmic_dis = dis_cluster(MmiQClass[:,offindex],RKmic)

# ╔═╡ 7a49fbdf-661f-42f4-9350-e11aa5c52211
RKmac = RKma.centers;

# ╔═╡ 4e8db42b-43f3-4df1-aedb-3875580060ce
MmaQCenter_reduced = transform(MmaQClass_pca, RKmac)';

# ╔═╡ 0979c4e4-e87f-4b52-84ae-ca813a2c1dae
RKmac_dis = dis_cluster(MmaQClass[:,offindex],RKmac)

# ╔═╡ e1364e68-9e7d-4184-b31d-7c7a33164be1
RKtvc = RKtv.centers;

# ╔═╡ 586e0e3a-f5b3-4436-9340-a6335a932ce4
MtvQCenter_reduced = transform(MtvQClass_pca, RKtvc)';

# ╔═╡ 7821a10e-70ce-4e24-966d-e8efceb4dcf4
RKtvc_dis = dis_cluster(MtvQClass[:,offindex],RKtvc)

# ╔═╡ 3f016b1f-ddc0-45e7-9e63-d6ecbba5d405
RKmc = RKm.centers;

# ╔═╡ 034abe2c-b53c-4723-b9e8-3334554c85c9
MmQCenter_reduced = transform(MmQClass_pca, RKmc)';

# ╔═╡ 36eb3e16-83cf-4721-981f-868192084a96
RKmc_dis = dis_cluster(MmQClass[:,offindex],RKmc)

# ╔═╡ 1e39281b-a974-4558-9d3a-11e9394eaba3
RKqc = RKq.centers;

# ╔═╡ a7fe8c0c-20a5-43d1-9251-ee15926fbcc0
MqQCenter_reduced = transform(MqQClass_pca, RKqc)';

# ╔═╡ 9a90d2dc-c07d-43e0-ab25-fcd0108ae434
RKqc_dis = dis_cluster(MqQClass[:,offindex],RKqc)

# ╔═╡ 0e4a8927-6e64-4af4-85c1-4cb5272fde5b
RKc = RK.centers;

# ╔═╡ ab5d47c3-1983-4f29-b3e6-5c6a0f25542f
MCenter_reduced = transform(MClass_pca, RKc)';

# ╔═╡ 0e02c4b1-c7b2-437d-91f9-eda6b32dfde3
RKc_dis = dis_cluster(MClass[:,offindex],RKc)

# ╔═╡ bd71acbc-adac-41a0-8738-3b1c48254fc9
RK_color_off = [label_color_off[x] for x in RKc_dis]

# ╔═╡ d2107d93-e095-4945-b139-6bcea3d530cc
p = scatter(
	MClass_reduced[index,1], 
	MClass_reduced[index,2], 
	group=RK_re,
	color=RK_color_re,
	marker=RK_label_re[index], 
	fontfamily="Computer Modern",
	label=""); for i in 1:3
	scatter!([], [], marker=:square, color=ann_color[i], label="Label $(ann_MLLabel[i])"); scatter!(
	[], [],
	marker=RK_label_re[ann_unique[i]],
	color=:gray,
	label="Class $(ann_MLLabel[i])"
	)
	scatter!(MClass_reduced[offindex,1], 
	MClass_reduced[offindex,2], 
	color=RK_color_off,
	marker=RK_label_re[offindex], alpha=0.2, label="")
	#scatter!(MCenter_reduced[:,1], MCenter_reduced[:,2], marker=RK_label_re[[1,201,101]], color=:gray, label="")
end

# ╔═╡ 5486a162-442c-4565-9064-ebc14a7ad736
p

# ╔═╡ d5b37625-4129-4c05-a119-0e3aa1b9d319
# ╠═╡ disabled = true
#=╠═╡
savefig(p, "k-means-LinMNIST_157-eucl.pdf")
  ╠═╡ =#

# ╔═╡ 57624cf7-63ac-4633-aaa5-71993a221995
RK_off = [label_off[x] for x in RKc_dis]

# ╔═╡ 23d4b1e3-fff7-4557-95db-4712d0dd4867
md"
- quality measurements:
	- Rand index, RI (agreement probability) (↑): similarity between the two data clusterings, here the labelling from the k-means and the gound truth labelling.
	- Variation of information, VI (shared information distance) (↓): distance between the two clusterings.
	- V-measure, V_bM (↓): the harmonic mean of homogeneity and completeness of the clustering.
	- Average Silhouette Width, ASW (↑): correctness of point-to-cluster asssignment by comparing the distance of the point to its cluster and to the other clusters.
	- Dunn index, DI (↑): the ratio of the minimal distance between clusters and the maximal cluster diameter.
	- Davies-Bouldin index, DBI (↓): similarity between the cluster and the other most similar one, averaged over all clusters.
	- Xie-Beni index, XBI (↓): ratio betwen inertia within clusters and minimal distance between the cluster centers.
	- Calinski-Harabsz index, CHI (↑):corrected ratio of between cluster centers inertia and within-clusters inertia.
"

# ╔═╡ 9b1a18ca-c177-448a-b240-c151942356e6
distances_tv = pairwise(SqEuclidean(), MtvQClass);

# ╔═╡ 3d0b098d-6763-4183-88f8-f1cbc5ce6917
rename_l = Dict(1 => 1, 5 => 2, 7 => 3)

# ╔═╡ 385a949d-999d-4a60-a813-b30e88b758fe
MLLabels = [rename_l[x] for x in MLLabel[index]]

# ╔═╡ 36d4ef6b-90e5-42f4-8883-18c8f50aca6f
rename_p = Dict(2.0 => 2, 3.0 => 3, 1.0 => 1)

# ╔═╡ b265b053-444e-4267-bbf1-6f349f730152
RKc_diss = [rename_p[x] for x in RKc_dis]

# ╔═╡ 7bc33ff1-bbab-4830-831e-397fd3baa4f3
randindex(MLLabels,RK.assignments)

# ╔═╡ ecc584e5-834d-4ce6-a73e-b7b8bacbbc12
randindex(MLLabels,RKq.assignments)

# ╔═╡ 7c0799d4-0fc0-4271-9415-c1f66f257553
randindex(MLLabels,RKtv.assignments)

# ╔═╡ 5ee4bcbc-0316-4671-aaf8-ae1c8f0a3b7d
randindex(MLLabels,RKm.assignments)

# ╔═╡ 73c7e8d4-fd05-47d7-bbcd-2f000f222f9f
randindex(MLLabels,RKmi.assignments)

# ╔═╡ 57ab77c8-8ae2-482b-9919-e4588dbd22ce
randindex(MLLabels,RKma.assignments)

# ╔═╡ fc49725c-2cad-47ef-801f-3fb1f4905ae9
Clustering.varinfo(MLLabels,RKc_diss)#RK.assignments)

# ╔═╡ 84a71715-4ee6-40c0-8397-8ca06a4a7fb1
Clustering.varinfo(MLLabels,RKq.assignments)

# ╔═╡ c5c65067-0971-41ba-8be0-37b5ec874327
Clustering.varinfo(MLLabels,RKtv.assignments)

# ╔═╡ 0d646daf-5e5e-4f9a-95c8-eef291cccc7e
Clustering.varinfo(MLLabels,RKm.assignments)

# ╔═╡ 4c903525-80c7-4b33-b7fa-d4fbf674feaa
Clustering.varinfo(MLLabels,RKmi.assignments)

# ╔═╡ 5b46a97b-f72c-44a3-b6e1-51c080403d90
Clustering.varinfo(MLLabels,RKma.assignments)

# ╔═╡ 39fdce50-e10a-47b4-b1df-58fd81604a55
Clustering.vmeasure(MLLabels,RK.assignments), Clustering.vmeasure(MLLabels,RK.assignments, β=0.5), Clustering.vmeasure(MLLabels,RK.assignments, β=2.0)

# ╔═╡ e513b185-a092-4a31-bc0f-3e7b35cc38b8
Clustering.vmeasure(MLLabels,RKq.assignments), Clustering.vmeasure(MLLabels,RKq.assignments, β=0.5), Clustering.vmeasure(MLLabels,RKq.assignments, β=2.0)

# ╔═╡ 0e8b4919-af76-43bd-83d2-aab3cb85bef0
Clustering.vmeasure(MLLabels,RKtv.assignments), Clustering.vmeasure(MLLabels,RKtv.assignments, β=0.5), Clustering.vmeasure(MLLabels,RKtv.assignments, β=2.0)

# ╔═╡ bb87f60a-a367-40a9-9e8c-ffc0b465220d
Clustering.vmeasure(MLLabels,RKm.assignments), Clustering.vmeasure(MLLabels,RKm.assignments, β=0.5), Clustering.vmeasure(MLLabels,RKm.assignments, β=2.0)

# ╔═╡ 7f89a973-6197-4b2a-83e8-d13232c77e91
Clustering.vmeasure(MLLabels,RKmi.assignments), Clustering.vmeasure(MLLabels,RKmi.assignments, β=0.5), Clustering.vmeasure(MLLabels,RKmi.assignments, β=2.0)

# ╔═╡ c81e8250-27d5-4eb2-8250-10c95b759c96
clustering_quality(MClass, RK.centers, 
RK.assignments, quality_index=:silhouettes)

# ╔═╡ 894aba1c-710c-444b-a34f-076da2fbf0e2
clustering_quality(MqQClass, RKq.centers, 
RKq.assignments, quality_index=:silhouettes)

# ╔═╡ b26bb253-38cc-4d71-bd93-b9e9cbfce2c5
clustering_quality(MtvQClass, RKtv.centers, 
RKtv.assignments, quality_index=:silhouettes)

# ╔═╡ 1c3e34c0-6468-40ac-95a0-0fdbb17eb316
clustering_quality(MmQClass, RKm.centers, 
RKm.assignments, quality_index=:silhouettes)

# ╔═╡ 45232c1f-c8c9-4a71-ba65-396b8a18a775
clustering_quality(MmiQClass, RKmi.centers, 
RKmi.assignments, quality_index=:silhouettes)

# ╔═╡ 5ffa2174-ff49-4818-b9b2-8b12d17ce11e
clustering_quality(MClass, RK.centers, 
RK.assignments, quality_index=:dunn)

# ╔═╡ 9e07094c-7b85-4e53-a75e-579cdc1ca335
clustering_quality(MqQClass, RKq.centers, 
RKq.assignments, quality_index=:dunn)

# ╔═╡ 0aa000f7-823c-47ad-8999-7b178982b20b
clustering_quality(MtvQClass, RKtv.centers, 
RKtv.assignments, quality_index=:dunn)

# ╔═╡ 4d9c4d2c-87f8-4d1d-a0b8-b8d7fb3eec27
clustering_quality(MmQClass, RKm.centers, 
RKm.assignments, quality_index=:dunn)

# ╔═╡ 59f8f1df-f024-4da1-b805-594f18768106
clustering_quality(MmiQClass, RKmi.centers, 
RKmi.assignments, quality_index=:dunn)

# ╔═╡ ceb274e6-c2f4-4124-ae0c-7084e9ce1dfa
clustering_quality(MClass, RK.centers, RK.assignments, quality_index=:davies_bouldin)

# ╔═╡ df432ede-9992-4a10-b577-ba792a341418
clustering_quality(MqQClass, RKq.centers, RKq.assignments, quality_index=:davies_bouldin)

# ╔═╡ fc9fe287-de12-419c-956e-5db0a1dda77a
clustering_quality(MtvQClass, RKtv.centers, RKtv.assignments, quality_index=:davies_bouldin)

# ╔═╡ 8c153373-7d2e-4d38-a783-963b32f22729
clustering_quality(MmQClass, RKm.centers, RKm.assignments, quality_index=:davies_bouldin)

# ╔═╡ 7dc3c142-d791-425a-a58e-85dc287cdf8d
clustering_quality(MmiQClass, RKmi.centers, RKmi.assignments, quality_index=:davies_bouldin)

# ╔═╡ 7a9ab72a-6451-4aae-abb7-72878531dc46
clustering_quality(MClass, RK.centers, RK.assignments, quality_index=:xie_beni)

# ╔═╡ f2877f00-4175-4fab-92d8-36d46406a2a3
clustering_quality(MqQClass, RKq.centers, RKq.assignments, quality_index=:xie_beni)

# ╔═╡ 262c98d5-6b68-4a51-a6a8-fb93272a9211
clustering_quality(MtvQClass, RKtv.centers, RKtv.assignments, quality_index=:xie_beni)

# ╔═╡ ad37bded-e960-4f0b-99c5-6858d6750d98
clustering_quality(MmQClass, RKm.centers, RKm.assignments, quality_index=:xie_beni)

# ╔═╡ 69f01efc-6f39-4ba9-b899-8ee7862f90cb
clustering_quality(MmiQClass, RKmi.centers, RKmi.assignments, quality_index=:xie_beni)

# ╔═╡ 4d11169b-ea24-4da9-a057-702082756ab5
clustering_quality(MClass, RK.centers, RK.assignments, quality_index=:calinski_harabasz)

# ╔═╡ 457e07cf-38a5-4533-a41a-90c3ac590cbd
clustering_quality(MqQClass, RKq.centers, RKq.assignments, quality_index=:calinski_harabasz)

# ╔═╡ 7ada1b60-dd16-4569-90a7-fba51bc50788
clustering_quality(MtvQClass, RKtv.centers, RKtv.assignments, quality_index=:calinski_harabasz)

# ╔═╡ 60b2dbe3-9436-4afe-ac4f-4cfbb54f868a
clustering_quality(MmQClass, RKm.centers, RKm.assignments, quality_index=:calinski_harabasz)

# ╔═╡ 59e84514-13bc-4c2b-b9df-032393bd6fd8
clustering_quality(MmiQClass, RKmi.centers, RKmi.assignments, quality_index=:calinski_harabasz)

# ╔═╡ 3c6c8794-dcb8-43cc-95d8-1e52d655f708
md"Plots of the different quality measures over the k-means clustering results."

# ╔═╡ 1d4661a6-6276-4f63-8083-84fe071bc5c4
# ╠═╡ disabled = true
#=╠═╡
k_eucl = []; k_rcdt = []; k_tv = []; k_m = []; k_mi = [];
  ╠═╡ =#

# ╔═╡ fc5d0517-4d13-4131-8ebf-4844aa9fa814
# ╠═╡ disabled = true
#=╠═╡
lab = [:silhouettes, :dunn, :davies_bouldin, :xie_beni, :calinski_harabasz]
  ╠═╡ =#

# ╔═╡ 327b1783-e55d-41bd-ba5f-dc26df935c91
# ╠═╡ disabled = true
#=╠═╡
for k in [2,3,4,5,6]
	RK = kmeans(MClass, k, weights=ones(300))
	RKq = kmeans(MqQClass, k, weights=ones(300))
	RKtv = kmeans(MtvQClass, k, weights=ones(300))
	RKm = kmeans(MmQClass, k, weights=ones(300))
	RKmi = kmeans(MmiQClass, k, weights=ones(300))
	append!(k_eucl,clustering_quality(MClass, RK.centers, RK.assignments, quality_index=lab[1]))
	append!(k_rcdt,clustering_quality(MqQClass, RKq.centers, RKq.assignments, quality_index=lab[1]))
	append!(k_tv,clustering_quality(MtvQClass, RKtv.centers, RKtv.assignments, quality_index=lab[1]))
	append!(k_m,clustering_quality(MmQClass, RKm.centers, RKm.assignments, quality_index=lab[1]))
	append!(k_mi,clustering_quality(MmiQClass, RKmi.centers, RKmi.assignments, quality_index=lab[1]))
end
  ╠═╡ =#

# ╔═╡ 85ef57e1-7203-4781-8dec-1e0af35eb19d
# ╠═╡ disabled = true
#=╠═╡
using LaTeXStrings
  ╠═╡ =#

# ╔═╡ c55d5516-11ad-43de-af01-c20814b4beba
# ╠═╡ disabled = true
#=╠═╡
f = Plots.plot(k_eucl,label="Eucl."); Plots.plot!(k_rcdt, label="RCDT"); Plots.plot!(k_tv,label="tvNR-CDT"); Plots.plot!(k_m,label="mNR-CDT"); Plots.plot!(k_mi, xticks=(1:5, 2:6),label="miNR-CDT", yscale = :log10)
  ╠═╡ =#

# ╔═╡ c93d51d8-3407-4a7d-b74e-05cb73351027
# ╠═╡ disabled = true
#=╠═╡
savefig(f, "k-means-calinski_harabasz-LinMNIST_157.pdf")
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─a8d8409d-72cb-4e42-832f-0cc142aa021f
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╟─0380ed85-ee40-47f8-9a12-689a0ad857f2
# ╠═81783bfb-d7a2-4c18-a4f8-b634f3bbc59b
# ╟─98131234-5ab0-4954-bd68-0646241ed22a
# ╠═81170d86-6140-41ce-a1e4-24e70c0530ff
# ╠═bf8448db-7cb1-42ba-9f1e-03b775b31cb8
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╟─eff42224-7bdb-4d52-be3c-0fe19911cf2f
# ╠═8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
# ╠═bbbcd04c-8b4f-4c44-958d-9e4089ada051
# ╠═81fd55d8-24df-4047-b235-20468b2c111c
# ╠═a4860242-45bb-4f48-816c-3ed1bc82a9c6
# ╠═81e32395-78d9-4a5f-b6f0-ba2d6f01c8ee
# ╠═3a83ba1e-1565-49a0-94e1-f433a4762760
# ╠═13849a92-c3d3-42e9-a959-ca456452aaeb
# ╠═0366e109-8b69-4b5e-b805-eb52bcc50aa2
# ╠═2a0147ee-a92c-494a-94b8-bec39fd31a58
# ╠═a35feac1-880e-4a03-ac4a-f89904c83233
# ╟─c0499ac9-97b0-4ec9-bc0c-a6031b7fa223
# ╟─570bf90e-0258-470c-90ef-a0ae53144a5f
# ╠═b3eac197-9352-4d18-b90d-675ba000adc1
# ╠═2fa2178b-07c1-4617-90c2-8631ae99974f
# ╠═9ea0eb6a-9ce9-4863-adda-af47780cf2c7
# ╠═11dac364-a57a-4f0f-b534-2e730248223e
# ╠═553a3e8d-ce84-4b82-8821-6927917b43e9
# ╠═74d09be3-2f2d-40ae-b8d5-c0cbdbd334ba
# ╠═4650de34-d084-42f8-b74e-1ff2673fffdf
# ╠═0eaf0f0a-8685-4b29-b666-69b1f747f223
# ╠═196a93d4-530d-4ce3-b881-9b57a27613d2
# ╠═18baf544-bb3f-40e4-9a21-7661aae925fa
# ╠═5a0a14bb-58d5-4c6b-a9ae-243eba459c60
# ╠═d4e0e65e-322c-4b8d-95f8-6fa8e310d0cd
# ╠═727fbe9d-2ebf-4e17-9f37-175afa78b399
# ╠═fda1a44f-1259-40bc-a16d-a205418bd0f6
# ╠═82f04dcd-b40f-4e95-ae03-1d0ef962673e
# ╠═d187834a-5652-4761-b6ec-6dfb245195be
# ╠═d49f0576-e17c-4aae-8086-59fbc89ea078
# ╠═ce620b29-5a21-4cd9-bc14-124e8031a2c7
# ╠═588571f4-f51a-4864-89a8-76d54f6719b4
# ╠═0b1cec64-e07b-404d-8b09-67ee616032ab
# ╠═934b504c-b6b8-433e-90fc-6af35cd7053f
# ╠═e0ade989-78b0-40aa-aada-8791f688f164
# ╠═e3691904-1ea1-423c-bfd9-0442d39290a3
# ╠═bc469388-9c04-48f9-89a4-76add5f37ce3
# ╠═96176a20-6537-418a-b0a1-99d1485c7cf4
# ╠═f1997ae9-3d0d-4103-ae27-8bae7b0f357d
# ╠═b889bb2b-f924-48ef-9de5-0a8f3be0aef9
# ╠═cb11c33c-3ae0-47a8-8903-3bce5aea8af4
# ╠═b737a14f-62ba-4b77-9e85-7120913ede05
# ╠═86e981a1-3c7b-4bd1-ad42-fbc9b025bfd6
# ╠═3d4c1de2-1ee6-4e6d-a2d0-acf80676d384
# ╠═04e5d165-5c91-4adb-b64d-869d48605cb1
# ╠═75830189-02aa-43ea-8f6d-90b94b5e88f9
# ╠═1de81b0f-934d-413f-ae77-0bb036b69027
# ╠═d2107d93-e095-4945-b139-6bcea3d530cc
# ╠═5486a162-442c-4565-9064-ebc14a7ad736
# ╠═d5b37625-4129-4c05-a119-0e3aa1b9d319
# ╠═08ab4693-f7ba-45ae-95a6-1cd56398a1c3
# ╠═04f0f05c-e0d1-44b4-836a-cce9895aa67b
# ╠═ab5c4caf-97aa-4a1b-9b9a-c95c8ff89d6b
# ╠═bd71acbc-adac-41a0-8738-3b1c48254fc9
# ╠═57624cf7-63ac-4633-aaa5-71993a221995
# ╠═b8911df4-8b5c-4f66-9c86-64d322b099d6
# ╠═3cbed232-dfe5-45ca-a684-4ea7d6f5ee12
# ╠═c70cc796-3870-434f-bf2e-9299ced4d60d
# ╠═7a49fbdf-661f-42f4-9350-e11aa5c52211
# ╠═4e8db42b-43f3-4df1-aedb-3875580060ce
# ╠═0979c4e4-e87f-4b52-84ae-ca813a2c1dae
# ╠═e1364e68-9e7d-4184-b31d-7c7a33164be1
# ╠═586e0e3a-f5b3-4436-9340-a6335a932ce4
# ╠═7821a10e-70ce-4e24-966d-e8efceb4dcf4
# ╠═3f016b1f-ddc0-45e7-9e63-d6ecbba5d405
# ╠═034abe2c-b53c-4723-b9e8-3334554c85c9
# ╠═36eb3e16-83cf-4721-981f-868192084a96
# ╠═1e39281b-a974-4558-9d3a-11e9394eaba3
# ╠═a7fe8c0c-20a5-43d1-9251-ee15926fbcc0
# ╠═9a90d2dc-c07d-43e0-ab25-fcd0108ae434
# ╠═0e4a8927-6e64-4af4-85c1-4cb5272fde5b
# ╠═ab5d47c3-1983-4f29-b3e6-5c6a0f25542f
# ╠═0e02c4b1-c7b2-437d-91f9-eda6b32dfde3
# ╟─23d4b1e3-fff7-4557-95db-4712d0dd4867
# ╠═9b1a18ca-c177-448a-b240-c151942356e6
# ╠═3d0b098d-6763-4183-88f8-f1cbc5ce6917
# ╠═385a949d-999d-4a60-a813-b30e88b758fe
# ╠═36d4ef6b-90e5-42f4-8883-18c8f50aca6f
# ╠═b265b053-444e-4267-bbf1-6f349f730152
# ╠═7bc33ff1-bbab-4830-831e-397fd3baa4f3
# ╠═ecc584e5-834d-4ce6-a73e-b7b8bacbbc12
# ╠═7c0799d4-0fc0-4271-9415-c1f66f257553
# ╠═5ee4bcbc-0316-4671-aaf8-ae1c8f0a3b7d
# ╠═73c7e8d4-fd05-47d7-bbcd-2f000f222f9f
# ╠═57ab77c8-8ae2-482b-9919-e4588dbd22ce
# ╠═fc49725c-2cad-47ef-801f-3fb1f4905ae9
# ╠═84a71715-4ee6-40c0-8397-8ca06a4a7fb1
# ╠═c5c65067-0971-41ba-8be0-37b5ec874327
# ╠═0d646daf-5e5e-4f9a-95c8-eef291cccc7e
# ╠═4c903525-80c7-4b33-b7fa-d4fbf674feaa
# ╠═5b46a97b-f72c-44a3-b6e1-51c080403d90
# ╠═39fdce50-e10a-47b4-b1df-58fd81604a55
# ╠═e513b185-a092-4a31-bc0f-3e7b35cc38b8
# ╠═0e8b4919-af76-43bd-83d2-aab3cb85bef0
# ╠═bb87f60a-a367-40a9-9e8c-ffc0b465220d
# ╠═7f89a973-6197-4b2a-83e8-d13232c77e91
# ╠═c81e8250-27d5-4eb2-8250-10c95b759c96
# ╠═894aba1c-710c-444b-a34f-076da2fbf0e2
# ╠═b26bb253-38cc-4d71-bd93-b9e9cbfce2c5
# ╠═1c3e34c0-6468-40ac-95a0-0fdbb17eb316
# ╠═45232c1f-c8c9-4a71-ba65-396b8a18a775
# ╠═5ffa2174-ff49-4818-b9b2-8b12d17ce11e
# ╠═9e07094c-7b85-4e53-a75e-579cdc1ca335
# ╠═0aa000f7-823c-47ad-8999-7b178982b20b
# ╠═4d9c4d2c-87f8-4d1d-a0b8-b8d7fb3eec27
# ╠═59f8f1df-f024-4da1-b805-594f18768106
# ╠═ceb274e6-c2f4-4124-ae0c-7084e9ce1dfa
# ╠═df432ede-9992-4a10-b577-ba792a341418
# ╠═fc9fe287-de12-419c-956e-5db0a1dda77a
# ╠═8c153373-7d2e-4d38-a783-963b32f22729
# ╠═7dc3c142-d791-425a-a58e-85dc287cdf8d
# ╠═7a9ab72a-6451-4aae-abb7-72878531dc46
# ╠═f2877f00-4175-4fab-92d8-36d46406a2a3
# ╠═262c98d5-6b68-4a51-a6a8-fb93272a9211
# ╠═ad37bded-e960-4f0b-99c5-6858d6750d98
# ╠═69f01efc-6f39-4ba9-b899-8ee7862f90cb
# ╠═4d11169b-ea24-4da9-a057-702082756ab5
# ╠═457e07cf-38a5-4533-a41a-90c3ac590cbd
# ╠═7ada1b60-dd16-4569-90a7-fba51bc50788
# ╠═60b2dbe3-9436-4afe-ac4f-4cfbb54f868a
# ╠═59e84514-13bc-4c2b-b9df-032393bd6fd8
# ╟─3c6c8794-dcb8-43cc-95d8-1e52d655f708
# ╠═1d4661a6-6276-4f63-8083-84fe071bc5c4
# ╠═fc5d0517-4d13-4131-8ebf-4844aa9fa814
# ╠═327b1783-e55d-41bd-ba5f-dc26df935c91
# ╠═85ef57e1-7203-4781-8dec-1e0af35eb19d
# ╠═c55d5516-11ad-43de-af01-c20814b4beba
# ╠═c93d51d8-3407-4a7d-b74e-05cb73351027
