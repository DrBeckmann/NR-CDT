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
	using JLD2
	using Random
	Random.seed!(42)
end

# ╔═╡ b3eac197-9352-4d18-b90d-675ba000adc1
using Clustering, MultivariateStats, Distances, Statistics, TSne

# ╔═╡ 85ef57e1-7203-4781-8dec-1e0af35eb19d
# ╠═╡ disabled = true
#=╠═╡
using LaTeXStrings
  ╠═╡ =#

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
"""

# ╔═╡ f1fc2e9a-2d65-421d-90da-6a9d28959dac
I₁ = render(OrbAndCross(Circle(),Star(1)), width=8);

# ╔═╡ da811304-9c01-47a9-8465-065dba6ec1fc
I₁₂ = render(OrbAndCross(Circle(),Star(4)), width=8);

# ╔═╡ e26b8000-9bc6-437f-b7b4-8c1b3883538f
I₁₃ = render(OrbAndCross(Circle(),Star(8)), width=8);

# ╔═╡ d71ce2b5-87a9-41ab-a5d7-030bc2da6b5e
J₁₁ = extend_image(I₁, (256, 256))

# ╔═╡ d7fb239f-26cc-4929-b917-253390965932
J₁₂ = extend_image(I₁₂, (256, 256))

# ╔═╡ 0300491a-aa9c-43f0-933f-cb3f63a84f03
J₁₃ = extend_image(I₁₃, (256, 256))

# ╔═╡ 93befea9-d5a0-4e98-bff9-104ec9966b7e
I₂₁ = render(OrbAndCross(Square(),Star(1)), width=8);

# ╔═╡ 56cccfe4-5b0a-43f5-b18c-8a806e28c050
I₂₂ = render(OrbAndCross(Square(),Star(4)), width=8);

# ╔═╡ 2ecb9724-df90-44fb-a85e-631a407a36fb
I₂₃ = render(OrbAndCross(Square(),Star(8)), width=8);

# ╔═╡ dbf12106-5882-44e7-adba-fa6e9385d0a7
J₂₁ = extend_image(I₂₁, (256, 256))

# ╔═╡ ab03c600-d474-4ebc-a7cb-36b077adedb4
J₂₂ = extend_image(I₂₂, (256, 256))

# ╔═╡ 07af2e97-fbfc-4d40-9a3f-17cc48d99be6
J₂₃ = extend_image(I₂₃, (256, 256))

# ╔═╡ 51cd4fd0-1579-4c5b-8bdf-c25078fb1e60
I₃₁ = render(Shield(Circle()), width=8);

# ╔═╡ 53c400a3-81ad-4474-8236-7fa95d5fbb70
I₃₂ = render(Shield(Square()), width=8);

# ╔═╡ 030f9e22-ce1a-4801-b2fc-13dc0212e5ad
I₃₃ = render(Shield(Triangle()), width=8);

# ╔═╡ 3d76acd4-fcb8-4dda-b23e-d0f44b25b7c4
J₃₁ = extend_image(I₃₁, (256, 256))

# ╔═╡ 9f2cbed6-2a96-4d57-a7ea-3da2b85c1f67
J₃₂ = extend_image(I₃₂, (256, 256))

# ╔═╡ 98234e90-7299-46df-993f-85d3886a0e18
J₃₃ = extend_image(I₃₃, (256, 256))

# ╔═╡ c5ce0972-1126-4c36-92c3-b48c1d827776
I₄₁ = render(OrbAndCross(Triangle(),Star(1)), width=8);

# ╔═╡ 36827c02-bf3a-448d-b233-fa23b5e7e9e1
I₄₂ = render(OrbAndCross(Triangle(),Star(4)), width=8);

# ╔═╡ 5aabeab7-3bf5-48ca-88e8-577d15a32b09
I₄₃ = render(OrbAndCross(Triangle(),Star(8)), width=8);

# ╔═╡ 74ed80d3-04ae-484c-b91e-fee5a0b382c6
J₄₁ = extend_image(I₄₁, (256, 256))

# ╔═╡ 197ecbf7-7d21-4fac-a9c6-386446840002
J₄₂ = extend_image(I₄₂, (256, 256))

# ╔═╡ be4fc72b-0852-4a43-9a8a-cf079aa034ce
J₄₃ = extend_image(I₄₃, (256, 256))

# ╔═╡ 3d31ae1d-71a7-46d2-bf5e-c3069587353d
#J = [J₁₁,J₁₂,J₁₃,J₂₁,J₂₂,J₂₃,J₃₁,J₃₂,J₃₃,J₄₁,J₄₂,J₄₃]; Label = [1,2,3,4,5,6,7,8,9,10,11,12];
J = [J₁₁,J₂₂,J₃₃]; Label = [1,5,12];

# ╔═╡ 9a9fd3b2-8fb5-430a-8b43-9fab2618e9e8
Class, Labels = generate_academic_classes(J, Label, class_size=20);

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.5, 1.15), 
	scale_y = (0.5, 1.15),
	rotate=(-180.0, 180.0), 
	shear_x=(-45.0, 45.0),
	shear_y=(-45.0, 45.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
Random.seed!(42); TClass = A.(Class)

# ╔═╡ eff42224-7bdb-4d52-be3c-0fe19911cf2f
md"
# Setting the RCDT 
with 300 radii, 128 Radon angles, and 64 interpolation points.
The max- and mean-normalized RCDT are applied on the generated dataset.
"

# ╔═╡ 8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
R = RadonTransform(851,128,0.0) # (301,128), top

# ╔═╡ bbbcd04c-8b4f-4c44-958d-9e4089ada051
RCDT = RadonCDT(64, R)

# ╔═╡ 81fd55d8-24df-4047-b235-20468b2c111c
NRCDT = NormRadonCDT(RCDT)

# ╔═╡ a4860242-45bb-4f48-816c-3ed1bc82a9c6
rcdt = RCDT.(TClass);

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
MClass = reshape(vcat(Array{Float64}.(Class)...), :, length(Class));

# ╔═╡ 9ea0eb6a-9ce9-4863-adda-af47780cf2c7
MqQClass = reshape(vcat(rcdt...), :, length(rcdt));

# ╔═╡ 11dac364-a57a-4f0f-b534-2e730248223e
MtvQClass = reshape(vcat(tvqClass...), :, length(tvqClass));

# ╔═╡ 553a3e8d-ce84-4b82-8821-6927917b43e9
MmQClass = reshape(vcat(mqClass...), :, length(mqClass));

# ╔═╡ 74d09be3-2f2d-40ae-b8d5-c0cbdbd334ba
MmiQClass = reshape(vcat(miqClass...), :, length(miqClass));

# ╔═╡ a417781a-96c1-4341-8de8-de2a4847a51f
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

# ╔═╡ da3779ab-f514-4960-8de6-01d5d728324b
MmaQClass_pca = fit(PCA, MmaQClass; maxoutdim=2);  # PCA expects n x d

# ╔═╡ fda1a44f-1259-40bc-a16d-a205418bd0f6
MClass_reduced = transform(MClass_pca, MClass)'  # Now 2 x n

# ╔═╡ 82f04dcd-b40f-4e95-ae03-1d0ef962673e
MqQClass_reduced = transform(MqQClass_pca, MqQClass)'  # Now 2 x n

# ╔═╡ d187834a-5652-4761-b6ec-6dfb245195be
MtvQClass_reduced = transform(MtvQClass_pca, MtvQClass)'  # Now 2 x n

# ╔═╡ d49f0576-e17c-4aae-8086-59fbc89ea078
MmQClass_reduced = transform(MmQClass_pca, MmQClass)'  # Now 2 x n

# ╔═╡ ce620b29-5a21-4cd9-bc14-124e8031a2c7
MmiQClass_reduced = transform(MmiQClass_pca, MmiQClass)'  # Now 2 x n

# ╔═╡ c410947f-cace-431d-9047-f1991a609682
MmaQClass_reduced = transform(MmaQClass_pca, MmaQClass)'  # Now 2 x n

# ╔═╡ 361897e7-2d69-4f44-99a2-2fa6daa3c281
index=vcat(1:10,21:30,41:50)

# ╔═╡ 78e57189-2aee-491c-8e66-32c75c472751
offindex=vcat(11:20,31:40,51:60)

# ╔═╡ e0ade989-78b0-40aa-aada-8791f688f164
RK = kmeans(MClass[:,index], 3, weights=ones(30))

# ╔═╡ e3691904-1ea1-423c-bfd9-0442d39290a3
RKq = kmeans(MqQClass[:,index], 3, weights=ones(30))

# ╔═╡ bc469388-9c04-48f9-89a4-76add5f37ce3
RKtv = kmeans(MtvQClass[:,index], 3, weights=ones(30))

# ╔═╡ 96176a20-6537-418a-b0a1-99d1485c7cf4
RKm = kmeans(MmQClass[:,index], 3, weights=ones(30))

# ╔═╡ f1997ae9-3d0d-4103-ae27-8bae7b0f357d
RKmi = kmeans(MmiQClass[:,index], 3, weights=ones(30))

# ╔═╡ bdba9556-154a-485b-8c06-d13dfb81453b
RKma = kmeans(MmaQClass[:,index], 3, weights=ones(30))

# ╔═╡ cb11c33c-3ae0-47a8-8903-3bce5aea8af4
rename_map = Dict(1 => 1, 2 => 12, 3 => 5)

# ╔═╡ b737a14f-62ba-4b77-9e85-7120913ede05
label_marker = Dict(1 => :circle, 5 => :star5, 12 => :utriangle)

# ╔═╡ 86e981a1-3c7b-4bd1-ad42-fbc9b025bfd6
label_color = Dict(1 => :blue, 5 => :red, 12 => :green)

# ╔═╡ 3d4c1de2-1ee6-4e6d-a2d0-acf80676d384
RK_re = [rename_map[x] for x in RK.assignments]

# ╔═╡ 04e5d165-5c91-4adb-b64d-869d48605cb1
RK_label_re = [label_marker[x] for x in Labels]

# ╔═╡ 75830189-02aa-43ea-8f6d-90b94b5e88f9
RK_color_re = [label_color[x] for x in RK_re]

# ╔═╡ 1de81b0f-934d-413f-ae77-0bb036b69027
ann_unique = [1,21,41]; ann_MLLabel = Labels[ann_unique]; ann_color = [:blue, :red, :green];

# ╔═╡ dfd2f0fc-5409-405a-961e-304d60ccac50
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

# ╔═╡ 19139203-1ac7-4afb-89d0-3b9c8f25c6cf
label_color_off = Dict(3.0 => :red, 1.0 => :blue, 2.0 => :green)

# ╔═╡ 2eb799f0-c45d-425f-a01d-2b77dc95c3d4
label_off = Dict(3.0 => 2, 1.0 => 1, 2.0 => 3)

# ╔═╡ fcf48d0e-5881-48f5-b44d-f3cfb77eede5
RKmic = RKmi.centers;

# ╔═╡ 89ab8024-1e23-437c-9d0e-06e521f1b6d3
MmiQCenter_reduced = transform(MmiQClass_pca, RKmic)';

# ╔═╡ affeb321-ccb7-45bc-b9a5-fd96f1668184
RKmic_dis = dis_cluster(MmiQClass[:,offindex],RKmic)

# ╔═╡ 2d3783ee-7fd5-4ef5-987e-ef4fb3a38d86
RKmc = RKm.centers;

# ╔═╡ 498e5266-fb08-4d8b-be25-a1d421c60b25
MmQCenter_reduced = transform(MmQClass_pca, RKmc)';

# ╔═╡ 6b1ca0c7-951d-48b2-86de-bb0ff1a80aca
RKmc_dis = dis_cluster(MmQClass[:,offindex],RKmc)

# ╔═╡ 19af7d7b-b86a-4ae1-81c0-e137e41cfe9b
RKmac = RKma.centers;

# ╔═╡ 80820ad4-a752-4bbd-ba91-a745818f338d
MmaQCenter_reduced = transform(MmaQClass_pca, RKmac)';

# ╔═╡ 75ac813d-abb8-446c-b4c1-16594c5521a0
RKmac_dis = dis_cluster(MmaQClass[:,offindex],RKmac)

# ╔═╡ d15b7e98-176d-4570-870f-aa5e846d133a
RKtvc = RKtv.centers;

# ╔═╡ 0731cf47-a435-4125-8b20-2ddb32410c05
MtvQCenter_reduced = transform(MtvQClass_pca, RKtvc)';

# ╔═╡ a6292eaa-7c5d-4087-b862-501196805c9b
RKtvc_dis = dis_cluster(MtvQClass[:,offindex],RKtvc)

# ╔═╡ 980b0ef2-2134-4a6c-b90c-63a2945af6ff
RKqc = RKq.centers;

# ╔═╡ b1c3ff02-36c5-4eaf-93eb-e5a144ba47c7
MqQCenter_reduced = transform(MqQClass_pca, RKqc)';

# ╔═╡ d31382a1-b222-46d8-bfc6-59b99e93344c
RKqc_dis = dis_cluster(MqQClass[:,offindex],RKqc)

# ╔═╡ 4ca461e9-ed31-40c3-aec9-442927731562
RKc = RK.centers;

# ╔═╡ 602d81a0-95fd-4c19-8634-c8d261a46294
MCenter_reduced = transform(MClass_pca, RKc)';

# ╔═╡ 91823b4b-de2a-4a0b-b49a-5e3b632f9417
RKc_dis = dis_cluster(MClass[:,offindex],RKc)

# ╔═╡ 9393f200-7177-441c-abc4-24d5166964ef
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
	);
	scatter!(MClass_reduced[offindex,1], 
	MClass_reduced[offindex,2], 
	color=RK_color_off,
	marker=RK_label_re[offindex], alpha=0.2, label="")
	scatter!(MCenter_reduced[:,1], MCenter_reduced[:,2], marker=RK_label_re[[1,41,21]], color=:gray, label="")
end

# ╔═╡ 5486a162-442c-4565-9064-ebc14a7ad736
p

# ╔═╡ e16753b7-5140-4956-a6bc-068b876e278b
# ╠═╡ disabled = true
#=╠═╡
savefig(p, "k-means-academic_1512-eucl.pdf")
  ╠═╡ =#

# ╔═╡ b3d8bd5a-d45e-4147-9646-f330f4fe051a
RK_off = [label_off[x] for x in RKc_dis]

# ╔═╡ eecc6ee2-8061-446e-a00d-b9d3b7025f67
md"
- using t-SNE reduction.
"

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

# ╔═╡ 7d22aaa6-b0b4-4284-bd88-d873bf0de7ff
Labelss = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3]

# ╔═╡ 7bc33ff1-bbab-4830-831e-397fd3baa4f3
randindex(Labelss,RK.assignments)

# ╔═╡ ecc584e5-834d-4ce6-a73e-b7b8bacbbc12
randindex(Labelss,RKq.assignments)

# ╔═╡ 7c0799d4-0fc0-4271-9415-c1f66f257553
randindex(Labelss,RKtv.assignments)

# ╔═╡ 5ee4bcbc-0316-4671-aaf8-ae1c8f0a3b7d
randindex(Labelss,RKm.assignments)

# ╔═╡ 73c7e8d4-fd05-47d7-bbcd-2f000f222f9f
randindex(Labelss,RKmi.assignments)

# ╔═╡ 5964f006-55f2-4107-93ae-b1232dffc550
randindex(Labelss,RKma.assignments)

# ╔═╡ fc49725c-2cad-47ef-801f-3fb1f4905ae9
Clustering.varinfo(Labelss,RK_off) #RK.assignments)

# ╔═╡ 84a71715-4ee6-40c0-8397-8ca06a4a7fb1
Clustering.varinfo(Labelss,RKq.assignments)

# ╔═╡ c5c65067-0971-41ba-8be0-37b5ec874327
Clustering.varinfo(Labelss,RKtv.assignments)

# ╔═╡ 0d646daf-5e5e-4f9a-95c8-eef291cccc7e
Clustering.varinfo(Labelss,RKm.assignments)

# ╔═╡ 4c903525-80c7-4b33-b7fa-d4fbf674feaa
Clustering.varinfo(Labelss,RKmi.assignments)

# ╔═╡ 39fdce50-e10a-47b4-b1df-58fd81604a55
Clustering.vmeasure(Labelss,RK.assignments), Clustering.vmeasure(Labelss,RK.assignments, β=0.5), Clustering.vmeasure(Labelss,RK.assignments, β=2.0)

# ╔═╡ e513b185-a092-4a31-bc0f-3e7b35cc38b8
Clustering.vmeasure(Labelss,RKq.assignments), Clustering.vmeasure(Labelss,RKq.assignments, β=0.5), Clustering.vmeasure(Labelss,RKq.assignments, β=2.0)

# ╔═╡ 0e8b4919-af76-43bd-83d2-aab3cb85bef0
Clustering.vmeasure(Labels,RKtv.assignments), Clustering.vmeasure(Labels,RKtv.assignments, β=0.5), Clustering.vmeasure(Labels,RKtv.assignments, β=2.0)

# ╔═╡ bb87f60a-a367-40a9-9e8c-ffc0b465220d
Clustering.vmeasure(Labels,RKm.assignments), Clustering.vmeasure(Labels,RKm.assignments, β=0.5), Clustering.vmeasure(Labels,RKm.assignments, β=2.0)

# ╔═╡ 7f89a973-6197-4b2a-83e8-d13232c77e91
Clustering.vmeasure(Labels,RKmi.assignments), Clustering.vmeasure(Labels,RKmi.assignments, β=0.5), Clustering.vmeasure(Labels,RKmi.assignments, β=2.0)

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
# ╠═f1fc2e9a-2d65-421d-90da-6a9d28959dac
# ╠═da811304-9c01-47a9-8465-065dba6ec1fc
# ╠═e26b8000-9bc6-437f-b7b4-8c1b3883538f
# ╠═d71ce2b5-87a9-41ab-a5d7-030bc2da6b5e
# ╠═d7fb239f-26cc-4929-b917-253390965932
# ╠═0300491a-aa9c-43f0-933f-cb3f63a84f03
# ╠═93befea9-d5a0-4e98-bff9-104ec9966b7e
# ╠═56cccfe4-5b0a-43f5-b18c-8a806e28c050
# ╠═2ecb9724-df90-44fb-a85e-631a407a36fb
# ╠═dbf12106-5882-44e7-adba-fa6e9385d0a7
# ╠═ab03c600-d474-4ebc-a7cb-36b077adedb4
# ╠═07af2e97-fbfc-4d40-9a3f-17cc48d99be6
# ╠═51cd4fd0-1579-4c5b-8bdf-c25078fb1e60
# ╠═53c400a3-81ad-4474-8236-7fa95d5fbb70
# ╠═030f9e22-ce1a-4801-b2fc-13dc0212e5ad
# ╠═3d76acd4-fcb8-4dda-b23e-d0f44b25b7c4
# ╠═9f2cbed6-2a96-4d57-a7ea-3da2b85c1f67
# ╠═98234e90-7299-46df-993f-85d3886a0e18
# ╠═c5ce0972-1126-4c36-92c3-b48c1d827776
# ╠═36827c02-bf3a-448d-b233-fa23b5e7e9e1
# ╠═5aabeab7-3bf5-48ca-88e8-577d15a32b09
# ╠═74ed80d3-04ae-484c-b91e-fee5a0b382c6
# ╠═197ecbf7-7d21-4fac-a9c6-386446840002
# ╠═be4fc72b-0852-4a43-9a8a-cf079aa034ce
# ╠═3d31ae1d-71a7-46d2-bf5e-c3069587353d
# ╠═9a9fd3b2-8fb5-430a-8b43-9fab2618e9e8
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
# ╠═a417781a-96c1-4341-8de8-de2a4847a51f
# ╠═0eaf0f0a-8685-4b29-b666-69b1f747f223
# ╠═196a93d4-530d-4ce3-b881-9b57a27613d2
# ╠═18baf544-bb3f-40e4-9a21-7661aae925fa
# ╠═5a0a14bb-58d5-4c6b-a9ae-243eba459c60
# ╠═d4e0e65e-322c-4b8d-95f8-6fa8e310d0cd
# ╠═da3779ab-f514-4960-8de6-01d5d728324b
# ╠═fda1a44f-1259-40bc-a16d-a205418bd0f6
# ╠═82f04dcd-b40f-4e95-ae03-1d0ef962673e
# ╠═d187834a-5652-4761-b6ec-6dfb245195be
# ╠═d49f0576-e17c-4aae-8086-59fbc89ea078
# ╠═ce620b29-5a21-4cd9-bc14-124e8031a2c7
# ╠═c410947f-cace-431d-9047-f1991a609682
# ╠═361897e7-2d69-4f44-99a2-2fa6daa3c281
# ╠═78e57189-2aee-491c-8e66-32c75c472751
# ╠═e0ade989-78b0-40aa-aada-8791f688f164
# ╠═e3691904-1ea1-423c-bfd9-0442d39290a3
# ╠═bc469388-9c04-48f9-89a4-76add5f37ce3
# ╠═96176a20-6537-418a-b0a1-99d1485c7cf4
# ╠═f1997ae9-3d0d-4103-ae27-8bae7b0f357d
# ╠═bdba9556-154a-485b-8c06-d13dfb81453b
# ╠═cb11c33c-3ae0-47a8-8903-3bce5aea8af4
# ╠═b737a14f-62ba-4b77-9e85-7120913ede05
# ╠═86e981a1-3c7b-4bd1-ad42-fbc9b025bfd6
# ╠═3d4c1de2-1ee6-4e6d-a2d0-acf80676d384
# ╠═04e5d165-5c91-4adb-b64d-869d48605cb1
# ╠═75830189-02aa-43ea-8f6d-90b94b5e88f9
# ╠═1de81b0f-934d-413f-ae77-0bb036b69027
# ╠═d2107d93-e095-4945-b139-6bcea3d530cc
# ╠═5486a162-442c-4565-9064-ebc14a7ad736
# ╠═e16753b7-5140-4956-a6bc-068b876e278b
# ╠═dfd2f0fc-5409-405a-961e-304d60ccac50
# ╠═19139203-1ac7-4afb-89d0-3b9c8f25c6cf
# ╠═2eb799f0-c45d-425f-a01d-2b77dc95c3d4
# ╠═9393f200-7177-441c-abc4-24d5166964ef
# ╠═b3d8bd5a-d45e-4147-9646-f330f4fe051a
# ╠═fcf48d0e-5881-48f5-b44d-f3cfb77eede5
# ╠═89ab8024-1e23-437c-9d0e-06e521f1b6d3
# ╠═affeb321-ccb7-45bc-b9a5-fd96f1668184
# ╠═2d3783ee-7fd5-4ef5-987e-ef4fb3a38d86
# ╠═498e5266-fb08-4d8b-be25-a1d421c60b25
# ╠═6b1ca0c7-951d-48b2-86de-bb0ff1a80aca
# ╠═19af7d7b-b86a-4ae1-81c0-e137e41cfe9b
# ╠═80820ad4-a752-4bbd-ba91-a745818f338d
# ╠═75ac813d-abb8-446c-b4c1-16594c5521a0
# ╠═d15b7e98-176d-4570-870f-aa5e846d133a
# ╠═0731cf47-a435-4125-8b20-2ddb32410c05
# ╠═a6292eaa-7c5d-4087-b862-501196805c9b
# ╠═980b0ef2-2134-4a6c-b90c-63a2945af6ff
# ╠═b1c3ff02-36c5-4eaf-93eb-e5a144ba47c7
# ╠═d31382a1-b222-46d8-bfc6-59b99e93344c
# ╠═4ca461e9-ed31-40c3-aec9-442927731562
# ╠═602d81a0-95fd-4c19-8634-c8d261a46294
# ╠═91823b4b-de2a-4a0b-b49a-5e3b632f9417
# ╟─eecc6ee2-8061-446e-a00d-b9d3b7025f67
# ╟─23d4b1e3-fff7-4557-95db-4712d0dd4867
# ╠═9b1a18ca-c177-448a-b240-c151942356e6
# ╠═7d22aaa6-b0b4-4284-bd88-d873bf0de7ff
# ╠═7bc33ff1-bbab-4830-831e-397fd3baa4f3
# ╠═ecc584e5-834d-4ce6-a73e-b7b8bacbbc12
# ╠═7c0799d4-0fc0-4271-9415-c1f66f257553
# ╠═5ee4bcbc-0316-4671-aaf8-ae1c8f0a3b7d
# ╠═73c7e8d4-fd05-47d7-bbcd-2f000f222f9f
# ╠═5964f006-55f2-4107-93ae-b1232dffc550
# ╠═fc49725c-2cad-47ef-801f-3fb1f4905ae9
# ╠═84a71715-4ee6-40c0-8397-8ca06a4a7fb1
# ╠═c5c65067-0971-41ba-8be0-37b5ec874327
# ╠═0d646daf-5e5e-4f9a-95c8-eef291cccc7e
# ╠═4c903525-80c7-4b33-b7fa-d4fbf674feaa
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
