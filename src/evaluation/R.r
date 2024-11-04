# remotes::install_github("mojaveazure/seurat-disk")

# Data format transformation
library(SeuratDisk)
Convert("R/bm_proj.h5ad", dest="h5seurat", assay = "RNA", overwrite=F)

# Load converted data
seurat_object <- LoadH5Seurat("R/bm_proj.h5seurat", meta.data=FALSE) # misc=FALSE

meta.data <- read.csv("R/dg_obs.csv")
seurat_object@meta.data <- meta.data

# clustering
seurat_object <- FindNeighbors(seurat_object, dims=1:30)  #
seurat_object <- FindClusters(seurat_object, resolution=0.5)  # greater resolution, more clusters

colors <- c('#e5c494', '#a6cee3', '#a6d854', '#4daf4a', '#f781bf', '#e78ac3', '#ff7f00', '#e41a1c', '#e41a2c', '#984ea3')  # bm
colors <- c('#3ba458', '#404040', '#7a7a7a', '#fda762', '#6950a3', '#2575b7', '#08306b')  # fb
fig15 <- DimPlot(seurat_object, reduction="umap", group.by='clusters', label=TRUE, cols=colors)
print(fig15)
saveRDS(seurat_object, file = 'R/dentate.rds')


# install.packages("BiocManager")
# BiocManager::install(version = "3.14")
# BiocManager::install(c('BiocGenerics', 'DelayedArray', 'DelayedMatrixStats', 
#                        'limma', 'S4Vectors', 'SingleCellExperiment', 
#                        'SummarizedExperiment', 'batchelor', 'Matrix.utils'))
# BiocManager::install(c('BiocGenerics', 'DelayedArray', 'DelayedMatrixStats',
#                        'limma', 'lme4', 'S4Vectors', 'SingleCellExperiment',
#                        'SummarizedExperiment', 'batchelor', 'HDF5Array',
#                        'terra', 'ggrastr'))
#
# install.packages("devtools")
## devtools::install_github('cole-trapnell-lab/leidenbase')
# devtools::install_github('cole-trapnell-lab/monocle3')

# BiocManager::install("monocle")  # Monocle2

library(monocle)
hsc <- readRDS('R/bm.rds')

expression_matrix <- as(as.matrix(hsc@assays$RNA@counts), 'sparseMatrix')
cell.meta.data <- hsc@meta.data
gene.meta.data <- data.frame(gene_short_name=row.names(expression_matrix), row.names=row.names(expression_matrix))

cell.meta.data <- new('AnnotatedDataFrame', data=cell.meta.data)
gene.meta.data <- new('AnnotatedDataFrame', data=gene.meta.data)
cds <- newCellDataSet(expression_matrix, phenoData=cell.meta.data, featureData=gene.meta.data)

cds <- estimateSizeFactors(cds)
cds <- estimateDispersions(cds)
expression_genes <- rownames(hsc)[hsc$RNA@meta.features$highly_variable]  # as.logical

# Seurat to choose hvgs
expression_genes <- VariableFeatures(hsc)
cds <- setOrderingFilter(cds, expression_genes)
plot_ordering_genes(cds)
# clustering marker genes
deg.cluster <- FindAllMarkers(hsc)
expression_genes <- subset(deg.cluster, p_val_adj < 0.05)$gene
cds <- setOrderingFilter(cds, expression_genes)
plot_ordering_genes(cds)
# Monocle to choose hvgs
disp_table <- dispersionTable(cds)
disp.genes <- subset(disp_table, mean_expression >= 0.1 & dispersion_empirical >= 1 * dispersion_fit)
cds <- setOrderingFilter(cds, disp.genes)
plot_ordering_genes(cds)

cds <- reduceDimension(cds, reduction_method="ICA")
# cds <- reduceDimension(cds, verbose = F, scaling = T, max_components = 4, maxIter = 100, norm_method = 'log',  lambda = 20 * ncol(cds))
cds <- orderCells(cds)
plot_cell_trajectory(cds, color_by = c('Pseudotime', 'cluster'))



# BiocManager::install("celldex")
# BiocManager::install("SingleR")

library(celldex)
library(SingleR)

ref <- MouseRNAseqData()

obj_sr <- GetAssayData(seurat_object, slot="data")
obj.ref <- SingleR(test = obj_sr, ref = ref, labels = ref$label.main)

table(obj.ref$labels, seurat_object@meta.data$seurat_clusters)
seurat_object@meta.data$labels <- obj.ref$labels
fig16 <- DimPlot(seurat_object, group.by = c("seurat_clusters", "labels"), reduction = "umap")
print(fig16)



require(remotes)
remotes::install_github('JEFworks-Lab/veloviz')
# BiocMsnager::install("veloviz")
# BiocManager::install('pcaMethods')
# devtools::install_github("velocyto-team/velocyto.R")
library(veloviz)
# library(velocyto.R)

# download vinette dataset
download.file("https://zenodo.org/record/4632471/files/pancreas.rda?download=1", destfile = "pancreas.rda", method = "curl")

curr <- seurat_object$RNA@data
proj <- seurat_object$proj@data
hvgs <- seurat_object$RNA@meta.features$highly_variable
curr <- as.matrix(curr)[hvgs,]
proj <- proj[hvgs,]
proj[proj<0] <- 0

clusters <- meta.data$clusters
names(clusters) <- meta.data$index
# colors <- rev(plasma(length(unique(clusters))))
colors <- c('#3ba458', '#404040', '#7a7a7a', '#fda762', '#6950a3', '#2575b7', '#08306b', '#e1bfb0', '#e5d8bd', '#79b5d9', '#f14432', 
            '#fc8a6a', '#98d594', '#d0e1f2')  # dg
image(1:14,1,as.matrix(1:14), col=mycolor)
names(colors) <- c('Astrocytes', 'Cajal Retzius', 'Cck-Tox', 'Endothelial', 'GABA', 'Granule immature', 'Granule mature', 
                    'Microglia', 'Mossy', 'Neuroblast', 'OL', 'OPC', 'Radial Glia-like', 'nIPC')
# names(colors) <- unique(clusters)
cell.colors <- colors[clusters]  # as.character(clusters)
names(cell.colors) <- names(clusters)


veloviz <- buildVeloviz(curr=curr, proj=proj, nPCs=30, k=30, similarity.threshold=0.2, verbose=TRUE)
# normalize.depth=TRUE, use.ods.genes=TRUE, alpha=0.05, pca=TRUE, center=TRUE, scale=TRUE, 
# nPCs=10, k=10, similarity.threshold=0, distance.weight=1, distance.threshold=1, weighted=TRUE, verbose=FALSE
emb.veloviz <- veloviz$fdg_coords
plotEmbedding(emb.veloviz, groups=clusters, colors=cell.colors[rownames(emb.veloviz)], main='VeloViz with dynamical velocity',
              xlab = "VeloViz X", ylab = "VeloViz Y", alpha = 0.8, cex.lab = 1.5)
