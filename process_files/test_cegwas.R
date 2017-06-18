library(cegwas)

fname = '/Users/ajaver/Documents/GitHub/process-rig-data/process_files/CeNDR_all_avg.csv'
df = read.csv(fname, row.names = 1)
pheno = data.frame(t(df))
pheno = cbind(trait = colnames(df), pheno) #add the traits column to the first row
rownames(pheno) = seq(nrow(pheno)) #eliminate the columns names

processed_phenotypes <- process_pheno(pheno)
mapping_df <- gwas_mappings(processed_phenotypes)
processed_mapping_df <- process_mappings(mapping_df, phenotype_df = processed_phenotypes, CI_size = 50, snp_grouping = 200)


pdf("manhatan_plots.pdf", width=10, height=4)
manplot(processed_mapping_df)
dev.off()

pdf("phenotypes_boxplot.pdf")
pxg_plot(processed_mapping_df)
dev.off()

pdf("gene_variants.pdf")
gene_variants(processed_mapping_df)
dev.off()

