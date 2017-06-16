fname = '/Users/ajaver/Documents/GitHub/process-rig-data/process_files/CeNDR_all_avg.csv'
df = read.csv(fname, row.names = 1)
df_t = data.frame(t(df))
df_t$trait = colnames(df)
proc_pheno <- process_pheno(df_t)
dd = process_pheno(df_t[c('hips_bend_sd_abs10.0th'), ])
#colnames(df_t) = colnames(df)
#rownames(df_t) = rownames(df)

# pheno_f = data.frame(t(df_t["length_abs50.0th", ]))
#rownames(pheno_f) = c("trait")
#proc_pheno <- process_pheno(pheno_f)

