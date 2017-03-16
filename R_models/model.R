library(lme4)


add.transform <- function(feat.name){
  #return ('log10(abs(', feat.name, ') + 1)')
  return(feat.name)
}

get.model.feat <- function(feat.name, comp.data, valid.frac.thresh = 0.1){
  RANDOM_EFFECTS = '(1+is_control | exp_name/channel)'
  valid = is.finite(comp.data[[feat.name]])
  valid.comp.data = comp.data[valid,]
  valid.frac = sum(valid)/dim(comp.data)[1]
  if(valid.frac > valid.frac.thresh)
  {
    fit.full <- lmer(paste0(add.transform(feat.name), ' ~ is_control + ', RANDOM_EFFECTS), 
                     data = comp.data, na.action = na.exclude, REML=FALSE )
    
    fit.null <- lmer(paste0(add.transform(feat.name), ' ~ ', RANDOM_EFFECTS),
                     data = comp.data, REML=FALSE)
    
    likelihood <- anova(fit.null, fit.full, test="F")
    
    output = list('likelihood' = likelihood, 'fit.full' = fit.full, 'fit.null' = fit.null)
  }
  else {output = list()}
  
  return(output)
}

get.mod.linear <- function(features.names, comp.data){
  strain = as.character(comp.data[!comp.data$is_control]$Strain[1])
  get.model.comp = function(feat) {
    progress.wrapper(paste(strain, feat), get.model.feat, feat, comp.data)
  }
  feats.stats <- sapply(features.names, get.model.comp)
  return(feats.stats)
}


get.model.strain <- function(comp.data, strain, features.names){
  comp.data = get.comp.data(comp.data, strain)
  #calculate the linear model, displaying the time per feature
  txt = paste(strain,  "TOTAL")
  feats.stats = progress.wrapper(txt, get.mod.linear, features.names, comp.data)
  return(feats.stats)
}


progress.wrapper <- function(progress.txt, FUNC, ...){
  start.time <- Sys.time()
  output = FUNC(...)
  txt = sprintf('%s -> %2.2fs', progress.txt, Sys.time() - start.time)
  print(txt)
  return(output)
}

get.models <- function(features.data, control.strain){
  strains = levels(features.data$Strain)
  stopifnot(control.strain %in% strains)
  strains = strains[-which(strains == control.strain)]
  
  strains.stats = lapply(strains, function(x){get.model.strain(features.data, x, features.names)})
  names(strains.stats) = strains
  
  return(strains.stats)
}