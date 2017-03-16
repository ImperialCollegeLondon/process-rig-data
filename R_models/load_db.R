library(dplyr)
library(lubridate)
library(data.table)

read.experiments <- function(db_path) { 
  my_db <- src_sqlite(db_path)
  exp_sql = sql('SELECT * FROM experiments')
  experiments <- as.data.table(tbl(my_db, exp_sql))
  experiments = experiments[order(exp_name, base_name)]
  experiments <- within(experiments, {
    Strain <- as.factor(Strain)
    Picker <- as.factor(Picker)
    video_id <- as.factor(video_id)
    channel <- as.factor(channel)
    set_n <- as.factor(set_n)
    stage_pos <- as.factor(channel)
    exp_name <- factor(exp_name)
    video_timestamp <- ymd_hms(video_timestamp)
  })
  return(experiments)
}

read.feats <- function(db_path, feat.names, feat.table, control.strain=NULL){
  my_db <- src_sqlite(db_path)
  
  exp.fields = c("e.video_id AS video_id", 
                 "exp_name", "prefix", "set_n", 
                 "channel", "stage_pos", "video_timestamp", 
                 "N_Worms", "Picker", "Strain", 
                 "set_delta_time", "exp_delta_time",
                 "worm_index", "n_frames", "n_valid_skel", "first_frame")
  sql_cmd = paste0("SELECT ", paste(exp.fields, collapse = ',') , 
                   "," , paste(feat.names, collapse = ','),
                   " FROM experiments AS e JOIN ", feat.table, " AS tab ON tab.video_id = e.video_id")
  feat_means = as.data.table(tbl(my_db, sql(sql_cmd)))
  
  feat_means <- within(feat_means, {
    exp_name <- as.factor(exp_name)
    Strain <- as.factor(Strain)
    Picker <- as.factor(Picker)
    video_id <- as.factor(video_id)
    channel <- as.factor(channel)
    set_n <- as.factor(set_n)
    stage_pos <- as.factor(channel)
    video_timestamp <- ymd_hms(video_timestamp)
    worm_index <- as.factor(worm_index)
    
    if(!is.null(control.strain)) {
      is_control <- Strain == control.strain
    }
  })
  
  #make strain a key to allow us to search
  setkeyv(feat_means, "Strain")
  
  return (feat_means)
}

read.features.names <- function(db_path){
  my_db <- src_sqlite(db_path)
  #get list of features to be predicted
  features.names = colnames(tbl(my_db, sql("SELECT * FROM means LIMIT 1")))
  features.names = features.names[-which(features.names %in% c("worm_index", "n_frames", "n_valid_skel", "video_id", "first_frame"))]
  return(features.names)
}


get.comp.data = function(features.data, strain) {
  valid = (features.data$is_control | features.data$Strain == strain)
  comp.data = features.data[valid, ]
  comp.data = droplevels(comp.data)
  
  #check if i am really selecting two strains...
  n.strains = length(levels(comp.data$Strain))
  stopifnot(n.strains==2)
  
  return(comp.data)
}