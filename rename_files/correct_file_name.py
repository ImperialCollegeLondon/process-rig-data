# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 16:15:39 2016

@author: worm_rig
"""

import os
import glob
import numpy as np
import pandas as pd
from functools import partial

GECKO_DT_FMT = '%d%m%Y_%H%M%S'

def rig_focus_to_microns_per_pixel(focus):
    ''' convert rig focus to micros per pixel'''
    return 10 #new calibration #-0.1937*(focus)+13.4377

def read_rig_csv_db(csv_file):
    db = pd.read_csv(csv_file)
    db.dropna(inplace=True, how='all')
    
    db['microns_per_pixel']= db['Focus'].apply(rig_focus_to_microns_per_pixel)
    db[['Set_N', 'Rig_Pos', 'Camera_N']] = db[['Set_N', 'Rig_Pos', 'Camera_N']].astype(np.int)
    
    db_ind = {(row['Set_N'],row['Rig_Pos'],row['Camera_N']) : irow 
             for irow, row in db.iterrows()}
          
    if db.shape[0] == len(db_ind):
        return db, db_ind
    else:
        raise ValueError('There must be only one combination Set_N, Rig_Pos, Camera_N per sample in the .csv file')
    
    

def read_rig_log_file(log_files):
    
    def _log_parts(x):
        dd = x.rpartition(':')
        video_timestamp = pd.to_datetime(dd[0].strip(), format='%d/%m/%Y %H:%M:%S')
        plate_n = int(dd[-1].partition('Plate')[-1].strip())
        return video_timestamp, plate_n
    
    lines = []
    for log_file in log_files:
        with open(log_file, 'r') as fid:
            lines += fid.read().split('\n')
    
        
    move_times = [_log_parts(x) for x in lines if 'Moving to Plate' in x]
    move_times = pd.DataFrame(move_times, columns=['time', 'stage_pos'])
    
    move_times = move_times.sort_values(by = 'time')
    
    return move_times

def get_rig_pos(movie_time, rig_move_times, max_delta = pd.to_timedelta('5min')):
    '''get rig position by matching movie time with the stage log'''
    
    
    dt = movie_time - rig_move_times['time']
    dt = dt[dt >= np.timedelta64(0)]
    
    if dt.min() < max_delta:
        stage_pos = rig_move_times.loc[dt.argmin(), 'stage_pos']
    else:
        stage_pos = np.nan
        
    return stage_pos
    
def gecko_fnames_to_table(filenames):
    def _gecko_movie_parts(x):
        '''
        Read movie parts as expected from a Gecko file.
        '''
        dir_name, bb = os.path.split(x)
        base_name, ext = os.path.splitext(bb)
        
        parts = base_name.lower().split('_')
        
        if parts[-1] in ['skeletons', 'intensities', 'features', 'trajectories']:
            postfix = '_' + parts[-1]
            parts = parts[:-1]
        else:
            postfix = ''
        
        try:
            video_timestamp = pd.to_datetime(parts[-2] + '_' + parts[-1], 
                                   format=GECKO_DT_FMT)
            parts = parts[:-2]
        
        except ValueError:
            video_timestamp = pd.NaT
            
     
        def _part_n(start_str, parts_d):
            if parts[-1].startswith(start_str):
                part_n = int(parts_d[-1].replace(start_str, ''))
                parts_d = parts_d[:-1]
            else:
                part_n = -1
            return part_n, parts_d
        
            
        channel, parts = _part_n('ch', parts)
        stage_pos, parts = _part_n('pos', parts)
        set_n, parts = _part_n('set', parts)
        
        prefix = '_'.join(parts)
        
        return dir_name, base_name, ext, prefix, channel, stage_pos, set_n, video_timestamp, postfix
        
    fparts = [_gecko_movie_parts(x) for x in filenames]
    fparts_tab = pd.DataFrame(fparts, columns = ('directory', 'base_name', 'ext', 'prefix', 'channel', 'stage_pos', 'set_n', 'video_timestamp', 'postfix'))
    return fparts_tab

    
def get_movie_dirs(movies_dir_D, exp_name):
    movie_dirs = []
    for ii in range(3):
        n_pc = str(ii+1)
        dd = os.path.join(movies_dir_D, 'PC' + n_pc, exp_name + '_' + n_pc)
        movie_dirs.append(dd)
    
    assert all(os.path.exists(x) for x in movie_dirs)
    return movie_dirs
    
def read_extra_data(output_root_d, original_root_d):
    def _move_if_needed(fname, old_location):
        if not os.path.exists(fname):
            fname_original = os.path.join(old_location, os.path.basename(fname))
            if not os.path.exists(fname_original):
                raise FileNotFoundError(fname_original)
            os.rename(fname_original, fname)
    
            
            
    #get aux files location
    extra_dir = os.path.join(output_root_d, 'ExtraFiles')
    if not os.path.exists(extra_dir):
        os.makedirs(extra_dir)
    
    
    csv_file = os.path.join(extra_dir, exp_name + '.csv')
    _move_if_needed(csv_file, original_root_d)
        
    
    log_files = glob.glob(os.path.join(extra_dir, exp_name + '*.log'))
    if len(log_files) == 0:
        ori_log_files = glob.glob(os.path.join(original_root_d, exp_name + '*.log'))
        for ori_log_f in ori_log_files:
            new_file = ori_log_f.replace(original_root_d, extra_dir)
            os.rename(ori_log_f, new_file)
            log_files.append(new_file)
    
    
    #move to the new location if needed
    
    [_move_if_needed(log_file, original_root_d) for log_file in log_files]
    
    
    #read aux files
    db, db_ind = read_rig_csv_db(csv_file)
    if log_files:
        rig_move_times = read_rig_log_file(log_files)
    else:
        rig_move_times = pd.DataFrame()
        
        unique_triples = set(tuple(row[col] for col in ['Rig_Pos', 'Camera_N', 'Set_N']) for ii, row in db.iterrows())
        assert len(unique_triples) == len(db)
        
    return rig_move_times, db, db_ind
    
#%%
def get_new_names(movie_dir, pc_n, db, db_ind, rig_move_times, output_dir, f_ext=None):
    #%%
    
    if f_ext is None:
        fnames1 = glob.glob(os.path.join(movie_dir, '**', '*.mjpg'), recursive=True)
        fnames2 = glob.glob(os.path.join(movie_dir, '**', '*hdf5'), recursive=True)
        fnames = fnames1 + fnames2
    else:
        fnames2 = glob.glob(os.path.join(movie_dir, '**', f_ext), recursive=True)
    
    fparts_table = gecko_fnames_to_table(fnames)
    #correct the channel using the pc number
    fparts_table['channel']  += pc_n*2
    
    #get rig pos from log time
    if rig_move_times.size > 0:
        fparts_table['stage_pos'] = fparts_table['video_timestamp'].apply(partial(get_rig_pos, rig_move_times=rig_move_times))
    else:
        
        for ii, row in fparts_table.iterrows():
            good = (db['Camera_N'] == row['channel']) & (db['Set_N'] == row['set_n'])
            ind = np.where(good)[0]
            assert ind.size==1
            
            fparts_table.loc[ii, 'stage_pos'] = db.loc[ind[0], 'Rig_Pos']
            
    #%%
    dir_files_to_rename = []
    for old_fname, (irow, row) in zip(fnames, fparts_table.iterrows()):
        try:
            #match movie using set_n, pos_n, ch_n
            db_row = db.loc[db_ind[(row['set_n'], row['stage_pos'], row['channel'])]]
        except KeyError:
            #not match in the csv database
            print('FILE NOT LOCATED IN THE DATABASE: ' +  old_fname)
            continue
            
        new_prefix = new_prefix_fun(db_row)

        new_base = '{}_Set{}_Pos{}_Ch{}_{}{}{}'.format(new_prefix,
                                               int(row['set_n']), 
                                               int(row['stage_pos']), 
                                               int(row['channel']),
                                               row['video_timestamp'].strftime(GECKO_DT_FMT),
                                               row['postfix'], 
                                               row['ext'])

        new_fname = os.path.join(output_dir, new_base)
        #print(os.path.split(old_fname), os.path.split(new_fname))
        dir_files_to_rename.append((old_fname, new_fname))
    return dir_files_to_rename
#%% 
def get_new_names_pc(original_root, exp_name, output_root):
        
    #get data from the extra files
    rig_move_times, db, db_ind = read_extra_data(output_root, original_root)
    
    #create output directory where the files are going to be moved
    output_dir = os.path.join(output_root, 'RawVideos',  exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    #get de directories for a particular experiment
    movie_dirs = get_movie_dirs(original_root, exp_name)    
    
    
    enc_in = os.path.join(movie_dirs[0], 'wormencoder.ini')
    if os.path.exists(enc_in):
        enc_out = os.path.join(output_dir, 'wormencoder.ini')
        os.rename(enc_in, enc_out)
    
    #explore each directory and get the expected new name
    get_new_d = partial(get_new_names,
                        db = db, 
                        db_ind = db_ind, 
                        rig_move_times = rig_move_times, 
                        output_dir = output_dir)
    
    files_to_rename = [get_new_d(movie_dir, pc_n) for pc_n, movie_dir in enumerate(movie_dirs)]
    #flatten list
    files_to_rename = sum(files_to_rename, [])
    files_to_rename = [(x, y.replace('.hdf5', '.raw_hdf5')) for x,y in files_to_rename]
    


    return files_to_rename
            

def rename_after_bad_choice(output_root, exp_name, f_ext):
    #%%
    raw_dir = os.path.join(output_root, 'RawVideos',  exp_name)
    results_dir = os.path.join(output_root, 'Results',  exp_name)
    mask_dir = os.path.join(output_root, 'MaskedVideos',  exp_name)
    
    if 'mjpg' in f_ext:
        output_dir = raw_dir
        r_fnames = glob.glob(os.path.join(results_dir, '**', '*.hdf5'), recursive=True)
        m_fnames = glob.glob(os.path.join(mask_dir, '**', '*.hdf5'), recursive=True)
        fnames = r_fnames + m_fnames
    elif 'hdf5' in f_ext:
        output_dir = mask_dir
        fnames = glob.glob(os.path.join(results_dir, '**', '*.hdf5'), recursive=True)
    else:
        raise ValueError('f_ext must be mjpg or hdf5')
    
    #%%
    
    assert os.path.exists(output_dir)
    
    rig_move_times, db, db_ind = read_extra_data(output_root, '')
    
    files2rename = get_new_names(output_dir, 0, f_ext, db, db_ind, rig_move_times, output_dir, new_prefix_fun)
    #%%
    getbn = lambda x : os.path.splitext(os.path.basename(x))[0]
    old2new = {getbn(x):getbn(y) for x,y in files2rename}
    
    
    dd = [[(x,x.replace(key, old2new[key])) for x in fnames if key in x] for key in old2new]
    dd = sum(dd,[]) #flatten
    
    assert len(dd) == len(fnames)
    files2rename += dd
    #%%
    return files2rename
    

    
def _switch_pos_ch():
    #output_root = "/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/Test_20161027"
    results_dir = os.path.join(output_root, 'MaskedVideos')
    mask_dir = os.path.join(output_root, 'Results')
    r_fnames = glob.glob(os.path.join(results_dir, '**', '*.hdf5'), recursive=True)
    m_fnames = glob.glob(os.path.join(mask_dir, '**', '*.hdf5'), recursive=True)
    fnames = r_fnames + m_fnames
    
    
    
    def correct_name(x):
        prefix, mid, rest = x.partition('_Ch')
        ch_n, mid, rest = rest.partition('_Pos')
        pos_n, mid, rest = rest.partition('_')
        assert ch_n and pos_n
        return '{}_Pos{}_Ch{}_{}'.format(prefix, pos_n, ch_n, rest)
    
    files_to_rename = [(x, correct_name(x)) for x in fnames]
    return files_to_rename

    
def new_prefix_fun(db_row):
    if 'Vortex' in db_row:
        base_name = '{}_N{}'.format(db_row['Strain'], db_row['N_Worms'])
        if db_row['Vortex'] == 1:
            base_name += '_V'
        base_name
    else:
        base_name = '{}_N{}_F1-{}'.format(db_row['Strain'], db_row['N_Worms'], db_row['Food_Conc'])
    
    return base_name
        
def print_files_to_rename(files_to_rename):
    for fnames in files_to_rename:
        old_name, new_name = fnames
        new_name = os.path.basename(new_name)
        dnameo, fname_old = os.path.split(old_name)
        pc_n = [x for x in dnameo.split(os.sep) if x.startswith('PC')][0]
        print('%s => %s' % (os.path.join(pc_n, fname_old), new_name))
    
if __name__ == '__main__':
    #f_ext = '*hdf5'
    f_ext = '*.mjpg'
    exp_name = 'double_pick_151216'
    
    raw_movies_root = "/Volumes/behavgenom_archive$/RigRawVideos"
    output_root = "/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/short_movies/"
    
    files_to_rename = get_new_names_pc(raw_movies_root, 
                                       exp_name, 
                                       output_root)
    
    if not files_to_rename:
        print('No files found. Exiting.')
    else:
        print_files_to_rename(files_to_rename)
        reply = input('The previous files are going to be renamed. Do you wish to continue (y/N)?')
        reply = reply.lower()
        if reply in ['yes', 'ye', 'y']:
            print('Renaming files...')
            
            #move files and save the changes into _renamed.tsv
            save_renamed = os.path.join(output_root, 'ExtraFiles', exp_name + '_renamed.tsv')
            with open(save_renamed, 'a') as fid:
                for old_name, new_name in files_to_rename:
                    os.rename(old_name, new_name)
                    fid.write('{}\t{}\n'.format(old_name, new_name));
            
            print('Done.')
        else:
            print('Aborted.')
        
    #%%
    #f_ext = '*.hdf5'
    #dd = rename_after_bad_choice(output_root, exp_name, f_ext, new_prefix_fun)
    #wrong_naming = [[y for y in map(os.path.basename, x)] for x in dd if x[0]!=x[1]]
    
