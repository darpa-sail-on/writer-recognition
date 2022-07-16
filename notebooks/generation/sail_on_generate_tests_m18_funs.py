import numpy as np
import pandas as pd
import scipy.stats
import os
import argparse
import json
import glob
import ast
import random
from types import SimpleNamespace


def create_list_novel_sizes(cfg, sp, red_light_batch_index):
    """
        Use beta distribution to determine where novel examples appear.

    Returns: list x corresponding to the novel batches, where x[i] is the number of novel examples 
     for the i^th batch (where i is the index into the list of novel batches, length(x) = number of novel batches)
    """
    distparams = cfg.beta_dist_params[sp.dist_type]
    
    n_batch_novel  = int((cfg.batch_number - red_light_batch_index) * sp.prop_unknown)
    novel_size     = n_batch_novel * cfg.batch_size
    
    # Prepare a probability bin for a given novel distribution
    bin_prob = np.linspace(0, 1, cfg.batch_number - red_light_batch_index + 1).tolist()
    list_novelsize = []
    for i in range(len(bin_prob) - 1):
        list_novelsize.append(
            int((scipy.stats.beta.cdf(bin_prob[i+1], distparams[0], distparams[1], loc=0, scale=1) -
                 scipy.stats.beta.cdf(bin_prob[i], distparams[0], distparams[1], loc=0, scale=1)) * novel_size))

    list_novelsize = [max(0,min(cfg.batch_size,i)) for i in list_novelsize]
    return list_novelsize


def draw_from_dataset(df_main, ix, nsamp, rngstate):
    """
        Draw known and unknown samples from dataset, given the anonfile dataframe
        and indices into the dataframe for knowns and unknowns.    
    df_main: anonfile dataframe
    ix: indices into df_main for samples
    nsamp: number of samples
    rngstate: random number generator state
    """
    df_full = df_main.loc[ix]
    print("Drawing {} samples from {} rows.".format(nsamp, len(df_full)))
    return df_full.sample(n = nsamp, random_state = rngstate, replace = False) 


def specline_sanity_checks(cfg, sp, knowns, unknowns):
    assert(sp.protocol == "OND")
    assert(sp.task in ["hwr","ar"])
    if sp.task == "hwr":
        assert(sp.novelty_type in ["class","letter","background"])
    else:
        assert(sp.novelty_type in ["class","spatial","temporal"])
    assert(sp.red_light_level in ["E","M","H"])
    assert(sp.dist_type in ["low","mid","high","flat"])
    
    if sp.task == "hwr":
        df_kg = pd.read_csv(
            cfg.hwr.knowledge_graph_file, header=None, 
            names = ["ontology_id","class","type","class_id"]).set_index("class")
        assert(np.all(df_kg.loc[knowns]["ontology_id"] < cfg.hwr.known_classes_ontologyid))
        assert(np.all(df_kg.loc[unknowns]["ontology_id"] >= cfg.hwr.known_classes_ontologyid))
    else:
        df_kg = pd.read_csv(
            cfg.ar.knowledge_graph_file, header=None, 
            names = ["ontology_id","class","type","class_id"]).set_index("class")
        assert(np.all(df_kg.loc[knowns]["ontology_id"] < cfg.ar.known_classes_ontologyid))  
        assert(np.all(df_kg.loc[unknowns]["ontology_id"] >= cfg.ar.known_classes_ontologyid))
        
    return


def draw_from_dataset_by_task(cfg, sp, df_main, random_list, knowns, unknowns, list_knownsize, list_novelsize):
    """
        This is random selection of activities for the test level
    """
    print(sp)
    if sp.novelty_type == "class": # class level novelty
        if sp.task == 'hwr':
            df_known = draw_from_dataset(
                df_main,
                (df_main["source"].isin(cfg.hwr.source_include)) & 
                (df_main['ontology_id'] < cfg.hwr.known_classes_ontologyid),
                sum(list_knownsize), random_list[1])
            df_unknown = draw_from_dataset(
                df_main,
                (df_main["source"].isin(cfg.hwr.source_include)) & 
                (df_main['ontology_id'] >= cfg.hwr.known_classes_ontologyid),
                sum(list_novelsize), random_list[2])             
        else:
            df_known = draw_from_dataset(
                df_main,
                (df_main["source"].isin(cfg.ar.source_include)) &
                #(df_main["ontology_id"] < cfg.ar.known_classes_ontologyid) &
                (df_main['class'].isin(knowns)),
                sum(list_knownsize), random_list[1])
            df_unknown = draw_from_dataset(
                df_main,
                (df_main["source"].isin(cfg.ar.source_include))  & 
                #(df_main["ontology_id"] >= cfg.ar.known_classes_ontologyid) &
                (df_main['class'].isin(unknowns)),
                sum(list_novelsize), random_list[2])
    else: # background/hwr or temporal/ar novelty
        if sp.task == 'hwr':
            df_known = draw_from_dataset(
                df_main,
                (df_main['class'].isin(knowns)) & 
                (df_main['source'].isin(eval(sp.known_sources))),
                sum(list_knownsize), random_list[1])            
            df_unknown = draw_from_dataset(
                df_main,
                (df_main['class'].isin(knowns)) &
                (df_main['source'].isin(eval(sp.unknown_sources))),
                sum(list_novelsize), random_list[2])
        else: # ar
            df_known = draw_from_dataset(
                df_main,
                #(df_main["usage"]=='train') &
                (df_main['class'].isin(knowns)) &
                (df_main['source'].isin(eval(sp.known_sources))),
                sum(list_knownsize), random_list[1])    
            df_unknown = draw_from_dataset(
                df_main,
                (df_main['class'].isin(knowns)) &
                (df_main['source'].isin(eval(sp.unknown_sources))),                
                sum(list_novelsize), random_list[2])              
    return df_known, df_unknown


def set_novelty_type_columns(sp, df_main, knowns, unknowns):
    # Set the novel changes column
    if sp.task=="hwr":
        if sp.novelty_type == "class":
            df_main['letter']     = 0
            df_main['background'] = 0
        elif sp.novelty_type == "letter":
            df_main['letter']     = (df_main['source'].isin(eval(sp.unknown_sources))).astype(np.int)
            df_main['background'] = 0
        elif sp.novelty_type == "background":
            df_main['letter']     = 0 
            df_main['background'] = (df_main['source'].isin(eval(sp.unknown_sources))).astype(np.int)  
        else:
            print("Novelty level bad")
            raise(Exception())
    else: # activity recognition
        if sp.novelty_type == "class":
            df_main['spatial']  = 0
            df_main['temporal'] = 0
        elif sp.novelty_type == "spatial":
            df_main['spatial']  = df_main['source'].isin(eval(sp.unknown_sources)).astype(np.int) # source in novelty list
            df_main['temporal'] = 0
        elif sp.novelty_type == "temporal":
            df_main['spatial']  = 0
            df_main['temporal'] = df_main['source'].isin(eval(sp.unknown_sources)).astype(np.int)
        else:
            raise(Exception("Novelty level bad"))


def get_knowns_and_unknowns(cfg, sp, df_main, random_list):
    "Get class names to use for the knowns and the unknowns"
    if pd.isna(sp.n_random_known_classes) or sp.n_random_known_classes == 0: # this is a flag to manually specify known classes
        knowns = [] if pd.isna(sp.known_classes) else eval(sp.known_classes)
    else: # randomly select n_random_known_classes known classes
        np.random.seed(random_list[10])
        if sp.task == 'hwr':
            # randomly chose n_random_known_classes activities
            # Choose n_random_known_classes known activities for the test
            knowns = list(np.random.choice(
                df_main['class'].loc[
                    (df_main['ontology_id'] < cfg.hwr.known_classes_ontologyid)].unique().tolist(), sp.n_random_known_classes))
        else:
            knowns = list(np.random.choice(
                df_main['class'].loc[
                    (df_main['ontology_id'] < cfg.ar.known_classes_ontologyid)].loc[
                    ~df_main['class'].isin(cfg.ar.known_class_exclude)].unique().tolist(), sp.n_random_known_classes))

    #unknowns = [] if pd.isna(sp.unknown_classes) else eval(sp.unknown_classes)
    if pd.isna(sp.n_random_unknown_classes) or sp.n_random_unknown_classes == 0:
        unknowns = [] if pd.isna(sp.unknown_classes) else eval(sp.unknown_classes)
    else: # randomly select n_random_known_classes known classes
        np.random.seed(random_list[11])
        if sp.task == 'hwr':
            unknowns = list(np.random.choice(
                df_main['class'].loc[
                    (df_main['ontology_id'] >= cfg.hwr.known_classes_ontologyid)].unique().tolist(), sp.n_random_unknown_classes))
        else:
            unknowns = list(np.random.choice(
                df_main['class'].loc[
                    (df_main['ontology_id'] >= cfg.ar.known_classes_ontologyid)].unique().tolist(), sp.n_random_unknown_classes))

    return knowns, unknowns


def create_shuffled_test_dataframe(
    cfg, random_list, grp, df_known, df_unknown, outcols, red_light_batch_index, list_knownsize, list_novelsize):
    #Shuffle data:
    df_known            = df_known.sample(frac=1, random_state=random_list[100+grp], replace=False)
    df_unknown          = df_unknown.sample(frac=1, random_state=random_list[200+grp], replace=False)
    #temporary group for sorting
    df_known['group']   = np.repeat([-1] * red_light_batch_index + list(range(cfg.batch_number - red_light_batch_index)), list_knownsize)
    df_unknown['group'] = np.repeat(list(range(len(list_novelsize))), list_novelsize)
    #assign novel label
    df_known['novel']   = 0
    df_unknown['novel'] = 1

    # Combined output based on the assigned distribution of novel instances
    # Reorder by group 
    # Add detection point
    df_out     = pd.concat([df_unknown[outcols], df_known[outcols]],axis=0).sort_values('group')
    df_out.insert(5,'detection',np.where(df_out.reset_index().index < cfg.batch_size * red_light_batch_index, 0, 1))

    #Shuffle known & unknown order within a group
    df_out = df_out.groupby("group").apply(
        lambda group_df: group_df.sample(
            frac=1,random_state=random_list[50+grp])).reset_index(drop=True)  
    
    return df_out
    
def check_assertions(cfg, specline, md, df_out):
    if specline.task == 'ar':
        df_known = df_out[df_out["detection"]==0]
        df_unknown = df_out[df_out["detection"]==1]
        if specline.novelty_type == "class":
            df_nonnovel = df_out[df_out["novel"]==0]
            df_novel = df_out[df_out["novel"]==1]            
            assert(
                np.all(df_nonnovel["ontology_id"] < cfg.ar.known_classes_ontologyid) &
                np.all(df_novel["ontology_id"] >= cfg.ar.known_classes_ontologyid)
            )
        if specline.novelty_type == "spatial":
            assert(np.all(df_out["novel"] == df_out["spatial"]))
        if specline.novelty_type == "temporal":
            assert(np.all(df_out["novel"] == df_out["temporal"]))
        if specline.novelty_type in ["spatial","temporal"]:
            assert(np.all(df_out["ontology_id"] < cfg.ar.known_classes_ontologyid))
            
        assert(len(df_out) == int(md["round_size"]) * int(md["n_rounds"]))
        assert(len(df_known) == int(md["red_light_index"]))
    return

def generate_test_group(cfg, df_main, specline):
    """
        Given a set of parameters from a given line in the specification, generate a 
        group of tests wrt the parameters. Each test in the group is a shuffled version
        of the other tests in the group.
    
    cfg: configuration params
    df_main: anonfile dataframe
    specline: line in the specification file
    """
    if specline.task == 'hwr':
        paths = cfg.hwr.paths
        ##Output columns
        outcols  = ['group','text','annonymous_id','novel','ontology_id','letter','background']
        outcols2 = ['annonymous_id','novel','detection','letter','background','ontology_id','text']
    else: # activity recognition
        paths = cfg.ar.paths
        ##Output columns
        outcols  = ['group','annonymous_id','novel','ontology_id','spatial','temporal']
        outcols2 = ['annonymous_id','novel','detection','spatial','temporal','ontology_id']
    
    # Generate the master random number list based on the master seed on the spec sheet
    # This is a test level seeds
    np.random.seed(specline.seed)
    random_list = np.random.randint(999999, size=300)

    # Create output folder if not exists
    outpath = os.path.join(paths[specline.novelty_type], str(specline.test_id))
    print("Output path: {}".format(outpath))
    os.makedirs(outpath, exist_ok=True)

    # vary # of known batches at three levels
    # and set the number of batches that are known batches
    random.seed(random_list[299])
    red_light_batch_index = random.choice(cfg.red_light_batch_indices[specline.red_light_level])
    
    # Number of novel/unknown examples per batch for each novel batch    
    list_novelsize = create_list_novel_sizes(cfg, specline, red_light_batch_index)  

    # Number of known examples per batch for each batch
    list_knownsize = (cfg.batch_size - np.array(list_novelsize)).tolist()
    list_knownsize = [cfg.batch_size] * red_light_batch_index + list_knownsize

    print("red_light_batch_index {}, list_novelsize {}, list_knownsize {}".format(
        red_light_batch_index, list_novelsize, list_knownsize))
    
    knowns, unknowns = get_knowns_and_unknowns(cfg, specline, df_main, random_list)

    # Set the novelty columns (aren't these columns redundant?)
    set_novelty_type_columns(specline, df_main, knowns, unknowns)

    # Sample from anonfile
    df_known, df_unknown = draw_from_dataset_by_task(
        cfg, specline, df_main, random_list, 
        knowns, unknowns, list_knownsize, list_novelsize)
    
    specline_sanity_checks(cfg, specline, knowns, unknowns)    

    print("Running activity gen test class for task {},\
 novelty_type {}, test {}, attr_known {}, attr_unknown {}, n_random_known_classes {}, knowns {} unknowns {}\
 df_known {} df_unknown {}".format(
        specline.task, specline.novelty_type, str(specline.test_id), 
        specline.known_sources, specline.unknown_sources, specline.n_random_known_classes, 
        knowns, unknowns, len(df_known), len(df_unknown)))        
    
    # Generate n tests by shuffling the order
    if specline.task == "hwr":
        n_groups = cfg.hwr.group_sizes[specline.novelty_type]
    else:
        n_groups = cfg.ar.group_sizes[specline.novelty_type]

    for grp in range(n_groups):
        df_out = create_shuffled_test_dataframe(
            cfg, random_list, grp, df_known, df_unknown, outcols, red_light_batch_index, list_knownsize, list_novelsize)
        
        #writer out test file and meta file
        outfile = specline.protocol + '.' + str(grp) + '.' + str(specline.test_id) + '.' + str(specline.seed)
        df_out[outcols2].to_csv(
            os.path.join(outpath, outfile + '_single_df.csv'), index=False)

        if specline.task == 'hwr':
            kc = cfg.hwr.known_classes_ontologyid # <kc are known classes by ontology id, >=kc are unknown
        else:
            kc = cfg.ar.known_classes_ontologyid  
            
        if specline.novelty_type == "class":
            unknowns_sc = unknowns
        else:
            unknowns_sc = eval(specline.unknown_sources)

        data = {
            "protocol": specline.protocol,
            "known_classes": str(kc), # 88 for AR, 50 for hwr
            "novel_classes": str(len(unknowns_sc)),
            "actual_novel_classes": unknowns_sc,                        
            "max_novel_classes": str(max(cfg.max_novel_classes, len(unknowns_sc))),

            "distribution": specline.dist_type, # High/flat
            "prop_novel": str(specline.prop_unknown), # Proportion of novelty       

            "degree": str(1),

            "detection": str(np.nonzero(df_out["novel"].to_numpy())[0][0]),
            "red_light": str(df_out['annonymous_id'].iloc[cfg.batch_size * red_light_batch_index]),
            "red_light_index": str(cfg.batch_size * red_light_batch_index),

            "n_rounds": str(cfg.batch_number), # number of batches/rounds
            "round_size": str(cfg.batch_size), # number of examples in each batch/round
            "pre_novelty_batches": str(min(cfg.pre_novelty_batches, red_light_batch_index)),
            "feedback_max_ids": str(int(cfg.feedback_max_ids_fraction * cfg.batch_number)),

            "seed": str(specline.seed)
        }
        print(data)
        meta_file = os.path.join(outpath, outfile + '_metadata.json')
        print("Writing to {}".format(meta_file))
        with open(meta_file, 'w') as fp:
            json.dump(data, fp, indent=2)
            
        check_assertions(cfg, specline, data, df_out)

            
def check_if_test_files_exist(cfg, specline):
    if specline.task == 'hwr':
        paths = cfg.hwr.paths
    else:
        paths = cfg.ar.paths    
    outpath = os.path.join(paths[specline.novelty_type], str(specline.test_id))

    if specline.task == "hwr":
        n_groups = cfg.hwr.group_sizes[specline.novelty_type]
    else:
        n_groups = cfg.ar.group_sizes[specline.novelty_type]
    
    for grp in range(n_groups):
        outfile = specline.protocol + '.' + str(grp) + '.' + str(specline.test_id) + '.' + str(specline.seed)
        datafile = os.path.join(outpath, outfile + '_single_df.csv')
        metafile = os.path.join(outpath, outfile + '_metadata.json')
        if not os.path.isfile(datafile): return False
        if not os.path.isfile(metafile): return False
    return True        


def prepare_data(cfg, task):
    """ Load anon file """
    if task == 'hwr':
        df_main = pd.read_csv(cfg.hwr.anonfile)

        df_main = df_main.loc[~df_main['source'].isin(cfg.hwr.source_exclude)]
        df_main['filename'] = df_main['path'].apply(lambda x: x.split('/')[-1])
        df_main['source'] = df_main['source'].apply(lambda x: x.split('-')[1])

        ###################
        #load string file here and merge it to df_main
        #Need lookup file for lines_generated
        ###################
        ###Double check this###
        #Most test from lines_generated
        #df_lines_fixed to re-generate some M12 tests   
        df_lines = pd.concat(list(map(lambda x: pd.read_csv(x, quotechar='|'), cfg.hwr.lines_files)))
        df_main = pd.merge(df_main, df_lines[['filename','text']], on='filename', how='left')

        #Known/training sample
        #df_holdout = df_main.loc[df_main.usage=='train']
        #Unknown/testing sample
        #df         = df_main.loc[df_main.usage=='test']

    else: # activity recognition
        df_main = pd.read_csv(cfg.ar.anonfile)

    return df_main

#read in specification file to generate tests
#specfile is a filename of specification
def generate_tests(cfg, specfile, istart=0, iend=None, die_on_error=True, skip_if_exists=False):
    """
    Given the config file, the test specifications file, generate tests for each
    specification in the OND protocol.
    cfg: configuration params
    specfile: specifications file
    istart: start of this line of the specfile
    die_on_error: die if error when processing a line in the spec on not
    skip_if_exists: skip if file exists
    """
    spec = pd.read_csv(specfile)
    print("Read file {} with {} lines.".format(specfile, len(spec)))
    if spec.shape[0] != spec['test_id'].nunique():
        print('Check for the duplicate text number')
        return
    else:
        df_main = prepare_data(cfg, spec.loc[0,"task"])
        if iend is None: iend = len(spec)
        for i in range(istart, iend):
            print("Specfile line {}".format(i+1))
            specline = spec.loc[i]
            
            test_files_exist = check_if_test_files_exist(cfg, specline)
            if skip_if_exists and test_files_exist:
                print("Tests for specfile line {} exists; skipping.".format(i+1))
                continue                      
            
            try:
                generate_test_group(cfg, df_main, specline)
                spec.loc[i,"complete"] = 1 # completion
            except:
                spec.loc[i,"complete"] = 2 # error flag
                print("ERROR: generate_test_group failed")
                if die_on_error:
                    raise
                else:
                    continue
    print("Done")
    incompletes = spec["complete"]==2
    if incompletes.sum() > 0:
        print("There are {} incomplete specifications:".format(incompletes.sum()))
        #print(incompletes)
    return spec
