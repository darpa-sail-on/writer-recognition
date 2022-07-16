mport numpy as np
import pandas as pd
import scipy.stats
import os
import argparse
import json
import glob
import ast
import random
from types import SimpleNamespace
import csv

# Functions used to generate tests for activity recognition and handwriting recognition
# based on the specifications file and the anonymization file.

def check_assertions(cfg, specline, df_main, df_main_id, md, df_out):
    """
    Verify that the output data csv and metadata json is consistent with
    the specification line and anonfile
    """
    # {'protocol': 'OND',
    #  'known_classes': 88,
    #  'novel_classes': 3,
    #  'actual_novel_classes': ['ArmWrestling', 'BoxingSpeedBag', 'Wrestling'],
    #  'max_novel_classes': 4,
    #  'distribution': 'flat',
    #  'prop_novel': 0.3,
    #  'degree': 1,
    #  'detection': 419,
    #  'red_light': 'fe4b07d0-8949-4e8f-82cc-ce2562764c93.mp4',
    #  'n_rounds': 40,
    #  'round_size': 32,
    #  'pre_novelty_batches': 5,
    #  'feedback_max_ids': 4,
    #  'seed': 6438158}
    
    tcfg = cfg[specline["task"]]

    df_typicalworld = df_out[df_out["detection"]==0]
    df_novelworld = df_out[df_out["detection"]==1]
    df_typical = df_out[df_out["novel"]==0]
    df_novel = df_out[df_out["novel"]==1]

    #df_up = df_main_id.loc[df_out["annonymous_id"]]
    df_lu_typicalworld = df_main_id.loc[df_typicalworld["annonymous_id"]]
    df_lu_novelworld = df_main_id.loc[df_novelworld["annonymous_id"]]
    df_lu_typical = df_main_id.loc[df_typical["annonymous_id"]]
    df_lu_novel = df_main_id.loc[df_novel["annonymous_id"]]
    
    assert(md["protocol"] == "OND")
    assert(int(md["known_classes"]) == tcfg["known_classes_ontologyid"])

       
    if specline["novelty_type"] == "nonnovel":
        assert((df_lu_typicalworld["ontology_id"] < tcfg["known_classes_ontologyid"]).all())
        assert((df_lu_typical["ontology_id"] < tcfg["known_classes_ontologyid"]).all())
        assert((df_lu_novel["ontology_id"]  <  tcfg["known_classes_ontologyid"]).all())

        emp_prop_novel = (df_lu_novelworld["ontology_id"] < tcfg["known_classes_ontologyid"]).mean()
        emp_actual_novel_classes = df_lu_novel["class"].unique()
        assert(len(emp_actual_novel_classes) <= int(md["novel_classes"]))
        assert(set(emp_actual_novel_classes) <= set(md["actual_novel_classes"]))
    elif specline["novelty_type"] == "class":
        assert((df_lu_typicalworld["ontology_id"] < tcfg["known_classes_ontologyid"]).all())
        assert((df_lu_typical["ontology_id"] < tcfg["known_classes_ontologyid"]).all())
        assert((df_lu_novel["ontology_id"] >= tcfg["known_classes_ontologyid"]).all())

        emp_prop_novel = (df_lu_novelworld["ontology_id"] >= tcfg["known_classes_ontologyid"]).mean()
        assert((abs(emp_prop_novel - float(md["prop_novel"])) < 0.1), "emp_prop_novel: {} prop_novel: {}".format(emp_prop_novel, md["prop_novel"]))

        emp_actual_novel_classes = df_lu_novel["class"].unique()
        assert(len(emp_actual_novel_classes) <= int(md["novel_classes"]))
        assert(set(emp_actual_novel_classes) <= set(md["actual_novel_classes"]))
        assert(len(df_typicalworld) == int(md["detection"])) 
        assert(np.nonzero((df_out["annonymous_id"] == md["red_light"]).to_numpy())[0][0] == int(md["detection"]))
        
    else:
        assert((df_out["ontology_id"] < tcfg["known_classes_ontologyid"]).all())    

        assert(df_lu_typicalworld["source"].isin(eval(specline["known_sources"])).all())
        assert(df_lu_typical["source"].isin(eval(specline["known_sources"])).all())
        assert((df_lu_novel["source"].isin(eval(specline["unknown_sources"])).all()))

        emp_prop_novel = df_lu_novelworld["source"].isin(eval(specline["unknown_sources"])).mean()     
        assert((abs(emp_prop_novel - float(md["prop_novel"])) < 0.1), "emp_prop_novel: {} prop_novel: {}".format(emp_prop_novel, md["prop_novel"]))

        emp_actual_novel_sources = df_lu_novel["source"].unique()
        assert(len(emp_actual_novel_sources) <= int(md["novel_classes"]))
        assert(set(emp_actual_novel_sources) <= set(md["actual_novel_classes"]))
        assert(len(df_typicalworld) == int(md["detection"])) 
        assert(np.nonzero((df_out["annonymous_id"] == md["red_light"]).to_numpy())[0][0] == int(md["detection"]))
    
    assert(len(df_out["annonymous_id"]) == len(df_out["annonymous_id"].drop_duplicates()))
    assert(len(df_out) == int(md["round_size"]) * int(md["n_rounds"]))
    assert(len(df_typicalworld) >= int(md["round_size"]) * int(md["pre_novelty_batches"]))

    if specline["task"] == "ar":
        if specline["novelty_type"] in ["class","nonnovel"]:
            assert((df_out["spatial"] == 0).all())
            assert((df_out["temporal"] == 0).all())
        if specline["novelty_type"] == "spatial":
            assert(np.all(df_out["novel"] == df_out["spatial"]))
            assert((df_out["temporal"] == 0).all())
        if specline["novelty_type"] == "temporal":
            assert((df_out["spatial"] == 0).all())
            assert(np.all(df_out["novel"] == df_out["temporal"]))
        
    if specline["task"] == "hwr":
        if specline["novelty_type"] in ["class","nonnovel"]:
            assert((df_out["letter"] == 0).all())
            assert((df_out["background"] == 0).all())
        if specline["novelty_type"] == "letter":
            assert(np.all(df_out["novel"] == df_out["letter"]))
            assert((df_out["background"] == 0).all())
        if specline["novelty_type"] == "background":
            assert((df_out["letter"] == 0).all())
            assert(np.all(df_out["novel"] == df_out["background"]))
        
    return


def specline_sanity_checks(cfg, sp, known_classes, unknown_classes):
    """ Check the inputs for a given line in the specifications file """
    assert(sp["protocol"] == "OND")
    assert(sp["task"] in ["hwr","ar"])
    if sp["task"] == "hwr":
        assert(sp["novelty_type"] in ["class","nonnovel","letter","background"])
    else:
        assert(sp["novelty_type"] in ["class","nonnovel","spatial","temporal"])
    assert(sp["red_light_level"] in ["E","M","H"])
    assert(sp["dist_type"] in ["low","mid","high","flat"])

    if sp["novelty_type"] == "nonnovel":
        df_kg = pd.read_csv(
            cfg[sp["task"]]["knowledge_graph_file"], header=None, 
            names = ["ontology_id","class","type","class_id"]).set_index("class")
        assert(np.all(df_kg.loc[known_classes]["ontology_id"] < cfg[sp["task"]]["known_classes_ontologyid"]))
        assert(np.all(df_kg.loc[unknown_classes]["ontology_id"] < cfg[sp["task"]]["known_classes_ontologyid"]))
    else:
        df_kg = pd.read_csv(
            cfg[sp["task"]]["knowledge_graph_file"], header=None, 
            names = ["ontology_id","class","type","class_id"]).set_index("class")
        assert(np.all(df_kg.loc[known_classes]["ontology_id"] < cfg[sp["task"]]["known_classes_ontologyid"]))
        assert(np.all(df_kg.loc[unknown_classes]["ontology_id"] >= cfg[sp["task"]]["known_classes_ontologyid"]))
        
    return


def create_list_novel_sizes(cfg, sp, red_light_batch_index):
    """
        Use beta distribution to determine where novel examples appear.

    Returns: list x corresponding to the novel batches, where x[i] is the number of novel examples 
     for the i^th batch (where i is the index into the list of novel batches, length(x) = number of novel batches)
    """
    distparams = cfg["beta_dist_params"][sp["dist_type"]]
    
    n_batch_novel  = int((cfg["batch_number"] - red_light_batch_index) * sp["prop_unknown"])
    novel_size     = n_batch_novel * cfg["batch_size"]
    
    # Prepare a probability bin for a given novel distribution
    bin_prob = np.linspace(0, 1, cfg["batch_number"] - red_light_batch_index + 1).tolist()
    list_unknown_sizes = []
    for i in range(len(bin_prob) - 1):
        list_unknown_sizes.append(
            int((scipy.stats.beta.cdf(bin_prob[i+1], distparams[0], distparams[1], loc=0, scale=1) -
                 scipy.stats.beta.cdf(bin_prob[i], distparams[0], distparams[1], loc=0, scale=1)) * novel_size))

    list_unknown_sizes = [max(0,min(cfg["batch_size"],i)) for i in list_unknown_sizes]
    return list_unknown_sizes


def draw_from_dataset(df_full, nsamp, rngstate, replace=False):
    """
        Draw known and unknown samples from dataset, given the anonfile dataframe
        and indices into the dataframe for knowns and unknowns.    
    df_main: anonfile dataframe
    ix: indices into df_main for samples
    nsamp: number of samples
    rngstate: random number generator state
    """
    print("Drawing {} samples from {} rows.".format(nsamp, len(df_full)))
    return df_full.sample(n = nsamp, random_state = rngstate, replace = replace)


def get_anonfile_mask(cfg, sp, df_main, known_classes, unknown_classes, novelty_mode):
    tcfg = cfg[sp["task"]]
    # Next, draw from anonfile based on chosen classes and novelty type    
    if sp["novelty_type"] in ["class","nonnovel"]: # class level novelty
        # Select from all known classes for knowns, unknown classes for unknowns
        if novelty_mode == "known":
            mask = (df_main['class'].isin(known_classes  )) & (df_main["source"].isin(eval(sp["known_sources"])))
        else:
            mask = (df_main['class'].isin(unknown_classes)) & (df_main["source"].isin(eval(sp["unknown_sources"])))
    else: # letter/background novelty for hwr or spatial/temporal novelty for ar
        if novelty_mode == "known":
            mask = (df_main['class'].isin(known_classes)) & (df_main['source'].isin(eval(sp["known_sources"  ])))
        else:
            mask = (df_main['class'].isin(known_classes)) & (df_main['source'].isin(eval(sp["unknown_sources"])))
    
    return mask

def select_known_unknown_classes(cfg, sp, df_main, rng_seed_list, n_known, n_unknown):
    tcfg = cfg[sp["task"]]

    max_tries = 100
    # First choose known and unknown classes either from spec file or randomly
    if pd.isna(sp["n_random_known_classes"]) or sp["n_random_known_classes"] == 0: # this is a flag to manually specify known classes
        known_classes = [] if pd.isna(sp["known_classes"]) else eval(sp["known_classes"])
        assert(all(x not in tcfg["known_class_exclude"] for x in known_classes))
    else: # randomly select n_random_known_classes known classes
        np.random.seed(rng_seed_list[10])
        for it in range(max_tries):            
            known_classes = list(np.random.choice(
                df_main['class'][
                    (df_main['ontology_id'] < tcfg["known_classes_ontologyid"]) & (~df_main['class'].isin(tcfg["known_class_exclude"]))
                ].unique().tolist(), sp["n_random_known_classes"]))
            if get_anonfile_mask(cfg, sp, df_main, known_classes, None, "known").sum() >= n_known:
                break
        print("IT:",it)

    if pd.isna(sp["n_random_unknown_classes"]) or sp["n_random_unknown_classes"] == 0:
        unknown_classes = [] if pd.isna(sp["unknown_classes"]) else eval(sp["unknown_classes"])
        assert(all(x not in tcfg["unknown_class_exclude"] for x in unknown_classes))
    else: # randomly select n_random_known_classes known classes
        np.random.seed(rng_seed_list[11])
        for it in range(max_tries):            
            unknown_classes = list(np.random.choice(
                df_main['class'][
                    (df_main['ontology_id'] >= tcfg["known_classes_ontologyid"]) & (~df_main['class'].isin(tcfg["unknown_class_exclude"]))
                ].unique().tolist(), sp["n_random_unknown_classes"]))
            if get_anonfile_mask(cfg, sp, df_main, known_classes, unknown_classes, "unknown").sum() >= n_unknown:
                break
        print("IT:",it)
            
    return known_classes, unknown_classes


def draw_from_dataset_by_task(cfg, sp, df_main, rng_seed_list, list_known_sizes, list_unknown_sizes):
    """
        Draw lines from anonfile
    """
    set_novelty_type_columns(sp, df_main)

    tcfg = cfg[sp["task"]]
    
    n_known = sum(list_known_sizes)
    n_unknown = sum(list_unknown_sizes)

    known_classes, unknown_classes = select_known_unknown_classes(
        cfg, sp, df_main, rng_seed_list, n_known, n_unknown)

    print("Running activity gen test class for task {},\
 novelty_type {}, test {}, attr_known {}, attr_unknown {}, n_random_known_classes {}, known_classes {} unknown_classes {}".format(
        sp["task"], sp["novelty_type"], str(sp["test_id"]),
        sp["known_sources"], sp["unknown_sources"], sp["n_random_known_classes"],
        known_classes, unknown_classes))

    mask_known = get_anonfile_mask(cfg, sp, df_main, known_classes, None, "known")
    mask_unknown = get_anonfile_mask(cfg, sp, df_main, known_classes, unknown_classes, "unknown")
    df_known = draw_from_dataset(df_main[mask_known], n_known, rng_seed_list[1])
    df_unknown = draw_from_dataset(df_main[mask_unknown], n_unknown, rng_seed_list[2])

    return known_classes, unknown_classes, df_known, df_unknown


def set_novelty_type_columns(sp, df_main):
    # Set the novel changes column
    if sp["task"]=="hwr":
        if sp["novelty_type"] in  ["class","nonnovel"]:
            df_main['letter']     = 0
            df_main['background'] = 0
        elif sp["novelty_type"] == "letter":
            df_main['letter']     = (df_main['source'].isin(eval(sp["unknown_sources"]))).astype(np.int)
            df_main['background'] = 0
        elif sp["novelty_type"] == "background":
            df_main['letter']     = 0 
            df_main['background'] = (df_main['source'].isin(eval(sp["unknown_sources"]))).astype(np.int)  
        else:
            print("Novelty level bad")
            raise(Exception())

        appearance_column = np.zeros(len(df_main), dtype=np.int)
        if sp["novelty_type"] == "background":
            appearance_column[df_main["source"].apply(lambda x: x.startswith("iam-bg_"))] = 3
        if sp["novelty_type"] == "letter":
            appearance_column[df_main["source"].apply(lambda x: x.startswith("iam-letter_"))] = 2
            appearance_column[df_main["source"].apply(lambda x: x.startswith("iam-letter_inverted"))] = 1
            appearance_column[df_main["source"].apply(lambda x: x.startswith("iam-letter_slant"))] = 4
            appearance_column[df_main["source"].apply(lambda x: x.startswith("iam-letter_flip"))] = 5
        appearance_column[df_main["source"].apply(lambda x: x.startswith("iam-letter_clean"))] = 0
        df_main["appearance"] = appearance_column

    else: # activity recognition
        if sp["novelty_type"] in  ["class","nonnovel"]:
            df_main['spatial']  = 0
            df_main['temporal'] = 0
        elif sp["novelty_type"] == "spatial":
            df_main['spatial']  = df_main['source'].isin(eval(sp["unknown_sources"])).astype(np.int) # source in novelty list
            df_main['temporal'] = 0
        elif sp["novelty_type"] == "temporal":
            df_main['spatial']  = 0
            df_main['temporal'] = df_main['source'].isin(eval(sp["unknown_sources"])).astype(np.int)
        else:
            raise(Exception("Novelty level bad"))


def create_shuffled_test_dataframe(
    cfg, rng_seed_list, grp, df_known, df_unknown, outcols, red_light_batch_index, list_known_sizes, list_unknown_sizes):
    """
    Given a dataframe of non-novel examples (df_known) and novel examples (df_unknown), merge the two dataframes,
    then shuffle them, set detection point, novelty flag.
    """
    #Shuffle data:
    df_known            = df_known.sample(frac=1, random_state=rng_seed_list[100+grp], replace=False)
    df_unknown          = df_unknown.sample(frac=1, random_state=rng_seed_list[200+grp], replace=False)
    #temporary group for sorting
    df_known['group']   = np.repeat([-1] * red_light_batch_index + list(range(cfg["batch_number"] - red_light_batch_index)), list_known_sizes)
    df_unknown['group'] = np.repeat(list(range(len(list_unknown_sizes))), list_unknown_sizes)
    #assign novel label
    df_known['novel']   = 0
    df_unknown['novel'] = 1

    # Combined output based on the assigned distribution of novel instances
    # Reorder by group 
    # Add detection point
    df_out = pd.concat([df_unknown[outcols], df_known[outcols]],axis=0).sort_values('group')

    #Shuffle known & unknown order within a group
    df_out = df_out.groupby("group").apply(
        lambda group_df: group_df.sample(
            frac=1,random_state=rng_seed_list[50+grp])).reset_index(drop=True)  
    
    detection_column = np.zeros(len(df_out), dtype=np.int)
    nonzeros = np.nonzero(df_out["novel"].to_numpy())[0]
    if len(nonzeros) > 0:
        detection_index = np.nonzero(df_out["novel"].to_numpy())[0][0]
        detection_column[detection_index:] = 1
    df_out['detection'] = detection_column    

    return df_out


def write_single_df(tcfg, df_out, outpath, outfile):
    fd = open(os.path.join(outpath, outfile + '_single_df.csv'), "w")
    
    fd.write(",".join(tcfg["output_columns"]) + "\n")
    
    for (i,row) in df_out.iterrows():
        fd.write(",".join(map(str, row[tcfg["output_columns"]])) + "\n")
        
    fd.close()


def generate_test_group(cfg, df_main, df_main_id, specline, outpath):
    """
        Given a set of parameters from a given line in the specification, generate a 
        group of tests wrt the parameters. Each test in the group is a shuffled version
        of the other tests in the group.
    
    cfg: configuration params
    df_main: anonfile dataframe
    specline: line in the specification file
    """
    tcfg = cfg[specline["task"]]
       
    # Generate the master random number list based on the master seed on the spec sheet
    # This is a test level seeds
    np.random.seed(specline["seed"])
    rng_seed_list = np.random.randint(999999, size=300)

    # Create output folder if not exists
    #paths = tcfg["paths"]
    #outpath = os.path.join(paths[specline["novelty_type"]], str(specline["test_id"]))
    outpath = os.path.join(outpath, str(specline["test_id"]))
    print("Output path: {}".format(outpath))
    os.makedirs(outpath, exist_ok=True)

    # vary # of known batches at three levels
    # and set the number of batches that are known batches
    random.seed(rng_seed_list[299])
    red_light_batch_index = random.choice(cfg["red_light_batch_indices"][specline["red_light_level"]])
    
    # Number of novel/unknown examples per batch for each novel batch    
    list_unknown_sizes = create_list_novel_sizes(cfg, specline, red_light_batch_index)  

    # Number of known examples per batch for each batch
    list_known_sizes = (cfg["batch_size"] - np.array(list_unknown_sizes)).tolist()
    list_known_sizes = [cfg["batch_size"]] * red_light_batch_index + list_known_sizes
    
    # Sample from anonfile
    known_classes, unknown_classes, df_known, df_unknown = draw_from_dataset_by_task(
        cfg, specline, df_main, rng_seed_list,
        list_known_sizes, list_unknown_sizes)

    # do sanity checks for input
    specline_sanity_checks(cfg, specline, known_classes, unknown_classes)    

    if specline["task"] == "hwr":
        df_known["text"] = list(map(lambda x: "|{}|".format(x), df_known["text"]))
        df_unknown["text"] = list(map(lambda x: "|{}|".format(x), df_unknown["text"]))

    # Generate n tests by shuffling the order
    n_groups = tcfg["group_sizes"][specline["novelty_type"]]

    for grp in range(n_groups):
        df_out = create_shuffled_test_dataframe(
            cfg, rng_seed_list, grp, df_known, df_unknown, tcfg["selected_columns"],
            red_light_batch_index, list_known_sizes, list_unknown_sizes)

        if specline["novelty_type"] in  ["class","nonnovel"]:
            unknowns_sc = unknown_classes
        else:
            unknowns_sc = eval(specline["unknown_sources"])

        detection_index = -1
        nonzeros = np.nonzero(df_out["novel"].to_numpy())[0]
        if len(nonzeros) > 0:
            detection_index = nonzeros[0]
           

        data = {
            "protocol": specline["protocol"],
            "known_classes": str(tcfg["known_classes_ontologyid"]), # 88 for AR, 50 for hwr
            "novel_classes": str(len(unknowns_sc)),
            "actual_novel_classes": unknowns_sc,                        
            "max_novel_classes": str(max(cfg["max_novel_classes"], len(unknowns_sc))),

            "distribution": specline["dist_type"], # High/flat
            "prop_novel": str(specline["prop_unknown"]), # Proportion of novelty       

            "degree": str(1),

            "detection": str(detection_index),
            "red_light": str(df_out['annonymous_id'].iloc[detection_index]) if detection_index>=0 else "",

            "n_rounds": str(cfg["batch_number"]), # number of batches/rounds
            "round_size": str(cfg["batch_size"]), # number of examples in each batch/round
            "pre_novelty_batches": str(min(cfg["pre_novelty_batches"], red_light_batch_index)),
            "feedback_max_ids": str(int(cfg["feedback_max_ids_fraction"] * cfg["batch_number"])),

            "seed": str(specline["seed"])
        }
        
        check_assertions(cfg, specline, df_main, df_main_id, data, df_out)
        
        data = {
            "protocol": specline["protocol"],
            "known_classes": tcfg["known_classes_ontologyid"], # 88 for AR, 50 for hwr
            "novel_classes": len(unknowns_sc),
            "actual_novel_classes": unknowns_sc,                        
            "max_novel_classes": max(cfg["max_novel_classes"], len(unknowns_sc)),

            "distribution": specline["dist_type"], # High/flat
            "prop_novel": float(specline["prop_unknown"]), # Proportion of novelty       

            "degree": 1,

            "detection": int(detection_index),
            "red_light": df_out['annonymous_id'].iloc[detection_index],

            "n_rounds": cfg["batch_number"], # number of batches/rounds
            "round_size": cfg["batch_size"], # number of examples in each batch/round
            "pre_novelty_batches": min(cfg["pre_novelty_batches"], red_light_batch_index),
            "feedback_max_ids": int(cfg["feedback_max_ids_fraction"] * cfg["batch_number"]),

            "seed": int(specline["seed"])
        }

        #writer out test file and meta file
        outfile = specline["protocol"] + '.' + str(grp) + '.' + str(specline["test_id"]) + '.' + str(specline["seed"])
        
        meta_file = os.path.join(outpath, outfile + '_metadata.json')
        print("Writing to {}".format(meta_file))
        with open(meta_file, 'w') as fp:
            json.dump(data, fp, indent=2)

        write_single_df(tcfg, df_out, outpath, outfile)


def check_if_test_files_exist(cfg, specline, outpath):
    """ Return true iff the output files associated with the specification line exists. """
    tcfg = cfg[specline["task"]]

    #paths = tcfg["paths"]
    #outpath = os.path.join(paths[specline["novelty_type"]], str(specline["test_id"]))
    outpath = os.path.join(outpath, str(specline["test_id"]))

    n_groups = tcfg["group_sizes"][specline["novelty_type"]]
    
    for grp in range(n_groups):
        outfile = specline["protocol"] + '.' + str(grp) + '.' + str(specline["test_id"]) + '.' + str(specline["seed"])
        datafile = os.path.join(outpath, outfile + '_single_df.csv')
        metafile = os.path.join(outpath, outfile + '_metadata.json')
        if not os.path.isfile(datafile): return False
        if not os.path.isfile(metafile): return False
    return True        


def load_anonfile(cfg, task):
    """ Load anon file and process the data. """
    tcfg = cfg[task]

    # Read anonfile(s)
    # Priority order: anonfile, anonfiles, anondir
    filereading = "unknown"
    try:
        anonname = tcfg["anonfile"]
        filereading = "file"
    except:
        try:
            anonname = tcfg["anonfiles"]
            filereading = "files"
        except:
            anonname = tcfg["anondir"]
            filereading = "dir"

    if filereading == "file":
        print("Reading anonfile {}".format(tcfg["anonfile"]))
        df_main = pd.read_csv(tcfg["anonfile"])

    elif filereading == "files":
        dfs = []
        for file in tcfg["anonfiles"]:
            print("Reading anonfile {}".format(file))
            dfs.append(pd.read_csv(file))
        df_main = pd.concat(dfs)            

    elif filereading == "dir":
        dfs = []
        for file in glob.glob(os.path.join(tcfg["anondir"], "*.csv")):
            print("Reading anonfile {}".format(file))
            dfs.append(pd.read_csv(file))
        df_main = pd.concat(dfs)

    else:
        raise(Exception("No anonfile set in cfg"))
        
    df_main = df_main.loc[~df_main['source'].isin(tcfg["source_exclude"])]
    df_main = df_main.drop_duplicates("annonymous_id")

    if task == "hwr":
        if df_main["class"].dtype == np.int:
            df_main['class'] = df_main['class'].apply(lambda x: "%03d"%x)

        # Use lines_files to look up the text associated with handwriting
        df_lines = pd.concat(list(map(lambda x: pd.read_csv(x, quotechar='|'), cfg["hwr"]["lines_files"])))
        df_main['filename'] = df_main['path'].apply(lambda x: x.split('/')[-1])
        df_main = pd.merge(df_main, df_lines[['filename','text']], on='filename', how='left')

    return df_main


def generate_tests(
    cfg, specfile, outpath,
    istart=0, iend=None, die_on_error=True, skip_if_exists=False):
    """
        Given the config file, the test specifications file, generate tests for each
        specification in the OND protocol.
    cfg: configuration params
    specfile: specifications file
    istart: start of this line of the specfile
    die_on_error: die if error when processing a line in the spec, else ignore error
    skip_if_exists: skip if output file exists
    """
    spec = pd.read_csv(specfile)
    print("Read file {} with {} lines.".format(specfile, len(spec)))
    if spec.shape[0] != spec['test_id'].nunique():
        print('Check for the duplicate text number')
        return

    assert(len(spec["task"].unique()) == 1)
    df_main = load_anonfile(cfg, spec.loc[0,"task"])
    df_main_id = df_main.set_index("annonymous_id")
    
    if iend is None: iend = len(spec)
    for i in range(istart, iend):
        print("Specfile line {}".format(i+1))
        specline = spec.loc[i]

        test_files_exist = check_if_test_files_exist(cfg, specline, outpath)
        if skip_if_exists and test_files_exist:
            print("Tests for specfile line {} exists; skipping.".format(i+1))
            continue                      

        try:
            generate_test_group(cfg, df_main, df_main_id, specline, outpath)
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
    return spec

