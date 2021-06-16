# interface module for matlab
import json
import utils
import csv
from markov import mkv


def compute_matlab_tps(file_input, separator, dir_out):
    sequences = utils.read_from_file(file_input, separator)
    # compute transitions frequencies
    tf_agent = mkv.markov_trans_freq(sequences[:10])
    tf_tot = mkv.markov_trans_freq(sequences)

    # with open(dir_out + "agent_sym_freq.json", "w") as fp:
    #     json.dump(tf_agent[0], fp, default=mkv.serialize_sets, ensure_ascii=False)
    # with open(dir_out + "process_sym_freq.json", "w") as fp:
    #     json.dump(tf_tot[0], fp, default=mkv.serialize_sets, ensure_ascii=False)

    with open(dir_out + "agent_tps.json", "w") as fp:
        json.dump(tf_agent, fp, default=mkv.serialize_sets, ensure_ascii=False)
    with open(dir_out + "process_tps.json", "w") as fp:
        json.dump(tf_tot, fp, default=mkv.serialize_sets, ensure_ascii=False)

    choords = list(tf_tot[0].keys())

    with open(dir_out + 'agent_sym.csv', mode='w') as asf:
        ags_writer = csv.writer(asf, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ags_writer.writerow(tf_agent[0].keys())
        ags_writer.writerow([x[1] for x in tf_agent[0].items()])

    with open(dir_out + 'process_sym.csv', mode='w') as psf:
        ps_writer = csv.writer(psf, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ps_writer.writerow(tf_tot[0].keys())
        ps_writer.writerow([x[1] for x in tf_tot[0].items()])

    with open(dir_out + 'agent.csv', mode='w') as af:
        agent_writer = csv.writer(af, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        agent_writer.writerow(["sym"] + choords)
        for itm in tf_agent[1].items():
            agent_writer.writerow([itm[0]] + _create_coords(itm[1], choords))

    with open(dir_out + 'process.csv', mode='w') as tf:
        tot_writer = csv.writer(tf, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        tot_writer.writerow(["sym"] + choords)
        for itm in tf_tot[1].items():
            tot_writer.writerow([itm[0]] + _create_coords(itm[1], choords))


def _create_coords(a_dict, dims):
    res = []
    for x in dims:
        if x in a_dict.keys():
            res.append(float(a_dict[x]))
        else:
            res.append(0)
    return res


compute_matlab_tps("C:/Users/matti/OneDrive/Desktop/DOTT/projects/NohGen2/matlab/irish.txt",
                   " ",
                   "C:/Users/matti/OneDrive/Desktop/DOTT/projects/NohGen2/matlab/")
