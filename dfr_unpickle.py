import pickle
import pprint

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Print out a pickle file as produced by dfr_evaluate_spurious.py")
    parser.add_argument(
        "--pkl_path", type=str,
        default="RSLTS_DFR/results.pkl",
        help="Location of pkl file")
    args = parser.parse_args()

    dbfile = open(args.pkl_path, 'rb')
    db = pickle.load(dbfile)
    dbfile.close()
    pprint.pprint(db)