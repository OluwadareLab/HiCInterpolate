import argparse
from downstream_analysis import run_compartment, run_embedtad, run_hiccups, run_hicgnn


def main(args):
    if args.input is None:
        print("Input file is missing")
        exit
    if args.output is None:
        print("output folder is missing")
        exit

    if args.ab_comp:
        run_compartment.run_ab_compartment(
            args.input, args.res, args.start, args.end, args.output)
    if args.loop:
        run_hiccups.run_hiccups(args.input, args.output, args.res, args.chrom, args.gid)
    if args.tads:
        run_embedtad.run_embedtad(args.input, args.output, args.res)
    if args.structure:
        run_hicgnn.run_hicgnn(args.input, args.output, args.start, args.end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='HiCInterpolate downstream analysis')
    parser.add_argument('-ab', '--ab-compartments', dest="ab_comp",  action='store_true',
                        help='include for A/B compartment analysis')
    parser.add_argument('-l', '--loop', dest="loop", action='store_true',
                        help='include for loop analysis')
    parser.add_argument('-t', '--tads', dest="tads",
                        action='store_true', help='include for TAD analysis')
    parser.add_argument('-s', '--structure', dest="structure",
                        action='store_true', help='include for 3D structure coordinates')
    parser.add_argument('-i', '--input', dest="input",
                        type=str, default=None, help='input nxn square matrix in .txt or .npy format')
    parser.add_argument('-o', '--output', dest="output",
                        type=str, default=None, help='output folder')
    parser.add_argument('-r', '--resolution', dest="res",
                        type=int, default=None, help='resolution or bin size')
    parser.add_argument('-c', '--chromosome', dest="chrom",
                        type=str, default=None, help='chromosome number (required for loops)')
    parser.add_argument('-g', '--genome_id', dest="gid",
                        type=str, default="hg19", help='genome id (required for loops). e.g. hg19, hg38, mm9, mm10')
    parser.add_argument('-sc', '--start', dest="start",
                        type=int, default=None, help='start bin. (required for A/B compartment and 3D structure coordinates)')
    parser.add_argument('-ec', '--end', dest="end",
                        type=int, default=None, help='end bin. (required for A/B compartment and 3D structure coordinates)')
    args = parser.parse_args()

    main(args)
