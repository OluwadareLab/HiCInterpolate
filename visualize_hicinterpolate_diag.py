from test_hicinterpolate_diag import parse_args, run_inference, set_seed


if __name__ == "__main__":
    set_seed(42)
    args = parse_args()
    run_inference(
        organism_filter=args.organism,
        chromosome_filter=args.chromosome,
        save_plots=not args.no_plots,
        overwrite=args.overwrite,
    )
