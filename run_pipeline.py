"""CLI entry point: python run_pipeline.py [--input path] [--out output] [--lags 1 2]"""
import argparse

from msci_em.pipeline import Config, run_pipeline


def main():
    ap = argparse.ArgumentParser(description="MSCI EM stock-weight prediction pipeline")
    ap.add_argument("--input", default="data/raw/MSCI_EM_Portfolio_since_Dec-21.xlsx")
    ap.add_argument("--out", default="output")
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 2],
                    help="lag lengths (periods) for return / share-change features")
    ap.add_argument("--no-vol", action="store_true", help="drop the volatility feature")
    ap.add_argument("--n-sims", type=int, default=1000)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = Config(lags=tuple(args.lags), n_sims=args.n_sims, use_vol=not args.no_vol)
    run_pipeline(args.input, cfg=cfg, out_dir=args.out, verbose=not args.quiet)


if __name__ == "__main__":
    main()
