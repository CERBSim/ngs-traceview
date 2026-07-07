import os
import sys

from ngapp.cli.serve_standalone import main as startup


def main():
    # allow `ngs-traceview myfile.trace` / `python -m ngs_traceview myfile.trace`
    # — serve_standalone's argparse doesn't accept positionals, so pass the
    # path via environment
    for arg in list(sys.argv[1:]):
        if not arg.startswith("-") and os.path.exists(arg):
            os.environ["NGS_TRACEVIEW_FILE"] = os.path.abspath(arg)
            sys.argv.remove(arg)
    startup(app_module="ngs_traceview.appconfig")


if __name__ == "__main__":
    main()
