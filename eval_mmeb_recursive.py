import sys

import eval_mmeb


def _has_recursive_flag(argv):
    for i, arg in enumerate(argv):
        if arg == "--recursive_eval_steps":
            return True
        if arg.startswith("--recursive_eval_steps="):
            return True
        # handle "--recursive-eval-steps" alias style if passed by mistake
        if arg == "--recursive-eval-steps":
            return True
    return False


def main():
    # Dedicated entrypoint for recursive-style evaluation.
    # If user does not pass a value explicitly, default to 2 steps.
    if not _has_recursive_flag(sys.argv):
        sys.argv.extend(["--recursive_eval_steps", "2"])
        print("[eval_mmeb_recursive] --recursive_eval_steps not provided, defaulting to 2")
    eval_mmeb.main()


if __name__ == "__main__":
    main()
