from __future__ import annotations

import sys

from .catalog import info, kernels


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] == "list":
        pat = args[1] if len(args) > 1 else ""
        for k in kernels(pat):
            print(f"{k['lookup']:32s} kernarg={k['kernarg_size']:4d} "
                  f"lds={k['group_segment']:5d} priv={k['private_segment']:4d}")
        return 0
    if args[0] == "info":
        import json
        print(json.dumps(info(args[1]), ensure_ascii=False, indent=2))
        return 0
    print(__doc__ or "用法: python3 -m k100lc_kernels [list|info NAME]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
