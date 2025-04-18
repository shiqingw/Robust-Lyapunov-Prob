import sys
import os
from pathlib import Path

def generate_sh_script_dreal_stability_solve(filename, exp_nums, dreal_precision):
    lines = []
    # 1) shebang + precision variable
    lines.append("#!/usr/bin/env bash\n")
    lines.append(f"PRECISION={dreal_precision:.1E}\n")

    # 2) helper log function
    lines.append("log(){ echo \"[$(date +'%Y-%m-%d %H:%M:%S')] $*\" >> \"$LOG\"; }\n")

    # 3) traps to forward SIGTERM/SIGINT
    lines.append("trap 'log \"Wrapper received SIGTERM; forwarding to $PID\"; kill -TERM $PID' TERM\n")
    lines.append("trap 'log \"Wrapper received SIGINT; forwarding to $PID\";  kill -INT  $PID' INT\n")

    # 4) loop over all experiments
    exp_list = " ".join(str(n) for n in exp_nums)
    lines.append(f"\nfor EXP_NUM in {exp_list}; do")
    lines.append("    # prepare directory + log file")
    lines.append("    DIR=$(printf \"eg2_results/%03d\" \"$EXP_NUM\")")
    lines.append("    LOG=\"$DIR/output_dreal_stability_solve_${PRECISION}.out\"")
    lines.append("    mkdir -p \"$DIR\"\n")

    lines.append("    log \"=== STARTING python: exp_num=$EXP_NUM, precision=$PRECISION ===\"")
    lines.append("    python -u eg2_strict_feedback_3d/dreal_stability_solve.py \\")
    lines.append("        --exp_num $EXP_NUM --dreal_precision $PRECISION >> \"$LOG\" 2>&1 &")
    lines.append("    PID=$!")
    lines.append("    log \"Launched python with PID $PID\"\n")

    lines.append("    # wait and capture exit status")
    lines.append("    wait $PID")
    lines.append("    STATUS=$?\n")

    lines.append("    if [ $STATUS -eq 0 ]; then")
    lines.append("        log \"Python process $PID exited normally (0).\"")
    lines.append("    elif [ $STATUS -gt 128 ]; then")
    lines.append("        SIG=$(( STATUS - 128 ))")
    lines.append("        log \"Python process $PID was killed by signal $SIG.\"")
    lines.append("    else")
    lines.append("        log \"Python process $PID exited with code $STATUS.\"")
    lines.append("    fi\n")

    lines.append("done\n")

    # write out the file
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", newline="\n") as f:
        f.write("\n".join(lines))

    # make it executable
    os.chmod(filename, 0o755)


if __name__ == "__main__":
    exp_nums = [1]       # same as before
    dreal_precision = 1e-3               # same as before

    # place the script one level up from this file
    out_path = Path(__file__).parent.parent / "run_dreal_stability_solve.sh"
    generate_sh_script_dreal_stability_solve(str(out_path), exp_nums, dreal_precision)
    print(f"Generated wrapper script: {out_path}")