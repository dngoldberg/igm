#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, sys, shutil
import time
import tensorflow as tf
import matplotlib.pyplot as plt


def params_print_all_comp_info(parser):
    pass

def initialize_print_all_comp_info(params, state):
    pass

def update_print_all_comp_info(params, state):
    pass

def finalize_print_all_comp_info(params, state):
 
    modules = [A for A in state.__dict__.keys() if 'tcomp_' in A]

    state.tcomp_all = [ np.sum([np.sum(getattr(state,m)) for m in modules]) ]

    print("Computational statistics report:")
    with open(
        os.path.join(params.working_dir, "computational-statistics.txt"), "w"
    ) as f:
        for m in modules:
            CELA = (
                m[6:],
                np.mean(getattr(state,m)),
                np.sum(getattr(state,m)),
                len(getattr(state,m)),
            )
            print(
                "     %24s  |  mean time per it : %8.4f  |  total : %8.4f  |  number it : %8.0f"
                % CELA,
                file=f,
            )
            print(
                "     %24s  |  mean time per it : %8.4f  |  total : %8.4f  |  number it  : %8.0f"
                % CELA
            )

    os.system(
        "echo rm "
        + os.path.join(params.working_dir, "computational-statistics.txt")
        + " >> clean.sh"
    )

    _plot_computational_pie(params, state)

def _plot_computational_pie(params, state):
    """
    Plot to the computational time of each model components in a pie
    """

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return "{:.0f}".format(val)

        return my_autopct

    total = []
    name = []

    modules = [A for A in state.__dict__.keys() if 'tcomp_' in A]
    modules.remove('tcomp_all')

    for m in modules:
        total.append(np.sum(getattr(state,m)[1:]))
        name.append(m[6:])

    sumallindiv = np.sum(total)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"), dpi=200)
    wedges, texts, autotexts = ax.pie(
        total, autopct=make_autopct(total), textprops=dict(color="w")
    )
    ax.legend(
        wedges,
        name,
        title="Model components",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )
    plt.setp(autotexts, size=8, weight="bold")
    #    ax.set_title("Matplotlib bakery: A pie")
    plt.tight_layout()
    plt.savefig(os.path.join(params.working_dir, "computational-pie.png"), pad_inches=0)
    plt.close("all")

    os.system(
        "echo rm "
        + os.path.join(params.working_dir, "computational-pie.png")
        + " >> clean.sh"
    )