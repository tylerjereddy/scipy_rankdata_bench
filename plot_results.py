import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


fig, ax = plt.subplots()

# NumPy 2.5.0 results on gp160 (i9-13900K)
# (incantation: `python bench.py --size 100000000 --backend numpy`)
# 4 trials
numpy_i9 = [5.148675503209233,
            5.151560884900391,
            5.147006139159203,
            5.204813475720584,
           ]
# CuPy 14.1.1 results on gp160 (4090)
# (incantation: `CUDA_VISIBLE_DEVICES=0 python bench.py --size 100000000 --backend cupy`)
# 4 trials
cupy_4090 = [0.3554808758199215,
             0.348264348693192,
             0.3581071700900793,
             0.35619060043245554,
            ]
# CuPy 14.1.1 results on chicoma (A100)
# (incantation: `CUDA_VISIBLE_DEVICES=0 python bench.py --size 100000000 --backend cupy`)
# 4 trials
cupy_a100 = [0.6823827699990943,
             0.7852864799788222,
             0.6816078000701964,
             0.6527298010187224,
            ]
# CuPy 14.1.1 results on selene (H100)
# (incantation: `CUDA_VISIBLE_DEVICES=0 python bench.py --size 100000000 --backend cupy`)
# 4 trials
cupy_h100 = [0.5819971700002498,
             0.6188228520004486,
             0.6098838119996799,
             0.5712698169991199,
            ]
# CuPy 14.1.1 results on gp160 (1080 Ti)
# (incantation: `CUDA_VISIBLE_DEVICES=1 python bench.py --size 100000000 --backend cupy`)
# 4 trials
cupy_1080 = [0.41221918258816004,
             0.41198909748345613,
             0.41145238373428583,
             0.4129036944359541,
            ]

# JAX 0.10.2 results on gp160 (i9-13900K)
# (incantation: `JAX_PLATFORMS="cpu" python bench.py --size 100000000 --backend jax`)
# 4 trials
jax_i9 = [0.49015018064528704,
          0.5264123771339655,
          0.5021754037588835,
          0.5257084434852004,
          ]

# JAX 0.10.2 results on gp160 (1080)
# (incantation: `CUDA_VISIBLE_DEVICES=1 python bench.py --size 100000000 --backend jax`)
# 4 trials
jax_1080 = [0.911471713334322,
            0.9187971036881208,
            0.8951585507020354,
            0.8859530910849571,
           ]

# JAX 0.10.2 results on gp160 (4090)
# (incantation: `CUDA_VISIBLE_DEVICES=0 python bench.py --size 100000000 --backend jax`)
# 4 trials
jax_4090 = [0.9070844165980816,
            0.8868432966992259,
            0.8999174265190959,
            0.9091755729168653,
           ]
# JAX 0.10.2 results on chicoma (A100)
# (incantation: `CUDA_VISIBLE_DEVICES=0 python bench.py --size 100000000 --backend jax`)
# 4 trials
jax_a100 = [1.8963053349871188,
            1.8582751370267943,
            1.9545096419751644,
            1.7758357039419934,
           ]
# JAX 0.10.2 results on selene (H100)
# (incantation: `CUDA_VISIBLE_DEVICES=0 python bench.py --size 100000000 --backend jax`)
# 4 trials
jax_h100 = [2.073358814999665,
            2.1594462329994712,
            2.0476015860003827,
            2.082513151999592,
           ]

# torch 2.12.1 results on gp160 (i9-13900K)
# (incantation: `CUDA_VISIBLE_DEVICES="" python bench.py --size 100000000 --backend torch`)
# 4 trials
torch_i9 = [3.7354863798245788,
            3.7818091763183475,
            3.75791558809578,
            3.7367754969745874,
           ]

# NOTE: torch 2.10.0 deps not compatible with older 1080
# torch 2.10.0 results on gp160 (4090)
# (incantation: `CUDA_VISIBLE_DEVICES=0 python bench.py --size 100000000 --backend torch`)
# 4 trials
torch_4090 = [0.32906035985797644,
              0.3287751730531454,
              0.32863392401486635,
              0.3285703118890524,
             ]
# torch 2.10.0 results on chicoma (A100)
# (incantation: `CUDA_VISIBLE_DEVICES=0 python bench.py --size 100000000 --backend torch`)
# 4 trials
torch_a100 = [0.6596070200903341,
              0.7665192249696702,
              0.6573314070701599,
              0.7447423950070515,
             ]
# torch 2.10.0 results on selene (H100)
# (incantation: `CUDA_VISIBLE_DEVICES=0 python bench.py --size 100000000 --backend torch`)
# 4 trials
torch_h100 = [0.5862254210005631,
              0.5603503700003785,
              0.575681786000132,
              0.5579765259990381,
             ]
# torch 2.14.0.dev20260622+rocm7.2 results on rzadams (MI300A)
# (incantation: `HIP_VISIBLE_DEVICES=0 python bench.py --size 100000000 --backend torch`)
# 4 trials
torch_mi300a = [2.484767579007894,
                2.4801498299930245,
                2.4980579330585897,
                2.5620238608680665,
             ]

# torch 2.12.1 results on ARM Mac (M3 MAX CPU)
# (incantation: `CPU_ONLY=1 python bench.py --size 100000000 --backend torch`)
# 4 trials
torch_m3_max_cpu = [5.127518416000385,
                    5.205726166999739,
                    5.184289208000337,
                    5.13092908300132,
                   ]

# torch 2.12.1 results on ARM Mac (M3 MAX GPU)
# (incantation: `python bench.py --size 100000000 --backend torch`)
# 4 trials
torch_m3_max_gpu = [13.36404520799988,
                    13.451699332999851,
                    13.381041290998837,
                    13.46170579200043,
                   ]


df = pd.DataFrame({"NumPy i9": numpy_i9,
                   "torch M3 Max CPU": torch_m3_max_cpu,
                   "torch i9": torch_i9,
                   "JAX i9": jax_i9,
                   "torch M3 Max GPU": torch_m3_max_gpu,
                   "torch MI300A": torch_mi300a,
                   "JAX H100": jax_h100,
                   "JAX A100": jax_a100,
                   "JAX 1080": jax_1080,
                   "JAX 4090": jax_4090,
                   "torch A100": torch_a100,
                   "CuPy A100": cupy_a100,
                   "CuPy H100": cupy_h100,
                   "torch H100": torch_h100,
                   "CuPy 1080": cupy_1080,
                   "CuPy 4090": cupy_4090,
                   "torch 4090": torch_4090})
sns.barplot(df, ax=ax)
ax.axhline(y=df["NumPy i9"].mean(),
           xmin=0,
           xmax=1,
           color="pink",
           ls="--")
ax.set_ylabel("Elapsed Time (s)")
ax.tick_params(axis='x', labelrotation=90)
fig.tight_layout()
fig.savefig("bench_rankdata.png", dpi=300)
