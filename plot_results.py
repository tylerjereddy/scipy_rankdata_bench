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


df = pd.DataFrame({"NumPy i9": numpy_i9,
                   "CuPy 1080": cupy_1080,
                   "CuPy 4090": cupy_4090,
                   "JAX i9": jax_i9,
                   "JAX 1080": jax_1080,
                   "JAX 4090": jax_4090})
sns.barplot(df, ax=ax)
ax.set_ylabel("Elapsed Time (s)")
fig.savefig("bench_rankdata.png", dpi=300)
