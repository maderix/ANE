// stories110m.h — Stories110M (Llama2-style, 12 layers, MHA)
#pragma once

#define MODEL_NAME "Stories110M"

#define DIM 768
#define HIDDEN 2048
#define HEADS 12
#define KV_HEADS 12
#define HD (DIM/HEADS)       // = 64
#define GQA_RATIO 1          // MHA: no GQA
#define Q_DIM (HEADS * HD)   // = 768 = DIM
#define KV_DIM (KV_HEADS * HD) // = 768 = DIM
#define SEQ 256
#define NLAYERS 12
#define VOCAB 32000

#define CKPT_PATH "ane_stories110M_dyn_ckpt.bin"
#define DEFAULT_DATA_PATH "../tinystories_data00.bin"
