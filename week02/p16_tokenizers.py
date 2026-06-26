from transformers import AutoTokenizer

# 测试文本（包含常见词、复合词、专业术语）
text = "The quick brown fox jumps over the lazy dog. Unhappiness and disconnection in modern society."

# 不同分词器
tokenizers = [
    "bert-base-uncased",      # WordPiece
    "roberta-base",           # BPE
    "gpt2",                   # BPE
    "google/t5-small-lm-adapt" # SentencePiece (类似BPE)
]

print(f"原始文本: {text}\n")
for model_name in tokenizers:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(text)
    print(f"{model_name:30} -> {len(tokens):3d} tokens")
    # 可选：打印具体token
    print(f"Tokens: {tokens}")

# 1024 tokens, 4080TI 16GB 显存不够   
# GPT-2  只有Transformer decoder 架构 ， 只能利用上文，不能利用下文 自注意力机制

# L 序列  L*L  1024*1024*12(注意力的头， 因为只有decoder)*2(精度，如果用float16，那就是两个字节)byte = 24MB(真正占用的显存) * 12（gpt2的层数） = 2.88G * 4（batch数量） = 11.5G + 230MB + 920M（优化器） = 13.6G + 中间激活值230M（梯度） = 13.8G

# 16G 显存
# maxlength: 512  batch size: 8


# 22G Titan

# 128^2= 16384  1x
# 256^2=65536   4x
# 512^2=262144  16x
# 1024^2=1048576 64x

# 1、减小 max_length 512
# 2、使用更粗粒度的 tokenizer  BERT 
# 3、启用梯度检查点  gradient checkpointing=True
# 4、启用混合精度训练（AMP - Automatic Mixed Precision）
# 5、使用多卡

# OOM 内存    OOV 显存


#  W   4*4 = 16 
# delta_w = A * B 
# r=2 (真实 8)

# W = [
#     [1,2,3,4],
#     [5,6,7,8],
#     [9,10,11,12],
#     [13,14,15,16]
# ]

# A(4*2)  B(2*4)
# A = [
#     [0.1,0.2],
#     [0.3,0.4],
#     [0.5,0.6],
#     [0.7,0.8]
# ]

# B = [
#     [0.1,0.2,0.3,0.4],
#     [0.5,0.6,0.7,0.8]
# ]   


# delta_w = [
#     [0.011, 0.014, 0.017, 0.020 ],
#     [0.025, 0.030, 0.035, 0.040 ],
#     [0.039, 0.046, 0.053, 0.060 ],
#     [0.053, 0.062, 0.071, 0.080 ]
# ]

# w + delta_w 

# (4 * 2) + ( 2 * 4 )= 16

# 1024 * 1024 = 1048576

# r = 8  LORA  (1024*8) + ( 8 * 1024 ) = 16384  

# 16384/1048576 = 0.015625   1.56%