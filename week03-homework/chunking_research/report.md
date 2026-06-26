# Sentence Chunking & Retrieval — Experiment Report / 句子切片与检索实验报告

## 1. Experiment Setup / 实验设置

**EN.** The corpus consists of three English research papers, converted from PDF to Markdown and placed under `data/`:
`1-FinGPT-Finance.md` (FinGPT, an open-source financial LLM), `2-RT2-Robotics.md` (RT-2, a vision-language-action model for robotic control), and `3-MedPaLM-Medicine.md` (Med-PaLM, clinical knowledge in LLMs). Retrieval and generation are powered by OpenAI models through LlamaIndex: `gpt-5` as the LLM and `text-embedding-3-small` as the embedding model. For every splitter we build a `VectorStoreIndex`, retrieve the top-5 nodes, generate an answer, and judge it against a reference answer with an LLM-as-judge. The evaluation question targets the FinGPT paper: *"What approach does FinGPT take to build financial large language models, and why?"* and the keyword `data-centric` is used to check whether the retrieved context contains the answer.

**中.** 语料是三篇英文论文，已从 PDF 转换为 Markdown 放在 `data/` 下：`1-FinGPT-Finance.md`（FinGPT，开源金融大模型）、`2-RT2-Robotics.md`（RT-2，用于机器人控制的视觉-语言-动作模型）、`3-MedPaLM-Medicine.md`（Med-PaLM，大模型中的临床知识）。检索与生成通过 LlamaIndex 调用 OpenAI 模型：LLM 用 `gpt-5`，嵌入模型用 `text-embedding-3-small`。对每种切片器都构建 `VectorStoreIndex`、检索 top-5 节点、生成回答，并用"LLM 当评委"对照标准答案打分。评测问题针对 FinGPT 这篇：*"FinGPT 采用什么方法来构建金融大语言模型，为什么？"*，并用关键词 `data-centric` 判断检索到的上下文是否包含答案。

## 2. Results / 实验结果

**EN.** The table below is produced directly by `main.py`. "Contains answer" checks the `data-centric` keyword in the retrieved context; "Answer accurate" is the LLM judge's verdict on completeness vs. the reference answer; "Redundancy" is a manual 1–5 score (1 = least redundant, 5 = most redundant); "Nodes" is the number of chunks the splitter produced over all three documents.

**中.** 下表由 `main.py` 直接输出。"上下文包含答案"检查检索上下文里是否有 `data-centric` 关键词；"回答准确"是 LLM 评委对照标准答案在完整性上的判定；"冗余度"是人工 1–5 打分（1=最不冗余，5=最冗余）；"节点数"是该切片器在三篇文档上切出的块数。

| Splitter / 切片器 | Contains answer / 含答案 | Answer accurate / 回答准确 | Redundancy / 冗余度 (1–5) | Nodes / 节点数 |
|---|---|---|---|---|
| Sentence(512/50)    | Yes / 是 | Yes / 是 | 3 | 187 |
| Token(128/4)        | Yes / 是 | No / 否  | 1 | 990 |
| Sentence Window(3)  | Yes / 是 | Yes / 是 | 4 | 2221 |
| Sentence(256/20)    | No / 否  | Yes / 是 | 2 | 407 |
| Sentence(512/50)    | Yes / 是 | Yes / 是 | 3 | 187 |
| Sentence(1024/100)  | Yes / 是 | Yes / 是 | 5 | 89 |

**EN.** Two cross-cutting observations. First, the only configuration judged inaccurate is `Token(128/4)`. Its 128-token chunks repeatedly cut sentences mid-way (visible in the terminal, e.g. "We highlight the importance of an au-"), so even though retrieval lands in the right region, the fragmented context yields a less complete answer than the other splitters — a direct link between over-small chunks and answer quality. Second, `Sentence(256/20)` is the only run whose top-5 context did **not** contain the `data-centric` keyword, yet the model still produced a complete, accurate answer. This shows that the literal keyword-presence check and actual answer completeness do not always agree: the model can assemble a correct answer from other relevant chunks even when the exact phrase is fragmented out of the top-k window.

**中.** 有两个贯穿全局的现象。其一，唯一被判为"不准确"的是 `Token(128/4)`。它的 128-token 小块反复把句子拦腰截断（终端里能看到，如 "We highlight the importance of an au-"），所以即便检索命中了正确区域，破碎的上下文也让它的回答不如其他切片器完整——这把"块过小"和"回答质量"直接联系了起来。其二，`Sentence(256/20)` 是唯一一个 top-5 上下文里**没有** `data-centric` 关键词的配置，但模型依然给出了完整、准确的回答。这说明"逐字关键词命中"和"回答是否完整"并不总是一致：即便确切短语被切碎、挤出了 top-k 窗口，模型仍能从其他相关块拼出正确答案。

## 3. Which parameters significantly affect the results, and why? / 哪些参数显著影响效果？为什么？

**EN.** `chunk_size` is by far the most influential parameter. It controls the granularity of retrieval and directly trades precision against context completeness:
- Small chunks (`Token(128/4)`, `Sentence(256/20)`) produce many nodes (990 / 407) and low redundancy, but each chunk carries little context. They can fragment a complete idea across several nodes and, as `Sentence(256/20)` shows, even push the key phrase out of the top-5 results.
- Large chunks (`Sentence(1024/100)`) produce few nodes (89) and rich, self-contained context, but the redundancy score climbs to 5 because each retrieved chunk also drags in a lot of neighboring, partly-overlapping text.

The choice of **splitter type** is the second major factor. `SentenceSplitter` respects sentence boundaries and gives semantically coherent chunks; `TokenTextSplitter` cuts purely by token count and frequently truncates sentences mid-way (visible in the terminal output, e.g. "We highlight the importance of an au-"); `SentenceWindowNodeParser` indexes one sentence per node (hence 2221 nodes — very precise matching) but uses a post-processor to expand each match back into a multi-sentence window at generation time, which is why its retrieved context is more complete than its tiny index nodes would suggest. `chunk_overlap` is a smaller, secondary knob discussed in Section 4.

**中.** `chunk_size`（块大小）是影响最大的参数。它决定检索的粒度，直接在"精确度"和"上下文完整性"之间做权衡：
- 小块（`Token(128/4)`、`Sentence(256/20)`）切出的节点很多（990 / 407）、冗余度低，但每块携带的上下文很少。它们会把一个完整观点拆散到多个节点，正如 `Sentence(256/20)` 所示，甚至会把关键短语挤出 top-5 结果。
- 大块（`Sentence(1024/100)`）节点很少（89）、上下文丰富且自洽，但冗余度飙到 5，因为每个被检索到的块还会带进大量相邻、部分重叠的文本。

**切片器类型**是第二大影响因素。`SentenceSplitter` 尊重句子边界，块在语义上更连贯；`TokenTextSplitter` 纯按 token 数硬切，经常把句子拦腰截断（终端输出里能看到，如 "We highlight the importance of an au-"）；`SentenceWindowNodeParser` 每个节点只索引一个句子（所以多达 2221 个节点——匹配非常精确），但通过后处理器在生成阶段把命中的句子还原成"前后多句的窗口"，所以它检索到的上下文比其细小的索引节点看起来要完整得多。`chunk_overlap` 是较次要的旋钮，放在第 4 节讨论。

## 4. Pros and cons of too-large vs. too-small chunk_overlap / chunk_overlap 过大或过小的利弊

**EN.** `chunk_overlap` repeats a slice of text between adjacent chunks so that an idea split across a boundary still survives intact in at least one chunk.
- **Too small (e.g. Token's overlap of 4):** lowest storage and compute, and the lowest redundancy (the `Token(128/4)` run scored 1). The cost is the highest risk of context fracture — a sentence cut at a boundary may not be fully present in any single chunk, hurting recall and answer completeness.
- **Too large (e.g. `Sentence(1024/100)`):** strongly preserves continuity so almost nothing is lost at boundaries, but it inflates storage and indexing cost, and — most visibly in our experiment — drives up redundancy. With overlap 100 and top-5 retrieval, the returned passages repeat large amounts of identical text (redundancy score 5), and that duplicated material can crowd out other relevant information and distract the LLM.

A common rule of thumb is to set `chunk_overlap` to roughly 10–20% of `chunk_size` (e.g. 512 → ~50), which is the balanced middle point we used for the `Sentence(512/50)` baseline.

**中.** `chunk_overlap` 让相邻块之间重复一小段文本，这样即使一个观点被边界切开，至少在某一个块里仍然是完整的。
- **过小（如 Token 的 overlap=4）：** 存储和计算成本最低，冗余度也最低（`Token(128/4)` 打了 1 分）。代价是上下文断裂的风险最高——在边界被切断的句子可能在任何单个块里都不完整，从而损害召回和回答完整性。
- **过大（如 `Sentence(1024/100)`）：** 强力保持连续性，边界处几乎不丢信息，但会抬高存储和索引成本，而且在本实验里最直观的是——把冗余度推高了。在 overlap=100、检索 top-5 的情况下，返回的片段重复了大量相同文本（冗余度 5 分），这些重复内容会挤占其他相关信息、干扰 LLM 判断。

常见经验法则是把 `chunk_overlap` 设为 `chunk_size` 的约 10–20%（例如 512 → ~50），这正是我们 `Sentence(512/50)` 基线所用的折中点。

## 5. Balancing precise retrieval and context richness / 如何在"精确检索"与"上下文丰富性"之间权衡

**EN.** The experiment makes the tension concrete: small chunks retrieve precisely but starve the LLM of context (`Sentence(256/20)` even missed the keyword), while large chunks supply rich context but are redundant (`Sentence(1024/100)` scored 5). Several strategies help reconcile the two:
1. **Tune `chunk_size` empirically.** Use a loop like the one in `main.py` to sweep sizes on your own data and question type — favor smaller chunks for sharp factual questions and larger chunks for summary/synthesis questions. For these papers, the `Sentence(512/50)` baseline is the most balanced.
2. **Use the sentence-window strategy.** `SentenceWindowNodeParser` indexes single sentences for precise matching, then expands each hit into a surrounding window for generation — getting the precision of small chunks and the context of large ones at once.
3. **Two-stage / parent-child retrieval.** Retrieve many small chunks first, then look up and re-rank the larger "parent" passages they belong to, sending only the top few rich passages to the LLM.
4. **Fusion retrieval + re-ranking.** Run several strategies (small-chunk and large-chunk) in parallel and fuse/re-rank the results to combine their strengths.

Overall, tuning `chunk_size`/`chunk_overlap` alone rarely reaches the optimum; combining a sensible baseline (≈512/50) with an advanced strategy such as sentence-window or parent-child retrieval is the most reliable way to balance precise retrieval against context richness.

**中.** 本实验把这种张力具体化了：小块检索精确但让 LLM 缺乏上下文（`Sentence(256/20)` 甚至漏掉了关键词），而大块上下文丰富却冗余（`Sentence(1024/100)` 打了 5 分）。有几种策略可以调和两者：
1. **经验性地调 `chunk_size`。** 用 `main.py` 里那样的循环，在你自己的数据和问题类型上扫描不同块大小——事实性强的精确问题偏向小块，归纳/综述类问题偏向大块。对这三篇论文，`Sentence(512/50)` 基线最均衡。
2. **使用句子窗口策略。** `SentenceWindowNodeParser` 用单句索引做精确匹配，再在生成时把命中句扩展成"前后窗口"——同时拿到小块的精确性和大块的上下文。
3. **两阶段 / 父子检索。** 先检索大量小块，再回溯并重排它们所属的更大"父块"，只把最相关的若干富上下文段落送给 LLM。
4. **融合检索 + 重排。** 并行跑多种策略（小块的和大块的），再对结果做融合/重排，取长补短。

总体而言，仅调 `chunk_size`/`chunk_overlap` 往往难以达到最优；把一个合理的基线（≈512/50）与句子窗口或父子检索这类进阶策略结合，才是在"精确检索"与"上下文丰富性"之间取得平衡的最可靠做法。
