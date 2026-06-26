# FinGPT: Open-Source Financial Large Language Models

FinGPT: Open-Source Financial Large Language Models
Hongyang Yang1, Xiao-Yang Liu2, Christina Dan Wang3∗
1AI4Finance Foundation†;
2Columbia University;
3New York University Shanghai
contact@ai4finance.org
Abstract
Large language models (LLMs) have shown the
potential of revolutionizing natural language pro-
cessing in diverse domains, sparking great interest
in finance. However, the finance domain presents
unique challenges, including high temporal sen-
sitivity, constant dynamism, and a low signal-
to-noise ratio (SNR). While proprietary models
like BloombergGPT have taken advantage of their
unique data accumulation, such privileged access
calls for an open-source alternative to democratize
internet-scale financial data.
In this paper, we present an open-source large
language model, FinGPT, for the finance sec-
tor. Unlike proprietary models, FinGPT takes a
data-centric approach, providing researchers and
practitioners with accessible and transparent re-
sources to customize their financial LLMs (Fin-
LLMs). We highlight the importance of an au-
tomatic data curation pipeline and the lightweight
low-rank adaptation technique in building Fin-
GPT. Furthermore, we provide fundamental tasks
as building blocks for benchmarking and show-
case potential applications as stepping stones for
users, such as robo-advising and sentiment anal-
ysis. Through collaborative efforts within the
open-source AI4Finance community, FinGPT aims
to stimulate innovation, democratize FinLLMs,
and unlock new opportunities in open finance.
Two associated code repos are https://github.
com/AI4Finance-Foundation/FinGPT and https://
github.com/AI4Finance-Foundation/FinNLP
1 Introduction
The continual expansion and evolution of artificial intel-
ligence have provided a fertile ground for the prolifera-
tion of LLMs [Vaswaniet al., 2017; Radfordet al., 2018;
Devlinet al., 2018; Ethayarajh, 2019; Lewiset al., 2019;
Lewiset al., 2020; Brownet al., 2020; Thoppilanet al.,
2022], thereby effecting a transformative shift across diverse
∗Corresponding author.
†AI4Finance Foundation: ai4finance.org
domains. This sweeping change has engendered keen interest
in the potential applications of financial LLMs (FinLLMs). It
is, however, evident that the acquisition of high-quality, rel-
evant, and up-to-date data stands as a critical factor in the
development of efficacious and efficient FinLLMs.
Utilizing LLMs in the finance sector reveals intricate hur-
dles. Firstly, there’s the issue of high temporal sensitivity.
Financial data are characterized by their time-sensitive na-
ture. Market-moving news or updates, once released, pro-
vide a narrow window of opportunity for investors to maxi-
mize their alpha (the measure of an investment’s relative re-
turn). Secondly, the financial landscape is marked by high
dynamism. It is in a constant state of flux due to the ceaseless
flow of news, social media updates, and other market-related
information. Given these constant changes, retraining LLMs
frequently is not only expensive but also impractical. Lastly,
financial data is often characterized by a low signal-to-noise
ratio (SNR) [Yanget al., 2020 ]. The useful information is
often hidden amongst a significant amount of irrelevant or
noisy data. Extracting valuable insights from this sea of in-
formation necessitates advanced techniques.
In the proprietary sphere, models like BloombergGPT[Wu
et al., 2023] have capitalized on their exclusive access to spe-
cialized data to train a FinLLM. However, the restricted ac-
cessibility and transparency of their data collections and train-
ing protocols have accentuated the demand for an open and
inclusive alternative. In response to this demand, we are wit-
nessing a shifting trend towards democratizing internet-scale
financial data in the open finance domain.
In this paper, we address these aforementioned challenges
associated with financial data and introduce FinGPT, an end-
to-end open-source framework for financial large language
models (FinLLMs). Adopting a data-centric approach, Fin-
GPT underscores the crucial role of data acquisition, clean-
ing, and preprocessing in developing open-source FinLLMs.
By championing data accessibility, FinGPT aspires to en-
hance research, collaboration, and innovation in finance,
paving the way for open finance practices.
Our contributions are summarized as follows:
•Data-centric approach: Recognizing the significance of
data curation, FinGPT adopts a data-centric approach and
implements rigorous cleaning and preprocessing methods
for handling varied data formats and types.
International Symposium on Large Language Models for Financial Services (FinLLM 2023)@IJCAI 2023 —
https://finllm.github.io/workshop
arXiv:2306.06031v2  [q-fin.ST]  15 Nov 2025
•End-to-end framework: FinGPT embraces a full-stack
framework with five layers:
–Data source layer: Assures comprehensive market
coverage, addressing the temporal sensitivity of finan-
cial data through real-time information capture.
–Data engineering layer: Primed for real-time NLP
data processing, this layer tackles the inherent chal-
lenges of high temporal sensitivity and low signal-to-
noise ratio in financial data.
–LLMs layer: Focusing on a range of fine-tuning
methodologies, this layer takes care of the highly dy-
namic nature of financial data, ensuring the model’s
relevance and accuracy.
–Tasks Layer: This layer is responsible for execut-
ing fundamental tasks. These tasks serve as the
benchmarks for performance evaluations and cross-
comparisons in the realm of FinLLMs.
–Applications layer: Showcasing practical applications
and demos, this layer highlights the potential capabil-
ity of FinGPT in the finance sector.
•Democratization: FinGPT, as an open-source framework,
aims to democratize financial data and FinLLMs, uncover-
ing untapped potentials in open finance. We envision Fin-
GPT as a catalyst for stimulating innovation within the fi-
nance domain. FinGPT is not limited to providing techni-
cal contributions, but also cultivates an open-source ecosys-
tem for FinLLMs, promoting real-time processing and cus-
tomized adaptation for users. By nurturing a robust col-
laboration ecosystem within the open-source AI4Finance
community, FinGPT is positioned to refine our understand-
ing and application of FinLLMs.
2 Related Work
2.1 The Raise of FinLLMs
Large Language Models (LLMs) have been recognized as a
technological breakthrough in NLP, such as GPT-3 and GPT-
4 [Brownet al., 2020; Jianget al., 2023; OpenAI, 2023;
Teamet al., 2023; Liuet al., 2024 ]. They take transformer-
based architectures, demonstrating impressive performance
across various text-generation tasks. As an offshoot of the
GPT family developed by OpenAI, ChatGPT was designed
to produce human-like texts based on input prompts. It has
shown significant utility in diverse applications, from draft-
ing emails to writing code and even in creating art content.
LLMs have been applied to various tasks within the finan-
cial sector [Dredzeet al., 2016; Araci, 2019; Baoet al., 2021;
DeLuciaet al., 2022 ], from predictive modeling to generating
insightful narratives from raw financial data. Recent literature
has focused on using these models for financial text analysis,
given the abundance of text data in this field, such as news
articles, earnings call transcripts, and social media posts.
The first example of FinLLMs is BloombergGPT [Wuet
al., 2023], which was trained on a mixed dataset of financial
and general data sources. Despite its impressive capabilities,
access limitations exist, and the prohibitive training cost has
motivated the need for low-cost domain adaptation.
Our FinGPT responds to the aforementioned hurdles, pre-
senting an open-source FinLLM. It employs Reinforcement
Learning from Human Feedback (RLHF) to understand and
adapt to individual preferences, paving the way for person-
alized financial assistants. We aim to combine the strengths
of general LLMs like ChatGPT with financial adaptation, ex-
ploiting LLM’s capability in open finance.
2.2 Why Open-Source FinLLMs?
AI4Finance Foundation1 is a non-profit, open-source organi-
zation that integrates Artificial Intelligence (AI) and financial
applications. With a proven track record of nurturing an in-
novative ecosystem of FinTech tools, such as FinRL [Yang
et al., 2020 ] and FinRobot [Yanget al., 2024 ], the founda-
tion is poised to accelerate the evolution of FinLLMs. Stead-
fast commitment and cutting-edge contributions may pave the
way for AI’s transformative applications in open finance.
• Advancing equal opportunities via democratizing Fin-
LLMs: Adopting an open-source methodology promotes
universal access to state-of-the-art technology, adhering to
the ethos of democratizing FinLLMs.
• Cultivating transparency and trust: Open-source FinLLMs
offer a comprehensive overview of their foundational code-
base, bolstering transparency and trust.
• Accelerating research and innovation: The open-source
model fuels progress in research and development within
the AI domain. It allows researchers to leverage existing
models, thus nurturing a faster progression of innovation
and scientific discovery.
• Enhancing education: Open-source FinLLMs serve as
robust educational tools, presenting students with the
prospect of exploring the complexities of FinLLMs through
direct engagement with fully operational models.
• Upgrade foundation infrastructure for financial text data by
community collaboration: This collaborative participation
bolsters the model’s long-term durability and effectiveness.
3 Overview of FinGPT: An Open-Source
Framework for FinLLMs
FinGPT represents an innovative open-source framework de-
signed specifically for FinLLMs. As delineated in Fig. 1,
FinGPT consists of four components: Data Source, Data En-
gineering, LLMs, and Applications. Each plays a crucial role
in maintaining the functionality and adaptability of FinGPT.
•Data source layer: The starting point is the Data Source
Layer, which orchestrates the acquisition of extensive fi-
nancial data from a wide array of online sources. This layer
ensures comprehensive market coverage by integrating data
from news websites, social media platforms, financial state-
ments, market trends, and more. The goal is to capture the
1https://ai4finance.org. The AI4Finance Foundation is a U.S.-
registered 501(c)(3) nonprofit public charity focused on promoting
open scientific research in financial AI, building open-source infras-
tructure, and supporting a global community of researchers through
shared datasets, benchmarks, and educational programs.
2
Trainable Models
Applications
Robo-
advisor
LLMs
LLM APIs 
Llama3
Claude
ChatGPT
LLaMA
Kimi
Grok
Gemini
Data
Engineering
 Tokenization
Data Cleaning
 Vector Embedding
 Feature Extraction
Data Source
Vector Database
(Storage)
Datasets
 AShare
A
W
S
Other
Systems
stocknet-
dataset
 ......
Data Integration
Cloud Native
 On-Premises
 Graphics Processing Unit Server
Cloud
News
 Finnhub
 Yahoo
Finance
  CNBC
 .......
Social
Media
  Twitter
   Weibo
  Reddit
 .......
Filings
 SEC
   NYSE
 NASDAQ
 .......
Trends
 Google Trends
 Seeking Alpha
.......
Mistral
Fine-tuning Methods
Low-
rank Adaptation (LoRA), 
QLoRA
Reinforcement Learning on 
Stock Prices (RLSP)
Quantitative
Trading
Portfolio
Optimization
Financial Sentiment
Analysis
Risk 
Management
Financial Fraud
Detection
Credit 
Scoring
ESG
Scoring
Financial 
Education
M&A 
Forecasting
Other
Applications
Low-Code
Development
FinGPT
...
...
G
C
P
A
Z
U
R
E
I
B
M
Tasks
 Terminology
Understanding
Information Extraction
 Sentiment Analysis
Intent Detection
Named-entity
Recognition (NER)
Numerical Reasoning
Summarization
Data Analysis
DeepSeek
Falcon
Qwen3
Gemma
FinNLP
Real-time data pipeline
APIs
Streaming Data
Data Augmentation
Prompt Construction
Chain-of-Thought
Retrieval-
Augmented 
Generation(RAG)
Figure 1: Overall framework of FinGPT.
nuance of the market, thereby addressing the inherent tem-
poral sensitivity.
•Data engineering layer: This layer focuses on the real-
time processing of text data to tackle the challenges ofhigh
temporal sensitivityandlow signal-to-noise ratioinherent
in financial data. It incorporates state-of-the-art NLP tech-
niques to filter noise and highlight the most salient pieces
of information.
•LLMs layer: Lying at the heart, it encompasses various
fine-tuning methodologies, prioritizing lightweight adapta-
tion, to keep the model updated and pertinent. By maintain-
ing an updated model, FinGPT can take care of the highly
dynamic nature of financial data, ensuring its responses are
in sync with the current financial climate.
•Tasks layer: The tasks layer is designed to provide build-
ing blocks. This layer serves a dual purpose: first, it exe-
cutes a variety of fundamental tasks that are crucial in the
FinLLMs landscape, such as sentiment analysis, content
summarization, and numerical reasoning. Second, it estab-
lishes a standardized set of metrics and attributes. These
standardized elements act not only as indicators but also as
benchmarks, facilitating both performance evaluation and
comparative analysis in the domain of FinLLMs.
•Application layer: The final component of FinGPT is
the Applications Layer, designed to demonstrate the prac-
tical applicability of FinGPT. It offers hands-on tutorials
and demo applications for financial tasks, including robo-
advisory services and sentiment analysis. These practical
demos not only serve as a guide to potential users but also
underscore the transformative potential of FinLLMs.
3.1 Data Sources
The first stage of FinGPT involves the collection of extensive
financial data from a wide array of online sources. These
include, but are not limited to:
•Financial news:Websites such as Reuters, CNBC, Yahoo
Finance, among others, are rich sources of financial news
and market updates. These sites provide valuable informa-
tion on market trends, company earnings, macroeconomic
indicators, and other financial events.
•Social media: Platforms such as Twitter, Facebook, Red-
dit, Weibo, and others, offer a wealth of information in
terms of public sentiment, trending topics, and immediate
reactions to financial news and events.
•Filings: Websites of financial regulatory authorities, such
as the SEC in the United States, offer access to company
filings. These filings include annual reports, quarterly earn-
ings, insider trading reports, and other important company-
specific information. Official websites of stock exchanges
(NYSE, NASDAQ, Shanghai Stock Exchange, etc.) pro-
vide crucial data on stock prices, trading volumes, company
listings, historical data, and other related information.
•Trends: Websites like Seeking Alpha, Google Trends, and
other finance blogs and forums provide access to analysts’
opinions, market predictions, the movement of specific se-
curities or market segments and investment advice.
3
•Academic datasets: Research-based datasets that offer cu-
rated and verified information for financial analysis.
To harness the wealth of information from these diverse
sources, FinGPT incorporates data acquisition tools capable
of scraping structured and unstructured data, including APIs,
web scraping tools, and direct database access where avail-
able. Moreover, the system is designed to respect the terms
of service of these platforms, ensuring data collection is ethi-
cal and legal.
Data APIs: In the FinGPT framework, APIs are used not
only for initial data collection but also for real-time data up-
dates, ensuring the model is trained on the most current data.
Additionally, error handling and rate-limiting strategies are
implemented to respect API usage limits and avoid disrup-
tions in the data flow.
3.2 Real-Time Data Curation Pipeline for
Financial NLP
Financial markets operate in real-time and are highly sensi-
tive to news and sentiment. Prices of securities can change
rapidly in response to new information, and delays in pro-
cessing that information can result in missed opportunities or
increased risk. As a result, real-time processing is essential in
financial NLP.
The primary challenge with a real-time NLP pipeline is
managing and processing the continuous inflow of data ef-
ficiently. The first step in the pipeline is to set up a system to
ingest data in real-time. This data could be streaming from
our data source APIs. Below are the steps to design a real-
time NLP pipeline for data ingestion.
Data cleaning: Real-time data can be noisy and inconsis-
tent. Therefore, real-time data cleaning involves removing
irrelevant data, handling missing values, text normalization
(like lowercasing), and error corrections.
Tokenization: In real-time applications, tokenization has
to be performed on the fly. This involves breaking down the
stream of text into smaller units or tokens.
Vector embedding: FinGPT encodes curated financial text
into dense semantic vectors using domain-adapted embed-
ding models. The embedding process incorporates entity-
aware representations (tickers, ratios, events) and temporal
metadata, allowing the system to capture fine-grained fi-
nancial meaning. All embeddings are indexed in a vector
database for low-latency retrieval, supporting RAG, event
clustering, and market-aligned RLSP training.
Feature extraction: Feature extraction involves trans-
forming raw data into an input that can be understood by ML
models. In real-time systems, this often needs to be a fast and
efficient process. Techniques such as TF-IDF, Bag of Words,
or embedding vectors like Word2Vec can be used.
Data augmentation: In the dynamic landscape of financial
markets, enhancing the variety and volume of training data is
crucial for building robust NLP models. Data augmentation
strategies will be employed to generate synthetic data that can
mimic the characteristics of actual financial data.
3.3 Large Language Models (LLMs)
Once the data has been properly prepared, it is used with
LLMs to generate insightful financial analyses. The LLM
layer includes:
•LLM APIs: Established LLM APIs offer foundational lan-
guage capabilities that serve as the base for further model
development and customization.
•Trainable models: Users can fine-tune FinGPT’s trainable
models on private data for personalized financial applica-
tions, ensuring relevance and accuracy in specific use cases.
•Fine-tuning methods: FinGPT supports various fine-
tuning methodologies, facilitating its adaptation into per-
sonalized robo-advisors efficiently and effectively.
•Prompt Engineering: Prompt Engineering is crucial for
optimizing input queries to LLMs, enhancing the extraction
of accurate financial information. This iterative process re-
quires careful crafting of prompts for nuanced responses,
necessitating a deep understanding of both finance and lan-
guage model characteristics.
Why lightweight fine-tuning LLMs for finance?
Fine-tuning or Instruction tuning of pre-existing LLMs for
finance, as described in [Ouyanget al., 2022 ], presents a
cost-efficient and time-saving alternative to the expensive and
lengthy process of retraining models from scratch.
BloombergGPT [Wuet al., 2023], though remarkable in its
finance-specific capabilities, comes with an intensive com-
putational requirement. It used approximately 1.3 million
GPU hours for training, which, when calculated using AWS
cloud’s $2.3 rate, translates to a staggering cost of around $3
million per training. In contrast to the high computational
cost of models like BloombergGPT [Wuet al., 2023 ], Fin-
GPT presents a more accessible solution by focusing on the
lightweight adaptation of top open-source LLMs. The cost
of adaptation falls significantly, estimated at around $300 per
fine-tuning.
This approach ensures timely updates and adaptability, es-
sential in the dynamic financial domain. Being open-source,
FinGPT not only promotes transparency but also allows
user customization, catering to the rising trend of personal-
ized financial advisory services. Ultimately, FinGPT’s cost-
effective, flexible framework holds the potential to democra-
tize financial language modeling and foster user-focused fi-
nancial services.
Fine-tuning via Low-rank Adaptation (LoRA)
In FinGPT, we fine-tune a pre-trained LLM utilizing a finan-
cial dataset. It’s well recognized that high-quality labeled
data is a pivotal determinant for many successful LLMs, in-
cluding ChatGPT. However, acquiring such top-notch labeled
data often proves costly in terms of time and resources and
generally requires the expertise of finance professionals.
When the application of LLMs is envisioned for the
scrutiny of financial texts and facilitation of quantitative trad-
ing strategies, it is imperative to contemplate the utilization
of the intrinsic labeling mechanisms available within the fi-
nancial marketplace. In light of this, FinGPT adopts the per-
centage of relative stock price changes corresponding to in-
dividual news articles as output labels. By assigning prede-
termined thresholds, these continuous labels are categorized
into three discrete sentiment classes: positive, negative, and
neutral.
4
Simultaneously, during the prompt engineering phase, the
model is meticulously instructed to elect one among the three
sentiment classes as its output. This meticulous approach en-
sures that the information gleaned during pre-training is max-
imally exploited, fostering the generation of insightful and re-
liable predictions on financial sentiment. The implementation
of Low-Rank Adaptation (LoRA) for LLMs [Huet al., 2021;
Dettmerset al., 2023 ], along with its quantized variant,
QLoRA [Dettmerset al., 2023 ], significantly streamlines the
model by reducing the count of trainable parameters from an
overwhelming 6.17 billion to a manageable 3.67 million.
Fine-tuning via Reinforcement Learning on Stock Prices
(RLSP)
Similarly, we can substitute Reinforcement Learning on
Stock Prices (RLSP) for Reinforcement Learning on Human
feedback, as utilized by ChatGPT. The reasoning behind this
substitution is that stock prices offer a quantifiable, objective
metric that reflects market sentiment in response to news and
events. This makes it a robust, real-time feedback mechanism
for training our model.
Reinforcement Learning (RL) allows the model to learn
through interaction with the environment and receiving feed-
back. In the case of RLSP, the environment is the stock
market, and the feedback comes in the form of stock price
changes. This approach permits FinGPT to refine its under-
standing and interpretation of financial texts, improving its
ability to predict market responses to various financial events.
By associating news sentiment with the subsequent perfor-
mance of the related stocks, RLSP provides an effective way
to fine-tune FinGPT. In essence, RLSP allows the model to
infer the market’s response to different news events and ad-
just its understanding and predictions accordingly.
Therefore, the integration of RLSP into the fine-tuning pro-
cess of FinGPT provides a powerful tool for improving the
model’s financial market understanding and predictive accu-
racy. By using actual stock price movements as feedback, we
are directly harnessing the wisdom of the market to make our
model more effective.
Retrieval Augmented Generation (RAG)
Retrieval-augmented generation (RAG) is a pivotal technique
incorporated within FinGPT [Zhanget al., 2023 ], as it seam-
lessly amalgamates the prowess of both context retrieval
mechanisms and Large Language Models (LLMs) to opti-
mize language generation tasks. This meticulous process en-
sures that the LLMs are not generating content in a vacuum
but are rather informed and nuanced in their output, draw-
ing from a rich tapestry of context provided by the retrieved
documents. These documents, working in tandem with the
input prompt, steer the LLMs effectively towards crafting re-
sponses that are not only accurate but deeply ingrained in the
relevant context, thereby increasing the utility and reliability
of the generated text.
3.4 Fundamental Tasks
FinGPT serves as a versatile tool in the financial sector, pro-
viding valuable assistance to both professionals and individ-
uals by effectively filtering and analyzing information. The
model excels in the following fundamental tasks:
•Summarization: FinGPT can efficiently condense lengthy
financial documents into concise summaries, preserving the
crucial information and insights. This function is invalu-
able for quickly understanding the essence of comprehen-
sive reports, news articles, or financial statements without
going through the entire content.
•Named-entity recognition (NER): The model is adept at
identifying and classifying named entities within the text,
such as company names, stock tickers, monetary values,
and percentages. This ability is crucial for extracting spe-
cific data points from unstructured text, facilitating more
structured and informed analysis.
•Information extraction: FinGPT can meticulously ex-
tract relevant information from various sources, providing
users with valuable insights. This capability is crucial for
decision-making processes, as it sifts through the noise to
highlight essential data and trends.
•Sentiment analysis: Sentiment Analysis is pivotal as a fun-
damental task due to its dual application in both identify-
ing market sentiment, namely financial sentiment analysis
and being utilized within robo-advisory platforms to dis-
cern client emotions during product recommendations.
•Data analysis: FinGPT can process and analyze vast
datasets, identifying patterns, anomalies, and significant
changes in the data. This feature supports data-driven
decision-making, offering a clearer understanding of mar-
ket dynamics and financial performance.
•Numerical reasoning: The model can perform calcula-
tions and numerical analysis based on the data provided in
the text, supporting users in evaluating financial metrics,
making projections, and assessing risks effectively.
•Terminology understanding: FinGPT is proficient in un-
derstanding and interpreting complex financial terminology
and jargon, making it a valuable assistant for both seasoned
professionals and individuals new to the financial sector.
•Intent detection: The model can accurately identify the
user’s intent behind a query, facilitating more effective and
relevant responses. This feature is particularly useful in de-
veloping intuitive and user-friendly financial advisory ap-
plications and services.
Various open-source datasets serve as benchmarks, effec-
tively engaging in a multitude of fundamental tasks. Ex-
amples include BloombergGPT[Wuet al., 2023 ], which uti-
lizes a curated selection of financial datasets derived from
the FLUE benchmark[Shahet al., 2022 ]. These datasets are
employed for a spectrum of essential tasks such as Senti-
ment Analysis and NER. Other noteworthy datasets include
FinRED [Sharmaet al., 2022 ], instrumental for information
extraction tasks, FINQA [Chenet al., 2021 ] for numerical
reasoning assessments, and FinRAD [Ghoshet al., 2021 ],
which is crucial for understanding and identifying financial
terms.
3.5 Potential Applications
FinGPT may find wide applications in financial services, aid-
ing professionals and individuals as a powerful information
filter. The potential applications include:
5
•Financial sentiment analysis: Evaluating sentiments
across different financial platforms for insightful invest-
ment guidance.
•Robo-advisor: The Robo-advisor function within Fin-
LLMs plays a pivotal role in providing personalized finan-
cial advice, minimizing the necessity for continual human
consultations.
•Quantitative trading: Producing trading signals for in-
formed trading decisions.
•Portfolio optimization: Utilizing numerous economic in-
dicators and investor profiles for optimal investment port-
folio construction.
•Credit scoring: Predicting creditworthiness from financial
data to aid lending decisions.
•Mergers and acquisitions (M&A) forecasting: Predict-
ing potential M&A activities by analyzing financial data
and company profiles, helping investors anticipate market
movements.
•ESG (Environmental, Social, Governance) scoring:
Evaluating companies’ ESG scores by analyzing public re-
ports and news articles.
•Risk management:Formulating effective risk strategies
by analyzing various risk factors.
•Fraud detection: Identifying potential fraudulent transac-
tion patterns for enhanced financial security.
•Automating KYC Processes:FinGPT can streamline
KYC procedures by analyzing documents for identity val-
idation, cross-checking information against databases, and
detecting inconsistencies. It can also interpret complex le-
gal documents using its NLP capabilities.
•Enhancing Anti-Money Laundering (AML) Measures:
FinGPT can be a valuable tool in ML operations. It can
be used to analyze the flow of funds, identify suspicious
patterns, and highlight transactions that require further in-
vestigation.
•Low-code development: Facilitating software creation
through user-friendly interfaces, reducing reliance on tra-
ditional programming.
•Financial education: Serving as an AI tutor simplifying
complex financial concepts for better financial literacy.
By linking these distinct yet interconnected components,
FinGPT provides a holistic and accessible solution for lever-
aging AI in finance, facilitating research, innovation, and
practical applications in the financial industry.
4 Data-Centric Approach for FinLLMs
For financial large language models (FinLLMs), a success-
ful strategy is not solely based on the capability of the model
architecture but is equally reliant on the training data. Our
data-centric approach prioritizes collecting, preparing, and
processing financial data.
Financial data comes from a variety of sources, with unique
characteristics. We delve into the specifics of different finan-
cial data sources, such as financial news, company fillings and
announcements, social media Discussions, and trends.
4.1 Financial News
Financial news carries vital information about the world
economy, specific industries, and individual companies. This
data source typically features:
•Timeliness:Financial news reports are timely and up-to-
date, often capturing the most recent developments in the
financial world.
•Dynamism:The information contained in financial news
is dynamic, changing rapidly in response to evolving eco-
nomic conditions and market sentiment.
•Influence:Financial news has a significant impact on fi-
nancial markets, influencing traders’ decisions and poten-
tially leading to dramatic market movements.
4.2 Company Filings and Announcements
Company filings and announcements are official documents
that corporations submit to regulatory bodies, providing in-
sight into a company’s financial health and strategic direction.
They feature:
•Granularity: These documents offer granular information
about a company’s financial status, including assets, liabil-
ities, revenue, and profitability.
•Reliability: Company fillings contain reliable and verified
data vetted by regulatory bodies.
•Periodicity: Company fillings are periodic, usually sub-
mitted on a quarterly or annual basis, offering regular snap-
shots of a company’s financial situation.
•Impactfulness: Company announcements often have sub-
stantial impacts on the market, influencing stock prices and
investor sentiment.
4.3 Social Media Discussions
Social media discussions related to finance will reflect pub-
lic sentiment towards specific stocks, sectors, or the overall
market. These discussions tend to exhibit:
•Variability:Social media discussions vary widely in tone,
content, and quality, making them rich, albeit complex,
sources of information.
•Real-time sentiment:These platforms often capture real-
time market sentiment, enabling the detection of trends and
shifts in public opinion.
•Volatility:Sentiments expressed on social media can be
highly volatile, changing rapidly in response to news events
or market movements.
4.4 Trends
Trends often observable through websites like Seeking Al-
pha, Google Trends, and other finance-oriented blogs and fo-
rums, offer critical insights into market movements and in-
vestment strategies. They feature:
•Analyst perspectives:These platforms provide access to
market predictions and investment advice from seasoned
financial analysts and experts.
6
•Market sentiment:The discourse on these platforms can
reflect the collective sentiment about specific securities,
sectors, or the overall market, providing valuable insights
into the prevailing market mood.
•Broad coverage:Trends data spans diverse securities and
market segments, offering comprehensive market coverage.
Each of these data sources provides unique insights into the
financial world. By integrating these diverse data types, Fin-
GPT can facilitate a comprehensive understanding of finan-
cial markets and enable effective financial decision-making.
5 Experiments: Financial Sentiment Analysis
In this section, we evaluate the sentiment analysis capability
of FinGPT. This experiment demonstrates the effectiveness
of FinGPT’s data-centric design and lightweight adaptation
methodology in real-world financial text classification.
5.1 Dataset
We utilize a large-scale financial news sentiment dataset cu-
rated through the FinGPT real-time data pipeline. The dataset
contains:
• Over620,000cleaned financial news headlines;
• Sources includingCNBC, Reuters, Yahoo Finance, Mar-
ketWatch, etc., collected through the FinNLP pipeline;
• Time span from2016–2024;
• Market-driven labels generated using short-term price
movement:
label=



Positive, r > θp
Negative, r <−θ n
Neutral,|r| ≤θ
whererdenotes the stock’s percentage price change fol-
lowing the news. This “self-labeled” approach aligns sen-
timent with true market reactions and avoids costly manual
annotation.
5.2 Model and Training Setup
We adopt a lightweight two-stage adaptation process.
LoRA-based Supervised Fine-Tuning
We fine-tune the pretrainedLlama-3.1-8B-Instruct
model using Low-Rank Adaptation (LoRA). Under a stan-
dard configuration of rankr= 8and scaling factorα= 16,
the total number of trainable parameters introduced by LoRA
is approximately8.3M, which is well below 0.1% of the orig-
inal 8B-parameter model.
The fine-tuning configuration is as follows:
• Trainable parameters:8.3M;
• LoRA rank:r= 8, scaling factorα= 16;
• Batch size:64;
• Learning rate:2×10 −4;
• Training epochs:3.
LoRA enables FinGPT to acquire domain-specific senti-
ment classification ability efficiently.
Reinforcement Learning on Stock Prices (RLSP)
To align the model with real market behavior, we further ap-
ply RLSP, where the environment is the financial market and
the reward is the stock price’s post-news reaction.
R=f(∆p)
This aligns the sentiment output with actual financial out-
comes and enhances generalization.
5.3 Baselines
We compare FinGPT against standard financial NLP base-
lines:
• FinBERT [Araci, 2019];
• BloombergGPT [Wuet al., 2023 ];
• ChatGPT (zero-shot) [OpenAI, 2023];
• Llama3.1-8B (zero-shot) [Grattafioriet al., 2024 ].
5.4 Evaluation Metrics
We evaluate performance using the following metrics:
• Accuracy;
• Precision, Recall, and F1-score for each class;
• Macro-F1 (to mitigate class imbalance);
• AUC for binary (positive/negative) subsets.
5.5 Results
Overall Performance
FinGPT outperforms all baselines significantly, demonstrat-
ing the benefits of data-centric labeling and RLSP reinforce-
ment alignment.
Ablation Study
LoRA performs most of the heavy lifting, while RLSP further
improves market alignment.
Case Study
We illustrate the model’s financial reasoning capability using
the following headline:
“Tesla cuts prices again in China as EV competi-
tion intensifies. ”
•Human Annotation: Negative (Price reductions are
commonly interpreted as a sign of weakening pric-
ing power and intensified competitive pressure, both of
which imply potential margin compression and typically
induce negative investor sentiment.)
•Base Llama3: Neutral (The model captures the surface-
level wording but fails to infer the underlying financial
implications of price competition.)
•FinGPT (SFT): Negative
•FinGPT (RLSP): Negative (with stronger alignment to
the subsequent price reaction)
This case highlights FinGPT’s ability to incorporate
domain-specific financial reasoning and to produce sentiment
predictions that are more consistent with market-impactful in-
terpretations.
7
Table 1: Sentiment Classification Performance
Model Acc. Macro-F1 Pos-F1 Neg-F1 Neu-F1
ChatGPT (0-shot) 63.4 61.7 64.0 59.1 62.0
Llama3.1-8B (0-shot) 57.9 54.4 56.1 53.2 54.0
FinBERT 71.2 69.9 73.0 69.1 67.5
FinGPT (LoRA-SFT) 78.8 77.3 79.6 76.8 75.4
FinGPT (SFT+RLSP) 82.1 80.9 83.4 81.5 77.8
Table 2: Ablation on LoRA and RLSP
Configuration Macro-F1
Base Llama3 54.4
+ LoRA SFT 77.3
+ RLSP80.9
5.6 Discussion
Key observations:
• Market-driven labels (self-labeled data) strongly im-
prove real-world applicability;
• LoRA reduces adaptation cost by∼1000×compared to
full fine-tuning;
• RLSP incorporates financial market feedback, distin-
guishing FinGPT from traditional supervised models.
This experiment confirms that FinGPT provides a scalable
and effective foundation for financial sentiment analysis.
6 Conclusion
In conclusion, the transformative integration of large lan-
guage models (LLMs) into the financial sector brings unique
complexities and vast opportunities. Navigating challenges
such as high temporal sensitivity, dynamic financial land-
scape, and a low signal-to-noise ratio in financial data calls
for efficient solutions. FinGPT responds innovatively by
leveraging pre-existing LLMs and fine-tuning them to spe-
cific financial applications. This approach significantly re-
duces adaptation costs and computational requirements com-
pared to models like BloombergGPT, offering a more acces-
sible, flexible, and cost-effective solution for financial lan-
guage modeling. Thus, it enables consistent updates to en-
sure model accuracy and relevance, a critical aspect in the
dynamic and time-sensitive world of finance.
7 Future Work
Future development of FinLLMs will focus on establish-
ing open, industry-level standards for financial large lan-
guage models. This includes advancing parameter-efficient
fine-tuning methods such as LoRA and QLoRA to support
low-cost, domain-specific customization across diverse finan-
cial institutions. Furthermore, FinLLMs will continue to
expand its unified data curation pipeline, promoting high-
quality, standardized financial datasets to streamline training
and evaluation. By integrating open-source tooling, repro-
ducible benchmarks, and transparent workflows, FinLLMs
aims to provide a foundation for reliable, scalable, and in-
teroperable financial AI systems.
Disclaimer: We are sharing codes for academic pur-
poses under the MIT education license. Nothing herein is
financial advice, and NOT a recommendation to trade real
money. Please use common sense and always first consult
a professional before trading or investing.
References
[Araci, 2019] Dogu Araci. Finbert: Financial sentiment
analysis with pre-trained language models.arXiv preprint
arXiv:1908.10063, 2019.
[Baoet al., 2021 ] Siqi Bao, Huang He, Fan Wang, Hua Wu,
Haifeng Wang, Wenquan Wu, Zhihua Wu, Zhen Guo, Hua
Lu, Xinxian Huang, et al. Plato-xl: Exploring the large-
scale pre-training of dialogue generation.arXiv preprint
arXiv:2109.09519, 2021.
[Brownet al., 2020 ] Tom Brown, Benjamin Mann, Nick Ry-
der, Melanie Subbiah, Jared D Kaplan, Prafulla Dhari-
wal, Arvind Neelakantan, Pranav Shyam, Girish Sastry,
Amanda Askell, et al. Language models are few-shot
learners.Advances in Neural Information Processing Sys-
tems, 33:1877–1901, 2020.
[Chenet al., 2021 ] Zhiyu Chen, Wenhu Chen, Charese Smi-
ley, Sameena Shah, Iana Borova, Dylan Langdon, Reema
Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge,
et al. Finqa: A dataset of numerical reasoning over finan-
cial data.arXiv preprint arXiv:2109.00122, 2021.
[DeLuciaet al., 2022 ] Alexandra DeLucia, Shijie Wu,
Aaron Mueller, Carlos Aguirre, Philip Resnik, and Mark
Dredze. Bernice: a multilingual pre-trained encoder for
Twitter. InProceedings of the Conference on Empir-
ical Methods in Natural Language Processing, pages
6191–6205, 2022.
[Dettmerset al., 2023 ] Tim Dettmers, Artidoro Pagnoni,
Ari Holtzman, and Luke Zettlemoyer. QLoRA: Ef-
ficient finetuning of quantized LLMs.arXiv preprint
arXiv:2305.14314, 2023.
[Devlinet al., 2018 ] Jacob Devlin, Ming-Wei Chang, Ken-
ton Lee, and Kristina Toutanova. Bert: Pre-training of
deep bidirectional transformers for language understand-
ing.arXiv preprint arXiv:1810.04805, 2018.
8
[Dredzeet al., 2016 ] Mark Dredze, Prabhanjan Kambadur,
Gary Kazantsev, Gideon Mann, and Miles Osborne. How
twitter is changing the nature of financial news discovery.
InProceedings of the second International Workshop on
Data Science for Macro-modeling, pages 1–5, 2016.
[Ethayarajh, 2019] Kawin Ethayarajh. How contextual are
contextualized word representations? comparing the ge-
ometry of bert, elmo, and gpt-2 embeddings.arXiv
preprint arXiv:1909.00512, 2019.
[Ghoshet al., 2021 ] Sohom Ghosh, Shovon Sengupta, Sudip
Naskar, and Sunny Kumar Singh. FinRead: A trans-
fer learning based tool to assess readability of defini-
tions of financial terms. InProceedings of the 18th In-
ternational Conference on Natural Language Processing
(ICON), pages 658–659, National Institute of Technology
Silchar, Silchar, India, December 2021. NLP Association
of India (NLPAI).
[Grattafioriet al., 2024 ] Aaron Grattafiori, Abhimanyu
Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek
Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur,
Alan Schelten, Alex Vaughan, et al. The llama 3 herd of
models.arXiv preprint arXiv:2407.21783, 2024.
[Huet al., 2021 ] Edward J Hu, Yelong Shen, Phillip Wallis,
Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen. LoRA: Low-rank adaptation of large
language models.International Conference on Learning
Representations, 2021.
[Jianget al., 2023 ] Albert Q. Jiang, Alexandre Sablayrolles,
Arthur Mensch, Chris Bamford, Devendra Singh Chaplot,
Diego de las Casas, Florian Bressand, Gianna Lengyel,
Guillaume Lample, Lucile Saulnier, L´elio Renard Lavaud,
Marie-Anne Lachaux, Pierre Stock, Teven Le Scao,
Thibaut Lavril, Thomas Wang, Timoth ´ee Lacroix, and
William El Sayed. Mistral 7b, 2023.
[Lewiset al., 2019 ] Mike Lewis, Yinhan Liu, Naman Goyal,
Marjan Ghazvininejad, Abdelrahman Mohamed, Omer
Levy, Ves Stoyanov, and Luke Zettlemoyer. Bart: De-
noising sequence-to-sequence pre-training for natural lan-
guage generation, translation, and comprehension.arXiv
preprint arXiv:1910.13461, 2019.
[Lewiset al., 2020 ] Patrick Lewis, Myle Ott, Jingfei Du, and
Veselin Stoyanov. Pretrained language models for biomed-
ical and clinical tasks: understanding and extending the
state-of-the-art. InProceedings of the 3rd Clinical Natural
Language Processing Workshop, pages 146–157, 2020.
[Liuet al., 2024 ] Aixin Liu, Bei Feng, Bing Xue, Bingx-
uan Wang, Bochao Wu, Chengda Lu, Chenggang
Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan,
et al. Deepseek-v3 technical report.arXiv preprint
arXiv:2412.19437, 2024.
[OpenAI, 2023] OpenAI. Chatgpt. https://chat.openai.com/,
2023. Large language model accessed via ChatGPT inter-
face.
[Ouyanget al., 2022 ] Long Ouyang, Jeffrey Wu, Xu Jiang,
Diogo Almeida, Carroll Wainwright, Pamela Mishkin,
Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex
Ray, et al. Training language models to follow instruc-
tions with human feedback.Advances in Neural Informa-
tion Processing Systems, 35:27730–27744, 2022.
[Radfordet al., 2018 ] Alec Radford, Karthik Narasimhan,
Tim Salimans, Ilya Sutskever, et al. Improving language
understanding by generative pre-training.OpenAI, 2018.
[Shahet al., 2022 ] Raj Sanjay Shah, Kunal Chawla, Dheeraj
Eidnani, Agam Shah, Wendi Du, Sudheer Chava, Na-
traj Raman, Charese Smiley, Jiaao Chen, and Diyi Yang.
When flue meets flang: Benchmarks and large pre-trained
language model for financial domain.arXiv preprint
arXiv:2211.00083, 2022.
[Sharmaet al., 2022 ] Soumya Sharma, Tapas Nayak,
Arusarka Bose, Ajay Kumar Meena, Koustuv Dasgupta,
Niloy Ganguly, and Pawan Goyal. Finred: A dataset for
relation extraction in financial domain. InCompanion
Proceedings of the Web Conference 2022, WWW ’22,
page 595–597, New York, NY , USA, 2022. Association
for Computing Machinery.
[Teamet al., 2023 ] Gemini Team, Rohan Anil, Sebastian
Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut,
Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Mil-
lican, et al. Gemini: a family of highly capable multimodal
models.arXiv preprint arXiv:2312.11805, 2023.
[Thoppilanet al., 2022 ] Romal Thoppilan, Daniel De Fre-
itas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha,
Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker,
Yu Du, et al. Lamda: Language models for dialog ap-
plications.arXiv preprint arXiv:2201.08239, 2022.
[Vaswaniet al., 2017 ] Ashish Vaswani, Noam Shazeer, Niki
Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you
need. InAdvances in Neural Information Processing Sys-
tems, volume 30. Curran Associates, Inc., 2017.
[Wuet al., 2023 ] Shijie Wu, Ozan Irsoy, Steven Lu, Vadim
Dabravolski, Mark Dredze, Sebastian Gehrmann, Prab-
hanjan Kambadur, David Rosenberg, and Gideon Mann.
BloombergGPT: A large language model for finance.
arXiv preprint arXiv:2303.17564, 2023.
[Yanget al., 2020 ] Hongyang Yang, Xiao-Yang Liu, Shan
Zhong, and Anwar Walid. Deep reinforcement learning
for automated stock trading: An ensemble strategy. InPro-
ceedings of the first ACM international conference on AI
in finance, pages 1–8, 2020.
[Yanget al., 2024 ] Hongyang Yang, Boyu Zhang, Neng
Wang, Cheng Guo, Xiaoli Zhang, Likun Lin, Junlin Wang,
Tianyu Zhou, Mao Guan, Runjia Zhang, et al. Fin-
robot: An open-source ai agent platform for financial ap-
plications using large language models.arXiv preprint
arXiv:2405.14767, 2024.
[Zhanget al., 2023 ] Boyu Zhang, Hongyang Yang, Tianyu
Zhou, Ali Babar, and Xiao-Yang Liu. Enhancing financial
sentiment analysis via retrieval augmented large language
models.ACM International Conference on AI in Finance
(ICAIF), 2023.
9
