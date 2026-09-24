# Chaoyu Wang

**Seeking RA in LLM Safety and Alignment**

Email: samuelwang997@gmail.com

## Education

### Northwestern University — Evanston, IL

*Sep 2024 – June 2025*

**Master of Science: Engineering Science & Applied Math**

- Coursework: Partial Differential Equations, High-Performance Scientific Computing, Data-Driven Methods for Dynamical Systems

### University of California San Diego — San Diego, CA

*Sep 2022 – June 2024*

**Bachelor of Science: Applied Math**

- Coursework: Applied Linear Algebra, Multivariate Calculus, Stochastic Processes, Numerical Analysis, Real Analysis

## Professional Experiences

### Ad Campaign Agent: SFT Fine-tuning & Tool-chain Alignment

**GuruGame HK** · Hongkong SAR · *Sep 2025 – Mar 2026*

- Built Ad campaign Agent covering 4 workflows (creative sourcing, Ad upload, performance monitoring, internal knowledge query) with chained tool calls, multi-turn slot-filling, and a Milvus-backed RAG pipeline for internal knowledge retrieval
- Constructed a 3,000+ seed dataset via rule-based templates and GPT synthesis; applied slot perturbation, semantic paraphrasing, and multi-turn augmentation with 80/20 stratified split
- Resolved long-tail imbalance in multi-turn samples by progressively decomposing dialogues (1→N turns) and oversampling final-turn responses 3×, expanding training set from 3,030 to 7,600+ samples
- Deployed after A/B validation with multi-dimensional evaluation (routing accuracy, function-call EM, end-to-end task success rate, human quality score) and established a data flywheel for continuous iteration

**Results:** End-to-end task completion 86%+ (from <40%); routing accuracy +18pp; function-call EM +25pp; inference cost reduced ~54% vs. pure API-chaining baseline

### Wealth Management RAG Q&A System & GRPO Alignment

**Huatai Securities (Nanjing)** · Nanjing, China · *Apr 2025 – Aug 2025*

- Built a RAG-based Q&A system over 2,000+ multimodal documents (PDF, PPT, scanned images) to deliver real-time client needs matching and advisory recommendations for sales and wealth management teams
- Implemented structure-aware chunking (section headers, table boundaries, clause numbering) with hierarchical tagging; fine-tuned BGE via contrastive learning on internal financial Q&A data to improve domain terminology retrieval
- Deployed hybrid retrieval (BM25 + dense vector) with a fine-tuned Cross-Encoder reranker; applied Context Injection and Query Rewrite to resolve coreference in multi-turn dialogues; enforced source attribution in prompts for compliance traceability
- Curated 3,000+ high-quality prompts from real consultation logs and red-team corpora; designed a 3-dimensional Reward Function (grounded faithfulness via LLM-as-Judge, multi-turn context consistency, response naturalness) to drive GRPO policy optimization on DeepSeek-R1-7B with group-normalized reward signals

**Results:** Recall@5 52%→88%, MRR 0.38→0.72; hallucination rate reduced from 8.0% to 1.2%; multi-turn context error rate from 3.0% to 0.5%; generalization task completion 81% on unseen scenarios

## Technical Skills

| Category | Skills |
| --- | --- |
| **Languages** | Python (Pytorch), JavaScript, SQL, MATLAB |
| **Machine Learning** | Synthetic Data Construction, Data Augmentation, SFT, PEFT / LoRA, RL Alignment (GRPO), RAG, Contrastive Learning, Dense Retrieval, Reranking |
| **Web & Fullstack** | Next.js, React, Flask, REST API, Prisma, Prompt Engineering |
| **Databases & Infra** | PostgreSQL, Milvus, Elasticsearch, Docker, Git, RunPod |
| **Data & Engineering** | Pandas, NumPy, Third-party API Integration, Data Pipeline Design |
