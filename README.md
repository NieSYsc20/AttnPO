# ATTNPO: Attention-Guided Process Supervision for Efficient Reasoning



<div align="center">
  <a href="https://arxiv.org/abs/2602.09953"><img alt="arXiv Paper"
    src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv"/></a>
  <a href="https://huggingface.co/collections/ShuaiyiNie/attnpo"><img alt="Hugging Face Collections"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow"/></a>
</div>

## 🚀 Release

**[2025/04/20]** 🔥 We have open-sourced the AttnPO codebase and model weights!

**[2025/04/06]** 🎉 Our paper has been accepted to **ACL 2026 (Main)**!

**[2025/02/10]** 🔥 We released our paper on arXiv: [AttnPO: Attention-Guided Process Supervision for Efficient Reasoning](https://arxiv.org/abs/2602.09953)

## 📖 Introduction

Large reasoning models trained with reinforcement learning and verifiable rewards (RLVR) achieve strong performance on complex reasoning tasks, yet often overthink, generating redundant reasoning without performance gains. 
- **Trajectory-level length penalties** often fail to effectively shorten reasoning length and degrade accuracy, as they uniformly treat all reasoning steps and lack fine-grained signals to distinguish redundancy from necessity.
- **Process-supervised methods** are typically resource-intensive and suffer from inaccurate credit assignment. 

To address these issues, we propose ATTNPO, a low-overhead process-supervised RL framework that leverages the model’s intrinsic attention signals for step-level credit assignment. We first identify a set of special attention heads that naturally focus on essential steps while suppressing redundant ones. By leveraging the attention scores of these heads, We then employ two sub-strategies to mitigate overthinking by discouraging redundant steps while preserving accuracy by reducing penalties on essential steps. Experimental results show that ATTNPO substantially reduces reasoning length while significantly improving performance across 9 benchmarks.

<div align="center">
  <img src="assets/kfh.png" style="width:50%; height:auto;" />
  <img src="assets/framework.png" style="width:50%; height:auto;" />
</div>
