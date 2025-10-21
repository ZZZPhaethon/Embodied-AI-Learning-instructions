<div align="center">

  <img src="assets/logo.png" alt="EmbodiedAI" width="600"/>

  <h1>具身智能学习指导 Embodied-AI-Learning—Instructions</h1>

</div>

本仓库主要是为了具身智能领域完全的新手入门所整理的学习资料和个人笔记心得，主要为了本人以后查阅相关资料以及为初学者提供相关的指南，希望能够帮到更多人。

<section id="start"></section>

# 1. Start From Here - 从这里开始，具身智能的定义与相关的类别划分

> **定义**：Embodied AI 是指让智能体（agent）在具备“身体”的情况下与真实或虚拟环境交互，通过感知、行动和反馈来学习智能行为。  
> 它融合了计算机视觉、自然语言处理、机器人学、强化学习、多模态理解等多个方向。

| 领域类别 | 典型任务 | 代表平台 / 模型 | 关键挑战 |
|-----------|-----------|-----------------|-----------|
|  **机器人与具身智能体** | 家用机器人、服务机器人、任务执行（如取物、清洁） | RoboGPT, SayCan, PaLM-E, RT-2 | 语言到动作的映射、安全约束、物理执行可靠性 |
|  **视觉导航与交互 (Embodied Navigation)** | 视觉-语言导航（VLN）、目标导航（ObjectNav, ImageNav） | Habitat, Matterport3D, Gibson, AI2-THOR | 空间理解、跨环境泛化、语言指令解析 |
|  **具身操作 (Embodied Manipulation)** | 抓取、旋转、开门、组装、烹饪等复杂操作 | RLBench, ManiSkill, Isaac Gym, Habitat 3.0 | 高维控制、视觉-动作对齐、Sim2Real 迁移 |
|  **虚拟世界与仿真环境学习** | 在3D环境中学习任务规划与交互 | ProcTHOR, VirtualHome, BEHAVIOR, iGibson | 逼真物理仿真、任务分解与泛化 |
|  **具身认知与学习 (Cognitive Embodied AI)** | 物理推理、社会行为学习、因果理解 | IntPhys, SocialGym, BabyAI | 具身推理建模、跨模态记忆、抽象概念形成 |
|  **语言-行动对齐 (Language-Action Alignment)** | 自然语言控制机器人、代码生成 | SayCan, Code-as-Policies, See-Say-Do | 指令语义 grounding、规划与动作生成一致性 |
|  **工具与评测生态 (Benchmarks & Tools)** | 多任务综合评测、通用智能体评估 | ALFRED, BEHAVIOR, EAI Leaderboard | 统一指标、任务复杂性、多模态能力评估 |
|  **新兴方向 (Emerging Directions)** | Embodied Foundation Models, 数字孪生、社会具身AI、AR/VR智能体 | PaLM-E, RT-X, VILA, GPT-4V, Metaverse Agents | 现实迁移、能耗与计算成本、伦理与安全性 |


## How - 如何学习这份指南

我们希望的是帮助新人快速建立领域认知, 所以设计理念是：**简要**介绍目前具身智能涉及到的主要技术, 让大家知道不同的技术能够解决什么问题, 未来想要深入发展的时候能够有头绪。
