<div align="center">

  <img src="assets/logo.png" alt="EmbodiedAI" width="600"/>

  <h1>具身智能入门 Embodied-AI-Learning—Instructions</h1>

</div>

本仓库主要是为了具身智能领域完全的新手入门所整理的学习资料和个人笔记心得，主要为了本人以后查阅相关资料以及为初学者提供相关的指南，希望能够帮到更多人。

<section id="start"></section>

## 具身智能的定义与相关的类别划分

> **定义**：Embodied AI 是指让智能体（agent）在具备“身体”的情况下与真实或虚拟环境交互，通过感知、行动和反馈来学习智能行为。  
> 它融合了计算机视觉、自然语言处理、机器人学、强化学习、多模态理解等多个方向。具体领域划分请见下表。

---

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

---

## 如何进行入门

感谢Jiyao哥提供给初学者的[具身初学者所需要具备的技能](./assets/For_Beginners.pdf),因此在入门过程中按照深度学习(deep learning),机器人学(Robotics)以及3D视觉(3DVISION)进行入门，在本仓库不同的类别下分别包含着与该领域相关的学习资料和内容，比如说，cs231n作为入门计算机视觉深度学习的良好学习材料，我在这里对于课程中的内容形成了笔记记录，并且完成了相应的assignments，所有的代码和笔记都在*Deep_learning*文件夹下，同理其余学习资料也会按照类别和内容整理在相应的类别下。


## 具身领域相关的信息
在这里特别感谢[具身智能学习指南](https://github.com/TianxingChen/Embodied-AI-Guide)所收集到的非常详细的从学习到工作以及社群的丰富信息，可以快速为初学者消除信息差，以下的内容是经过本人筛选并且结合实用性所总结

* 社交媒体:

  * 可以关注的公众号: **石麻日记 (超高质量!!!)**, 机器之心, 新智元, 量子位, Xbot具身知识库, 具身智能之心, 自动驾驶之心, 3D视觉工坊

  * AI领域值得关注的博主列表 [3]: [zhihu](https://zhuanlan.zhihu.com/p/682110383)

* 具身智能会投稿的较高质量会议与期刊：Science Robotics, TRO, IJRR, JFR, RSS, IROS, ICRA, ICCV, ECCV, ICML, CVPR, NeurIPS, ICLR, AAAI, ACL等。


* Awesome-Embodied-AI-Job (具身智能招贤榜): [Repo](https://github.com/StarCycle/Awesome-Embodied-AI-Job/tree/main)
  
* 社区:
  * DeepTimber Robotics Innovations Community, 深木科研交流社区: [website](https://gamma.app/public/DeepTimber-Robotics-Innovations-Community-A-Community-for-Multi-m-og0uv8mswl1a3q7?mode=doc)
  * 宇树具身智能社群: [website](https://www.unifolm.com/#/)
  * Simulately: Handy information and resources for physics simulators for robot learning research: [website](https://simulately.wiki/)
  * DeepTimber-地瓜机器人社区: [website](https://developer.d-robotics.cc/forumList?id=156&title=Deeptimber)
  * HuggingFace LeRobot (Europe, check the Discord): [website](https://github.com/huggingface/lerobot)
  * K-scale labs (US, check the Discord): [website](https://kscale.dev/)
