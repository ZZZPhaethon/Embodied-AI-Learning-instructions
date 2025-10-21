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
 
## 具身领域的公开数据集
* **Open X-Embodiment: Robotic Learning Datasets and RT-X Models**, [website](https://robotics-transformer-x.github.io/):  22种不同机器人平台的超过100万条真实机器人轨迹数据，覆盖了527种不同的技能和160,266项任务，主要集中在抓取和放置。
* **AgiBot World Datasets (智元机器人)**, [website](https://agibot-world.com/): 八十余种日常生活中的多样化技能，超过100万条轨迹数据，采集自**同构型机器人**, 多级质量把控和全程人工在环的策略，从采集员的专业培训，到采集过程中的严格管理，再到数据的筛选、审核和标注，每一个环节都经过了精心设计和严格把控。
* **RoboMIND**, [website](https://x-humanoid-robomind.github.io/): 包含了在479种不同任务中涉及96类独特物体的10.7万条真实世界演示轨迹，来自四种不同协作臂，任务被分为基础技能、精准操作、场景理解、柜体操作和协作任务五大类。
* **All Robots in One,** [website](https://imaei.github.io/project_pages/ario/): ARIO 数据集，包含了 **2D、3D、文本、触觉、声音 5 种模态的感知数据**，涵盖**操作**和**导航**两大类任务，既有**仿真数据**，也有**真实场景数据**，并且包含多种机器人硬件，有很高的丰富度。在数据规模达到三百万的同时，还保证了数据的统一格式，是目前具身智能领域同时达到高质量、多样化和大规模的开源数据集。
* **MimicGen** [26 Oct 2023, CoRL 2023],[repo](https://github.com/NVlabs/mimicgen),[website](https://mimicgen.github.io/)：基于Robosuite与MuJoCo开发的高效数据生成框架，主要聚焦于单臂机器人桌面操作任务，支持多种主流机器人型号。MimicGen提出了一种自动化的数据扩增方法，能够从少量真实人类演示中自动生成大量模拟数据，例如仅使用200段真人演示即可生成超过5万条仿真演示数据，涵盖18类常见机器人任务。
* **RoboCasa** [4 Jun 2024],[repo](https://github.com/robocasa/robocasa), [website](https://robocasa.ai/):基于RoboSuite与MimicGen在MuJoCo中构建的高仿真厨房任务仿真平台。RoboCasa提供了120个多样化厨房环境，包含超过2500个3D物体模型。平台支持单臂、双臂、人形机器人以及移动底座搭载机械臂的机器人系统。此外，RoboCasa内置了25种基础原子任务和75种组合任务，能够真实模拟机器人在复杂厨房场景中的多样化操作行为。
* **DexMimicGen** [6 Mar 2025, ICRA 2025],[repo](https://github.com/NVlabs/dexmimicgen/), [website](https://dexmimicgen.github.io/):以RoboSuite和MimicGen为基础，在MuJoCo平台上构建的高保真双臂桌面操作任务仿真环境。DexMimicGen涵盖9类典型双臂任务，提出了增强版real2sim2real数据自动生成技术，只需60段真实人类演示便可生成2.1万条高质量仿真数据。相比原版MimicGen，该框架显著提升了数据生成效率和真实感，使机器人双臂协作任务的仿真训练更具实用性。
* **FUSE Dataset** [ICRA 2025] [website](https://fuse-model.github.io/) 包含26,866条远程操控轨迹，涵盖桌面抓取、购物袋内抓取和按钮按压三类任务。机器人通过Meta Oculus Quest 2 VR头显操作，任务结合语言指令和复杂视觉遮挡，支持多传感器与语言融合的机器人策略研究。
* **BiPlay Dataset** [website](https://dit-policy.github.io/):为了解决现有双臂数据集任务单一、环境固定的问题，BiPlay数据集采用随机物体和背景，采集多样化双臂操作轨迹。数据由多段3.5分钟的机器人操作视频拆分成7023个带语言任务描述的剪辑，总计10小时数据，支持双臂操作泛化研究。
* **DROID (Distributed Robot Interaction Dataset)**[website](https://droid-dataset.github.io/)：包含76,000条示范轨迹，约350小时交互数据，覆盖564个场景和86个任务。数据由50名采集员在北美、亚洲和欧洲12个月内收集，场景和任务多样性显著提升。基于DROID训练的策略表现更优、鲁棒性和泛化能力更强。数据集、训练代码及硬件搭建指南均已开源。
* **BridgeData V2**[website](https://rail-berkeley.github.io/bridgedata/)：包含60,096条轨迹数据，涵盖24个环境和13类技能，支持基于目标图像或自然语言指令的多任务开放词汇学习。数据主要采集自7个玩具厨房环境及多样桌面、洗衣机等场景，轨迹包括50,365条远程操控示范和9,731条脚本策略执行。每条轨迹均标注对应自然语言任务描述，促进跨环境和跨机构的技能泛化研究。
* **Ego4DSounds** [website](https://ego4dsounds.github.io/)：作为Ego4D大规模第一人称视角数据集的多模态子集，包含超过120万条视频剪辑，覆盖3000多个不同日常场景和行为，如烹饪、清洁、购物和社交等。数据强调动作与环境声音的高度对应，配备带时间戳的动作叙述，支持具身智能中动作感知、多模态融合及声音生成等任务的研究。
* **RH20T**[website](https://rh20t.github.io/)：人机交互数据集，包含丰富的人脸和语音信息，使用时需注意隐私保护，仅限模型训练。数据原始规模约40TB，提供尺寸缩减版（约5TB RGB，10TB RGB-D）。包含7组RGB视频及对应深度数据，附带相机标定和机器人关节角度信息。数据通过Google Drive和百度云公开下载。

