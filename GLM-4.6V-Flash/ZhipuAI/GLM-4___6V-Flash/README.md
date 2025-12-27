---
frameworks:
- Pytorch
license: MIT License
tags: []
tasks:
- image-text-to-text
---

# GLM-4.6V

<div align="center">
<img src=https://raw.githubusercontent.com/zai-org/GLM-V/refs/heads/main/resources/logo.svg width="40%"/>
</div>

本模型属于 GLM-V 模型家族，相关内容可参考论文  
[GLM-4.1V-Thinking and GLM-4.5V: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning](https://huggingface.co/papers/2507.01006)。

-   **GLM-4.6V 博客**：[https://z.ai/blog/glm-4.6v](https://z.ai/blog/glm-4.6v)
-   **论文**：[https://huggingface.co/papers/2507.01006](https://huggingface.co/papers/2507.01006)
-   **GitHub 仓库**：[https://github.com/zai-org/GLM-V](https://github.com/zai-org/GLM-V)
-   **在线 Demo**：[https://chat.z.ai/](https://chat.z.ai/)
-   **API 访问**：[ZhipuAI 开放平台](https://docs.z.ai/guides/vlm/glm-4.5v)
-   **桌面助手应用**：[https://huggingface.co/spaces/zai-org/GLM-4.5V-Demo-App](https://huggingface.co/spaces/zai-org/GLM-4.5V-Demo-App)

## 简介

GLM-4.6V 系列包含两个版本：GLM-4.6V（106B），面向云端与高性能集群场景；GLM-4.6V-Flash（9B），面向本地部署与低延迟应用的轻量版。GLM-4.6V 在训练中将上下文窗口扩展至 128k tokens，并在相同参数规模下实现视觉理解的 SOTA 性能。

更重要的是，我们首次在模型架构中原生集成了 Function Calling 能力，有效打通了从“视觉感知”到“可执行行动”的桥梁，为真实业务场景中的多模态智能体提供统一的技术底座。

![GLM-4.6V Benchmarks](https://raw.githubusercontent.com/zai-org/GLM-V/refs/heads/main/resources/bench_46v.jpeg)

在主要多模态基准测试中取得同规模下的 SOTA 性能之外，GLM-4.6V 还带来了一系列重要特性：

- **原生多模态工具调用（Native Multimodal Function Calling）**  
  支持基于视觉的原生工具调用。图片、截图和文档页面可直接作为工具输入，无需文本转换；图表、检索图像和渲染页面等视觉输出，也能被模型理解并融入推理链，实现从“感知→理解→执行”的完整闭环。

- **图文交织内容生成（Interleaved Image-Text Content Generation）**  
  支持基于复杂多模态输入生成高质量混合内容。GLM-4.6V 能理解文档、用户输入以及工具检索图像构成的多模态上下文，并生成连贯的图文交织内容。模型在生成过程中可主动调用搜索与检索工具，以补充文本与视觉内容，生成丰富且视觉支撑强的结果。

- **多模态文档理解（Multimodal Document Understanding）**  
  支持最长 128K tokens 的多文档/长文档输入，直接以图像方式解析排版丰富的页面。模型可联合理解文本、布局、图表、表格与插图，实现无需转换为纯文本即可高质量解析复杂文档。

- **前端还原与视觉编辑（Frontend Replication & Visual Editing）**  
  能从 UI 截图中重建像素级逼真的 HTML/CSS，并支持自然语言驱动的可视化修改。模型可识别页面布局、组件与样式，生成干净代码，并根据用户指令迭代进行视觉编辑。

**本 Hugging Face 仓库托管的是 `GLM-4.6V-Flash` 模型，属于 `GLM-V` 系列的一部分。**

## 使用方法

### 环境安装

对于 `SGLang`：

```bash
pip install sglang>=0.5.6post1
pip install transformers>=5.0.0rc0
```

对于 `vLLM`：
```bash
pip install vllm>=0.12.0
pip install transformers>=5.0.0rc0
```

使用 `transformers` 快速开始:

```python
from transformers import AutoProcessor, Glm4vForConditionalGeneration
import torch

MODEL_PATH = "zai-org/GLM-4.6V-Flash"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png"
            },
            {
                "type": "text",
                "text": "describe this image"
            }
        ],
    }
]
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = Glm4vForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)
inputs.pop("token_type_ids", None)
generated_ids = model.generate(**inputs, max_new_tokens=8192)
output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
print(output_text)
```

## 评测设置

我们主要使用 vLLM 作为模型推理的后端。在视频任务上，为了获得更快且更稳定的性能，我们采用 SGLang。若要复现我们的排行榜结果，建议使用以下解码参数：

+	top_p: 0.6
+	top_k: 2
+	temperature: 0.8
+	repetition_penalty: 1.1
+	max_generate_tokens: 16K

更多使用方式请参考我们的 [Github](https://github.com/zai-org/GLM-V)￼。

## 已修复与待解决问题

自 GLM-4.1V 开源以来，我们收到了大量社区反馈，也清楚模型仍存在不少不足。在后续迭代中，我们尝试修复了一些常见问题——例如思维复读、输出格式错误等——在当前版本中已有一定改善。

但模型仍存在一些局限性，我们会尽快修复：
1. 纯文本问答能力仍有较大提升空间。本次研发侧重多模态视觉能力，后续版本将同步增强纯文本能力。
2.	在复杂 prompt 场景下，模型仍可能出现过度思考或复读现象。
3.	某些情况下模型会在末尾重复回答内容。
4.	在数数、识别具体人物等视觉感知任务上仍有改进空间。

感谢大家的耐心与包容，也欢迎在 issue 区提出建议与反馈，我们会尽力回应与改善！

## 引用

如果你使用了本模型，请引用以下论文：

```bibtex
@misc{vteam2025glm45vglm41vthinkingversatilemultimodal,
      title={GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning}, 
      author={V Team and Wenyi Hong and Wenmeng Yu and Xiaotao Gu and Guo Wang and Guobing Gan and Haomiao Tang and Jiale Cheng and Ji Qi and Junhui Ji and Lihang Pan and Shuaiqi Duan and Weihan Wang and Yan Wang and Yean Cheng and Zehai He and Zhe Su and Zhen Yang and Ziyang Pan and Aohan Zeng and Baoxu Wang and Bin Chen and Boyan Shi and Changyu Pang and Chenhui Zhang and Da Yin and Fan Yang and Guoqing Chen and Jiazheng Xu and Jiale Zhu and Jiali Chen and Jing Chen and Jinhao Chen and Jinghao Lin and Jinjiang Wang and Junjie Chen and Leqi Lei and Letian Gong and Leyi Pan and Mingdao Liu and Mingde Xu and Mingzhi Zhang and Qinkai Zheng and Sheng Yang and Shi Zhong and Shiyu Huang and Shuyuan Zhao and Siyan Xue and Shangqin Tu and Shengbiao Meng and Tianshu Zhang and Tianwei Luo and Tianxiang Hao and Tianyu Tong and Wenkai Li and Wei Jia and Xiao Liu and Xiaohan Zhang and Xin Lyu and Xinyue Fan and Xuancheng Huang and Yanling Wang and Yadong Xue and Yanfeng Wang and Yanzi Wang and Yifan An and Yifan Du and Yiming Shi and Yiheng Huang and Yilin Niu and Yuan Wang and Yuanchang Yue and Yuchen Li and Yutao Zhang and Yuting Wang and Yu Wang and Yuxuan Zhang and Zhao Xue and Zhenyu Hou and Zhengxiao Du and Zihan Wang and Peng Zhang and Debing Liu and Bin Xu and Juanzi Li and Minlie Huang and Yuxiao Dong and Jie Tang},
      year={2025},
      eprint={2507.01006},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.01006}, 
}
```