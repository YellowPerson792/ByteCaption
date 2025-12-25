# InternVL

The InternVL3 family of Visual Language Models was introduced in [InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models](https://huggingface.co/papers/2504.10479).

The abstract from the paper is the following:

*We introduce InternVL3, a significant advancement in the InternVL series featuring a native multimodal pre-training paradigm. Rather than adapting a text-only large language model (LLM) into a multimodal large language model (MLLM) that supports visual inputs, InternVL3 jointly acquires multimodal and linguistic capabilities from both diverse multimodal data and pure-text corpora during a single pre-training stage. This unified training paradigm effectively addresses the complexities and alignment challenges commonly encountered in conventional post-hoc training pipelines for MLLMs. To further improve performance and scalability, InternVL3 incorporates variable visual position encoding (V2PE) to support extended multimodal contexts, employs advanced post-training techniques such as supervised fine-tuning (SFT) and mixed preference optimization (MPO), and adopts test-time scaling strategies alongside an optimized training infrastructure. Extensive empirical evaluations demonstrate that InternVL3 delivers superior performance across a wide range of multi-modal tasks. In particular, InternVL3-78B achieves a score of 72.2 on the MMMU benchmark, setting a new state-of-the-art among open-source MLLMs. Its capabilities remain highly competitive with leading proprietary models, including ChatGPT-4o, Claude 3.5 Sonnet, and Gemini 2.5 Pro, while also maintaining strong pure-language proficiency. In pursuit of open-science principles, we will publicly release both the training data and model weights to foster further research and development in next-generation MLLMs.*

 Overview of InternVL3 models architecture, which is the same as InternVL2.5. Taken from the original checkpoint. 

 Comparison of InternVL3 performance on OpenCompass against other SOTA VLLMs. Taken from the original checkpoint. 

This model was contributed by [yonigozlan](https://huggingface.co/yonigozlan).
The original code can be found [here](https://github.com/OpenGVLab/InternVL).

## Usage example

### Inference with Pipeline

Here is how you can use the `image-text-to-text` pipeline to perform inference with the `InternVL3` models in just a few lines of code:

```python
>>> from transformers import pipeline

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {
...                 "type": "image",
...                 "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
...             },
...             {"type": "text", "text": "Describe this image."},
...         ],
...     },
... ]

>>> pipe = pipeline("image-text-to-text", model="OpenGVLab/InternVL3-1B-hf")
>>> outputs = pipe(text=messages, max_new_tokens=50, return_full_text=False)
>>> outputs[0]["generated_text"]
'The image showcases a vibrant scene of nature, featuring several flowers and a bee. \n\n1. **Foreground Flowers**: \n   - The primary focus is on a large, pink cosmos flower with a prominent yellow center. The petals are soft and slightly r'
```

### Inference on a single image

This example demonstrates how to perform inference on a single image with the InternVL models using chat templates.

> [!NOTE]
> Note that the model has been trained with a specific prompt format for chatting. Use `processor.apply_chat_template(my_conversation_dict)` to correctly format your prompts.

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText
>>> import torch

>>> model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map="auto", dtype=torch.bfloat16)

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
...             {"type": "text", "text": "Please describe the image explicitly."},
...         ],
...     }
... ]

>>> inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

>>> generate_ids = model.generate(**inputs, max_new_tokens=50)
>>> decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

>>> decoded_output
'The image shows two cats lying on a pink blanket. The cat on the left is a tabby with a mix of brown, black, and white fur, and it appears to be sleeping with its head resting on the blanket. The cat on the'
```

### Text-only generation

This example shows how to generate text using the InternVL model without providing any image input.

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText
>>> import torch

>>> model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map="auto", dtype=torch.bfloat16)

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {"type": "text", "text": "Write a haiku"},
...         ],
...     }
... ]

>>> inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(torch_device, dtype=torch.bfloat16)

>>> generate_ids = model.generate(**inputs, max_new_tokens=50)
>>> decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

>>> print(decoded_output)
"Whispers of dawn,\nSilent whispers of the night,\nNew day's light begins."
```

### Batched image and text inputs

InternVL models also support batched image and text inputs.

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText
>>> import torch

>>> model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map="auto", dtype=torch.bfloat16)

>>> messages = [
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
...                 {"type": "text", "text": "Write a haiku for this image"},
...             ],
...         },
...     ],
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
...                 {"type": "text", "text": "Describe this image"},
...             ],
...         },
...     ],
... ]

>>> inputs = processor.apply_chat_template(messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

>>> output = model.generate(**inputs, max_new_tokens=25)

>>> decoded_outputs = processor.batch_decode(output, skip_special_tokens=True)
>>> decoded_outputs
["user\n\nWrite a haiku for this image\nassistant\nSilky lake,  \nWooden pier,  \nNature's peace.",
 'user\n\nDescribe this image\nassistant\nThe image shows a street scene with a traditional Chinese archway, known as a "Chinese Gate" or "Chinese Gate of']
```

### Batched multi-image input

This implementation of the InternVL models supports batched text-images inputs with different number of images for each text.

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText
>>> import torch

>>> model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map="auto", dtype=torch.bfloat16)

>>> messages = [
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
...                 {"type": "text", "text": "Write a haiku for this image"},
...             ],
...         },
...     ],
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "image", "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"},
...                 {"type": "image", "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg"},
...                 {"type": "text", "text": "These images depict two different landmarks. Can you identify them?"},
...             ],
...         },
...     ],
>>> ]

>>> inputs = processor.apply_chat_template(messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

>>> output = model.generate(**inputs, max_new_tokens=25)

>>> decoded_outputs = processor.batch_decode(output, skip_special_tokens=True)
>>> decoded_outputs
["user\n\nWrite a haiku for this image\nassistant\nSilky lake,  \nWooden pier,  \nNature's peace.",
 'user\n\n\nThese images depict two different landmarks. Can you identify them?\nassistant\nYes, these images depict the Statue of Liberty and the Golden Gate Bridge.']
```

### Video input

InternVL models can also handle video inputs. Here is an example of how to perform inference on a video input using chat templates.

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

>>> model_checkpoint = "OpenGVLab/InternVL3-8B-hf"
>>> quantization_config = BitsAndBytesConfig(load_in_4bit=True)
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, quantization_config=quantization_config)

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {
...                 "type": "video",
...                 "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
...             },
...             {"type": "text", "text": "What type of shot is the man performing?"},
...         ],
...     }
>>> ]
>>> inputs = processor.apply_chat_template(
...     messages,
...     return_tensors="pt",
...     add_generation_prompt=True,
...     tokenize=True,
...     return_dict=True,
...     num_frames=8,
>>> ).to(model.device, dtype=torch.float16)

>>> output = model.generate(**inputs, max_new_tokens=25)

>>> decoded_output = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
>>> decoded_output
'The man is performing a forehand shot.'
```

### Interleaved image and video inputs

This example showcases how to handle a batch of chat conversations with interleaved image and video inputs using chat template.

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
>>> import torch

>>> model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map="auto", dtype=torch.bfloat16)

>>> messages = [
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "image", "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"},
...                 {"type": "image", "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg"},
...                 {"type": "text", "text": "These images depict two different landmarks. Can you identify them?"},
...             ],
...         },
...     ],
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "video", "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"},
...                 {"type": "text", "text": "What type of shot is the man performing?"},
...             ],
...         },
...     ],
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
...                 {"type": "text", "text": "Write a haiku for this image"},
...             ],
...         },
...     ],
>>> ]
>>> inputs = processor.apply_chat_template(
...     messages,
...     padding=True,
...     add_generation_prompt=True,
...     tokenize=True,
...     return_dict=True,
...     return_tensors="pt",
>>> ).to(model.device, dtype=torch.bfloat16)

>>> outputs = model.generate(**inputs, max_new_tokens=25)

>>> decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)
>>> decoded_outputs
['user\n\n\nThese images depict two different landmarks. Can you identify them?\nassistant\nThe images depict the Statue of Liberty and the Golden Gate Bridge.',
 'user\nFrame1: \nFrame2: \nFrame3: \nFrame4: \nFrame5: \nFrame6: \nFrame7: \nFrame8: \nWhat type of shot is the man performing?\nassistant\nA forehand shot',
 "user\n\nWrite a haiku for this image\nassistant\nSilky lake,  \nWooden pier,  \nNature's peace."]
```

## InternVLVisionConfig[[transformers.InternVLVisionConfig]]

#### transformers.InternVLVisionConfig[[transformers.InternVLVisionConfig]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/internvl/configuration_internvl.py#L21)

This is the configuration class to store the configuration of a [InternVLVisionModel](/docs/transformers/v5.0.0rc1/en/model_doc/internvl#transformers.InternVLVisionModel). It is used to instantiate an InternVLVisionModel
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield
a similar configuration to that of the InternVL3-1B.
e.g. [OpenGVLab/InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)

Example:

```python
>>> from transformers import InternVLVisionConfig, InternVLVisionModel

>>> # Initializing a InternVLVisionModel OpenGVLab/InternVL3-1B-hf style configuration
>>> configuration = InternVLVisionConfig()

>>> # Initializing a model (with random weights) from the OpenGVLab/InternVL3-1B-hf configuration
>>> model = InternVLVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

hidden_size (`int`, *optional*, defaults to 1024) : Dimensionality of the encoder layers and the pooler layer.

num_hidden_layers (`int`, *optional*, defaults to 24) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 16) : Number of attention heads for each attention layer in the Transformer encoder.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to add a bias to the queries, keys and values.

use_qk_norm (`bool`, *optional*, defaults to `False`) : Whether to apply normalization to the queries and keys before the attention operation.

intermediate_size (`int`, *optional*, defaults to 4096) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

hidden_dropout_prob (`float`, *optional*, defaults to 0.0) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_dropout (`float`, *optional*, defaults to 0.0) : Dropout probability for attention weights.

projection_dropout (`float`, *optional*, defaults to 0.0) : Dropout probability for the projection layer.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

norm_type (`str`, *optional*, defaults to `"layer_norm"`) : The type of normalization to use in the encoder. Can be `"layer_norm"` or `"rms_norm"`.

layer_norm_eps (`float`, *optional*, defaults to 1e-06) : The epsilon used by the layer normalization layers.

image_size (`int` or `list[int]`, *optional*, defaults to `[448, 448]`) : The size (resolution) of each image.

patch_size (`int` or `list[int]`, *optional*, defaults to `[14, 14]`) : The size (resolution) of each patch.

num_channels (`int`, *optional*, defaults to 3) : The number of input channels.

use_mask_token (`bool`, *optional*, defaults to `False`) : Whether to use a mask token for masked image modeling.

use_absolute_position_embeddings (`bool`, *optional*, defaults to `True`) : Whether to use BERT-style absolute position embeddings.

layer_scale_init_value (`float`, *optional*, defaults to 0.1) : Scale to use in the self-attention layers. 0.1 for base, 1e-5 for large. Set 0 to disable layer scale.

use_mean_pooling (`bool`, *optional*, defaults to `True`) : Whether to mean pool the final hidden states of the patches instead of using the final hidden state of the CLS token, before applying the classification head.

## InternVLConfig[[transformers.InternVLConfig]]

#### transformers.InternVLConfig[[transformers.InternVLConfig]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/internvl/configuration_internvl.py#L142)

This is the configuration class to store the configuration of a [InternVLForConditionalGeneration](/docs/transformers/v5.0.0rc1/en/model_doc/internvl#transformers.InternVLForConditionalGeneration). It is used to instantiate a
InternVL model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of InternVL3-1B.
e.g. [OpenGVLab/InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.0.0rc1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.0.0rc1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

```python
>>> from transformers import InternVLForConditionalGeneration, InternVLConfig

>>> # Initializing a InternVL style configuration
>>> configuration = InternVLConfig()

>>> # Initializing a model (with random weights) from the OpenGVLab/InternVL3-1B-hf configuration
>>> model = InternVLForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `InternVisonConfig`) : The config object or dictionary of the vision backbone.

text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen2Config`) : The config object or dictionary of the text backbone.

image_token_id (`int`, *optional*, defaults to 151667) : The image token index to encode the image prompt.

image_seq_length (`int`, *optional*, defaults to 256) : Number of image tokens to use per image patch.

downsample_ratio (`float`, *optional*, defaults to 0.5) : Factor by which to downsample the image.

projector_hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the projector.

vision_feature_layer (`int`, *optional*, defaults to -1) : The index of the layer to use as the image features.

vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`) : The feature selection strategy used to select the vision feature from the vision backbone. Can be one of `"default"` or `"full"`.

## InternVLVisionModel[[transformers.InternVLVisionModel]]

#### transformers.InternVLVisionModel[[transformers.InternVLVisionModel]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/internvl/modeling_internvl.py#L431)

The bare Internvl Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.0.0rc1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.InternVLVisionModel.forwardhttps://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/internvl/modeling_internvl.py#L449[{"name": "pixel_values", "val": ": Tensor"}, {"name": "bool_masked_pos", "val": ": typing.Optional[torch.BoolTensor] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [GotOcr2ImageProcessor](/docs/transformers/v5.0.0rc1/en/model_doc/got_ocr2#transformers.GotOcr2ImageProcessor). See [GotOcr2ImageProcessor.__call__()](/docs/transformers/v5.0.0rc1/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([InternVLProcessor](/docs/transformers/v5.0.0rc1/en/model_doc/internvl#transformers.InternVLProcessor) uses
  [GotOcr2ImageProcessor](/docs/transformers/v5.0.0rc1/en/model_doc/got_ocr2#transformers.GotOcr2ImageProcessor) for processing images).
- **bool_masked_pos** (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*) --
  Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).0`transformers.models.internvl.modeling_internvl.InternVLVisionModelOutputWithPooling` or `tuple(torch.FloatTensor)`A `transformers.models.internvl.modeling_internvl.InternVLVisionModelOutputWithPooling` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InternVLConfig](/docs/transformers/v5.0.0rc1/en/model_doc/internvl#transformers.InternVLConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Average of the last layer hidden states of the patch tokens (excluding the *[CLS]* token) if
  *config.use_mean_pooling* is set to True. If set to False, then the final hidden state of the *[CLS]* token
  will be returned.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [InternVLVisionModel](/docs/transformers/v5.0.0rc1/en/model_doc/internvl#transformers.InternVLVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([InternVLVisionConfig](/docs/transformers/v5.0.0rc1/en/model_doc/internvl#transformers.InternVLVisionConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.0.0rc1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.internvl.modeling_internvl.InternVLVisionModelOutputWithPooling` or `tuple(torch.FloatTensor)``

A `transformers.models.internvl.modeling_internvl.InternVLVisionModelOutputWithPooling` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InternVLConfig](/docs/transformers/v5.0.0rc1/en/model_doc/internvl#transformers.InternVLConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Average of the last layer hidden states of the patch tokens (excluding the *[CLS]* token) if
  *config.use_mean_pooling* is set to True. If set to False, then the final hidden state of the *[CLS]* token
  will be returned.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## InternVLModel[[transformers.InternVLModel]]

#### transformers.InternVLModel[[transformers.InternVLModel]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/internvl/modeling_internvl.py#L531)

The InternVL model which consists of a vision backbone and a language model, without a language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v5.0.0rc1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.InternVLModel.forwardhttps://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/internvl/modeling_internvl.py#L628[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "vision_feature_layer", "val": ": typing.Union[int, list[int], NoneType] = None"}, {"name": "vision_feature_select_strategy", "val": ": typing.Optional[str] = None"}, {"name": "cache_position", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.0.0rc1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.0.0rc1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/v5.0.0rc1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [GotOcr2ImageProcessor](/docs/transformers/v5.0.0rc1/en/model_doc/got_ocr2#transformers.GotOcr2ImageProcessor). See [GotOcr2ImageProcessor.__call__()](/docs/transformers/v5.0.0rc1/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([InternVLProcessor](/docs/transformers/v5.0.0rc1/en/model_doc/internvl#transformers.InternVLProcessor) uses
  [GotOcr2ImageProcessor](/docs/transformers/v5.0.0rc1/en/model_doc/got_ocr2#transformers.GotOcr2ImageProcessor) for processing images).
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v5.0.0rc1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.0.0rc1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **vision_feature_layer** (`Union[int, list[int], NoneType]`) --
  The index of the layer to select the vision feature. If multiple indices are provided,
  the vision feature of the corresponding indices will be concatenated to form the
  vision features.
- **vision_feature_select_strategy** (`str`, *optional*) --
  The feature selection strategy used to select the vision feature from the vision backbone.
  Can be one of `"default"` or `"full"`.
- **cache_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) --
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.0`transformers.models.internvl.modeling_internvl.InternVLModelOutputWithPast` or `tuple(torch.FloatTensor)`A `transformers.models.internvl.modeling_internvl.InternVLModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InternVLConfig](/docs/transformers/v5.0.0rc1/en/model_doc/internvl#transformers.InternVLConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the model.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.0.0rc1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
The [InternVLModel](/docs/transformers/v5.0.0rc1/en/model_doc/internvl#transformers.InternVLModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([InternVLConfig](/docs/transformers/v5.0.0rc1/en/model_doc/internvl#transformers.InternVLConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.0.0rc1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.internvl.modeling_internvl.InternVLModelOutputWithPast` or `tuple(torch.FloatTensor)``

A `transformers.models.internvl.modeling_internvl.InternVLModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InternVLConfig](/docs/transformers/v5.0.0rc1/en/model_doc/internvl#transformers.InternVLConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the model.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.0.0rc1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.

## InternVLForConditionalGeneration[[transformers.InternVLForConditionalGeneration]]

#### transformers.InternVLForConditionalGeneration[[transformers.InternVLForConditionalGeneration]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/internvl/modeling_internvl.py#L758)

The INTERNVL model which consists of a vision backbone and a language model.

This model inherits from [PreTrainedModel](/docs/transformers/v5.0.0rc1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.InternVLForConditionalGeneration.forwardhttps://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/internvl/modeling_internvl.py#L796[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "vision_feature_layer", "val": ": typing.Union[int, list[int], NoneType] = None"}, {"name": "vision_feature_select_strategy", "val": ": typing.Optional[str] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "cache_position", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "logits_to_keep", "val": ": typing.Union[int, torch.Tensor] = 0"}, {"name": "image_sizes", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.0.0rc1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.0.0rc1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/v5.0.0rc1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [GotOcr2ImageProcessor](/docs/transformers/v5.0.0rc1/en/model_doc/got_ocr2#transformers.GotOcr2ImageProcessor). See [GotOcr2ImageProcessor.__call__()](/docs/transformers/v5.0.0rc1/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([InternVLProcessor](/docs/transformers/v5.0.0rc1/en/model_doc/internvl#transformers.InternVLProcessor) uses
  [GotOcr2ImageProcessor](/docs/transformers/v5.0.0rc1/en/model_doc/got_ocr2#transformers.GotOcr2ImageProcessor) for processing images).
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v5.0.0rc1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.0.0rc1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **vision_feature_layer** (`Union[int, list[int], NoneType]`) --
  The index of the layer to select the vision feature. If multiple indices are provided,
  the vision feature of the corresponding indices will be concatenated to form the
  vision features.
- **vision_feature_select_strategy** (`str`, *optional*) --
  The feature selection strategy used to select the vision feature from the vision backbone.
  Can be one of `"default"` or `"full"`.
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
  config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
- **cache_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) --
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
- **logits_to_keep** (`Union[int, torch.Tensor]`, defaults to `0`) --
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).
- **image_sizes** (`torch.Tensor` of shape `(batch_size, 2)`, *optional*) --
  The sizes of the images in the batch, being (height, width) for each image.0`transformers.models.internvl.modeling_internvl.InternVLCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`A `transformers.models.internvl.modeling_internvl.InternVLCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InternVLConfig](/docs/transformers/v5.0.0rc1/en/model_doc/internvl#transformers.InternVLConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.0.0rc1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
The [InternVLForConditionalGeneration](/docs/transformers/v5.0.0rc1/en/model_doc/internvl#transformers.InternVLForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> import torch
>>> from transformers import AutoProcessor, AutoModelForImageTextToText

>>> torch_device = "cuda"
>>> processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-1B-hf")
>>> model = AutoModelForImageTextToText.from_pretrained(
...     "OpenGVLab/InternVL3-1B-hf", dtype=torch.bfloat16, device_map=torch_device
... )

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {
...                 "type": "image",
...                 "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
...             },
...             {
...                 "type": "image",
...                 "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
...             },
...             {"type": "text", "text": "These images depict two different landmarks. Can you identify them?"},
...         ],
...     },
... ]

>>> inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(torch_device)
>>> generate_ids = model.generate(**inputs, max_new_tokens=200)
>>> print(processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
The images depict the Statue of Liberty and the Golden Gate Bridge.
```

**Parameters:**

config ([InternVLConfig](/docs/transformers/v5.0.0rc1/en/model_doc/internvl#transformers.InternVLConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.0.0rc1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.internvl.modeling_internvl.InternVLCausalLMOutputWithPast` or `tuple(torch.FloatTensor)``

A `transformers.models.internvl.modeling_internvl.InternVLCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InternVLConfig](/docs/transformers/v5.0.0rc1/en/model_doc/internvl#transformers.InternVLConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.0.0rc1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.

## InternVLProcessor[[transformers.InternVLProcessor]]

#### transformers.InternVLProcessor[[transformers.InternVLProcessor]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/internvl/processing_internvl.py#L42)

Constructs a InternVL processor which wraps a [AutoImageProcessor](/docs/transformers/v5.0.0rc1/en/model_doc/auto#transformers.AutoImageProcessor) and
`PretrainedTokenizerFast` tokenizer into a single processor that inherits both the image processor and
tokenizer functionalities. See the `__call__()` and [decode()](/docs/transformers/v5.0.0rc1/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

**Parameters:**

image_processor ([AutoImageProcessor](/docs/transformers/v5.0.0rc1/en/model_doc/auto#transformers.AutoImageProcessor), *optional*) : The image processor is a required input.

tokenizer ([`PreTrainedTokenizer`, `PreTrainedTokenizerFast`], *optional*) : The tokenizer is a required input.

video_processor ([AutoVideoProcessor](/docs/transformers/v5.0.0rc1/en/model_doc/auto#transformers.AutoVideoProcessor), *optional*) : The video processor is a required input.

image_seq_length (`int`, *optional*, defaults to 256) : The number of image token to use per image patch. it should be set so that: image_seq_length = (config.image_size // config.patch_size) ** 2 * (config.scale_factor**2)

chat_template (`str`, *optional*) : A Jinja template which will be used to convert lists of messages in a chat into a tokenizable string.

## InternVLVideoProcessor[[transformers.InternVLVideoProcessor]]

#### transformers.InternVLVideoProcessor[[transformers.InternVLVideoProcessor]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/internvl/video_processing_internvl.py#L34)

sample_framestransformers.InternVLVideoProcessor.sample_frameshttps://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/internvl/video_processing_internvl.py#L50[{"name": "metadata", "val": ": VideoMetadata"}, {"name": "num_frames", "val": ": typing.Optional[int] = None"}, {"name": "fps", "val": ": typing.Union[int, float, NoneType] = None"}, {"name": "initial_shift", "val": ": typing.Union[bool, float, int, NoneType] = None"}, {"name": "**kwargs", "val": ""}]- **metadata** (`VideoMetadata`) --
  Metadata of the video containing information about total duration, fps and total number of frames.
- **num_frames** (`int`, *optional*) --
  Maximum number of frames to sample. Defaults to `self.num_frames`.
- **fps** (`int` or `float`, *optional*) --
  Target frames to sample per second. Defaults to `self.fps`.
- **initial_shift** (`bool`, `float` or `int`, defaults to `self.initial_shift`) --
  The initial shift to apply when sampling frames. If `True`, the shift is set so that frames are sampled from the middle of the video.0np.ndarrayIndices to sample video frames.

Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames.
If `fps` is passed along with metadata, `fps` frames per second are sampled uniformty. Arguments `num_frames`
and `fps` are mutually exclusive.

**Parameters:**

metadata (`VideoMetadata`) : Metadata of the video containing information about total duration, fps and total number of frames.

num_frames (`int`, *optional*) : Maximum number of frames to sample. Defaults to `self.num_frames`.

fps (`int` or `float`, *optional*) : Target frames to sample per second. Defaults to `self.fps`.

initial_shift (`bool`, `float` or `int`, defaults to `self.initial_shift`) : The initial shift to apply when sampling frames. If `True`, the shift is set so that frames are sampled from the middle of the video.

**Returns:**

`np.ndarray`

Indices to sample video frames.

