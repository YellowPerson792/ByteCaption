# Mistral 3

[Mistral 3](https://mistral.ai/news/mistral-small-3) is a latency optimized model with a lot fewer layers to reduce the time per forward pass. This model adds vision understanding and supports long context lengths of up to 128K tokens without compromising performance.

You can find the original Mistral 3 checkpoints under the [Mistral AI](https://huggingface.co/mistralai/models?search=mistral-small-3) organization.

> [!TIP]
> This model was contributed by [cyrilvallez](https://huggingface.co/cyrilvallez) and [yonigozlan](https://huggingface.co/yonigozlan).
> Click on the Mistral3 models in the right sidebar for more examples of how to apply Mistral3 to different tasks.

The example below demonstrates how to generate text for an image with [Pipeline](/docs/transformers/v5.0.0rc1/en/main_classes/pipelines#transformers.Pipeline) and the [AutoModel](/docs/transformers/v5.0.0rc1/en/model_doc/auto#transformers.AutoModel) class.

```py
import torch
from transformers import pipeline

messages = [
    {"role": "user",
        "content":[
            {"type": "image",
            "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",},
            {"type": "text", "text": "Describe this image."}
        ,]
    ,}
,]

pipeline = pipeline(
    task="image-text-to-text", 
    model="mistralai/Mistral-Small-3.1-24B-Instruct-2503", 
    dtype=torch.bfloat16,
    device=0
)
outputs = pipeline(text=messages, max_new_tokens=50, return_full_text=False)

outputs[0]["generated_text"]
'The image depicts a vibrant and lush garden scene featuring a variety of wildflowers and plants. The central focus is on a large, pinkish-purple flower, likely a Greater Celandine (Chelidonium majus), with a'
```

```py
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from accelerate import Accelerator 

torch_device = Accelerator().device
model_checkpoint = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = AutoModelForImageTextToText.from_pretrained(
    model_checkpoint, 
    device_map=torch_device, 
    dtype=torch.bfloat16
)

messages = [
    {"role": "user",
        "content":[
            {"type": "image",
            "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",},
            {"type": "text", "text": "Describe this image."}
        ,]
    ,}
,]

inputs = processor.apply_chat_template(
    messages, 
    add_generation_prompt=True, 
    tokenize=True, return_dict=True, 
    return_tensors="pt").to(model.device, dtype=torch.bfloat16)

generate_ids = model.generate(**inputs, max_new_tokens=20)
decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

decoded_output
'The image depicts a vibrant and lush garden scene featuring a variety of wildflowers and plants. The central focus is on a large, pinkish-purple flower, likely a Greater Celandine (Chelidonium majus), with a'
```

## Notes

- Mistral 3 supports text-only generation.

```py
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from accelerate import Accelerator

torch_device = Accelerator().device
model_checkpoint = ".mistralai/Mistral-Small-3.1-24B-Instruct-2503"
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, dtype=torch.bfloat16)

SYSTEM_PROMPT = "You are a conversational agent that always answers straight to the point, always end your accurate response with an ASCII drawing of a cat."
user_prompt = "Give me 5 non-formal ways to say 'See you later' in French."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_prompt},
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(0, dtype=torch.float16)
generate_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)
decoded_output = processor.batch_decode(generate_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]

print(decoded_output)
"1. À plus tard!
 2. Salut, à plus!
 3. À toute!
 4. À la prochaine!
 5. Je me casse, à plus!

```

 /\_/\
( o.o )
 > ^ >> from transformers import Mistral3ForConditionalGeneration, Mistral3Config, PixtralVisionConfig, MistralConfig

>>> # Initializing a Pixtral-vision config
>>> vision_config = PixtralVisionConfig()

>>> # Initializing a Mistral config
>>> text_config = MistralConfig()

>>> # Initializing a Mistral3 configuration
>>> configuration = Mistral3Config(vision_config, text_config)

>>> # Initializing a model from the mistral3.1 configuration
>>> model = Mistral3ForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `PixtralVisionConfig`) : The config object or dictionary of the vision backbone.

text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `MistralConfig`) : The config object or dictionary of the text backbone.

image_token_index (`int`, *optional*, defaults to 10) : The image token index to encode the image prompt.

projector_hidden_act (`str`, *optional*, defaults to `"gelu"`) : The activation function used by the multimodal projector.

vision_feature_layer (`Union[int, list[int]]`, *optional*, defaults to -1) : The index of the layer to select the vision feature. If multiple indices are provided, the vision feature of the corresponding indices will be concatenated to form the vision features.

multimodal_projector_bias (`bool`, *optional*, defaults to `False`) : Whether to use bias in the multimodal projector.

spatial_merge_size (`int`, *optional*, defaults to 2) : The downsampling factor for the spatial merge operation.

## MistralCommonBackend[[transformers.MistralCommonBackend]]

#### transformers.MistralCommonBackend[[transformers.MistralCommonBackend]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/tokenization_mistral_common.py#L153)

Class to wrap `mistral-common` tokenizers.

`mistral-common` is the official tokenizer library for Mistral AI models. To use it, you need to install it with:

```bash
pip install transformers[mistral-common]
```

Otherwise the tokenizer falls back to the Transformers implementation of the tokenizer.

For more info on `mistral-common`, see [mistral-common](https://github.com/mistralai/mistral-common).

This class is a wrapper around a `mistral_common.tokens.tokenizers.mistral.MistralTokenizer`.
It provides a Hugging Face compatible interface to tokenize using the official mistral-common tokenizer.

Supports the following methods from the `PreTrainedTokenizerBase` class:

- [get_vocab()](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.MistralCommonBackend.get_vocab): Returns the vocabulary as a dictionary of token to index.
  This is a lossy conversion for Tekkenizer as some decoding errors are collapsed into the same token.
- [encode()](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.MistralCommonBackend.encode): Encode a string to a list of integers.
- [decode()](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.MistralCommonBackend.decode): Decode a list of integers to a string.
- [batch_decode()](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.MistralCommonBackend.batch_decode): Decode a batch of list of integers to a list of strings.
- [convert_tokens_to_ids()](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.MistralCommonBackend.convert_tokens_to_ids): Convert a list of tokens to a list of integers.
- [convert_ids_to_tokens()](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.MistralCommonBackend.convert_ids_to_tokens): Convert a list of integers to a list of tokens.
- [tokenize()](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.MistralCommonBackend.tokenize): Tokenize a string.
- [get_special_tokens_mask()](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.MistralCommonBackend.get_special_tokens_mask): Get the special tokens mask for a list of tokens.
- [prepare_for_model()](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.MistralCommonBackend.prepare_for_model): Prepare a list of inputs for the model.
- [pad()](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.MistralCommonBackend.pad): Pad a list of inputs to the same length.
- [truncate_sequences()](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.MistralCommonBackend.truncate_sequences): Truncate a list of sequences to the same length.
- [apply_chat_template()](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.MistralCommonBackend.apply_chat_template): Apply a chat template to a list of messages.
- `__call__()`: Tokenize a string or a list of strings.
- [from_pretrained()](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.MistralCommonBackend.from_pretrained): Download and cache a pretrained tokenizer from the Hugging Face model hub or local directory.
- [save_pretrained()](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.MistralCommonBackend.save_pretrained): Save a tokenizer to a directory, so it can be reloaded using the `from_pretrained` class method.
- [push_to_hub()](/docs/transformers/v5.0.0rc1/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub): Upload tokenizer to the Hugging Face model hub.

Here are the key differences with the `PreTrainedTokenizerBase` class:

- Pair of sequences are not supported. The signature have been kept for compatibility but all arguments related to pair of sequences are ignored. The return values of pairs are returned as `None`.
- The `is_split_into_words` argument is not supported.
- The `return_token_type_ids` argument is not supported.
- It is not possible to add new tokens to the tokenizer. Also the special tokens are handled differently from Transformers. In `mistral-common`, special tokens are never encoded directly. This means that: `tokenizer.encode("")` will not return the ID of the `` token. Instead, it will return a list of IDs corresponding to the tokenization of the string `""`. For more information, see the [mistral-common documentation](https://mistralai.github.io/mistral-common/usage/tokenizers/#special-tokens).

If you have suggestions to improve this class, please open an issue on the [mistral-common GitHub repository](https://github.com/mistralai/mistral-common/issues) if it is related to the tokenizer or on the [Transformers GitHub repository](https://github.com/huggingface/transformers/issues) if it is related to the Hugging Face interface.

apply_chat_templatetransformers.MistralCommonBackend.apply_chat_templatehttps://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/tokenization_mistral_common.py#L1450[{"name": "conversation", "val": ": list[dict[str, str]] | list[list[dict[str, str]]]"}, {"name": "tools", "val": ": list[dict | collections.abc.Callable] | None = None"}, {"name": "add_generation_prompt", "val": ": bool = False"}, {"name": "continue_final_message", "val": ": bool = False"}, {"name": "tokenize", "val": ": bool = True"}, {"name": "padding", "val": ": bool | str | transformers.utils.generic.PaddingStrategy = False"}, {"name": "truncation", "val": ": bool = False"}, {"name": "max_length", "val": ": int | None = None"}, {"name": "return_tensors", "val": ": str | transformers.utils.generic.TensorType | None = None"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "**kwargs", "val": ""}]- **conversation** (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]) -- A list of dicts
  with "role" and "content" keys, representing the chat history so far.
- **tools** (`List[Union[Dict, Callable]]`, *optional*) --
  A list of tools (callable functions) that will be accessible to the model. If the template does not
  support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
  giving the name, description and argument types for the tool. See our
  [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
  for more information.
- **add_generation_prompt** (`bool`, *optional*) --
  This argument is a no-op for `MistralCommonBackend`. However it cannot be used at the same time as `continue_final_message` to keep the API consistent and
  if any conversation ends with an assistant message, it will raise an error. In such case, use `continue_final_message` instead.
- **continue_final_message** (bool, *optional*) --
  If this is set, the chat will be formatted so that the final
  message in the chat is open-ended, without any EOS tokens. The model will continue this message
  rather than starting a new one. This allows you to "prefill" part of
  the model's response for it. Cannot be used at the same time as `add_generation_prompt`.
- **tokenize** (`bool`, defaults to `True`) --
  Whether to tokenize the output. If `False`, the output will be a string.
- **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v5.0.0rc1/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) --
  Select a strategy to pad the returned sequences (according to the model's padding side and padding
  index) among:

  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
- **truncation** (`bool`, defaults to `False`) --
  Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
- **max_length** (`int`, *optional*) --
  Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If
  not specified, the tokenizer's `max_length` attribute will be used as a default.
- **return_tensors** (`str` or [TensorType](/docs/transformers/v5.0.0rc1/en/internal/file_utils#transformers.TensorType), *optional*) --
  If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable
  values are:
  - `'pt'`: Return PyTorch `torch.Tensor` objects.
- **return_dict** (`bool`, defaults to `False`) --
  Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
  If at least one conversation contains an image, its pixel values will be returned in the `pixel_values` key.
- **kwargs** (additional keyword arguments, *optional*) --
  Not supported by `MistralCommonBackend.apply_chat_template`.
  Will raise an error if used.0`Union[str, list[int], list[str], list[list[int]], BatchEncoding]`A list of token ids representing the tokenized chat so far, including control
tokens. This output is ready to pass to the model, either directly or via methods like `generate()`.

Converts a list of dictionaries with `"role"` and `"content"` keys to a list of token
ids.

**Parameters:**

conversation (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]) : A list of dicts with "role" and "content" keys, representing the chat history so far.

tools (`List[Union[Dict, Callable]]`, *optional*) : A list of tools (callable functions) that will be accessible to the model. If the template does not support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema, giving the name, description and argument types for the tool. See our [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use) for more information.

add_generation_prompt (`bool`, *optional*) : This argument is a no-op for `MistralCommonBackend`. However it cannot be used at the same time as `continue_final_message` to keep the API consistent and if any conversation ends with an assistant message, it will raise an error. In such case, use `continue_final_message` instead.

continue_final_message (bool, *optional*) : If this is set, the chat will be formatted so that the final message in the chat is open-ended, without any EOS tokens. The model will continue this message rather than starting a new one. This allows you to "prefill" part of the model's response for it. Cannot be used at the same time as `add_generation_prompt`.

tokenize (`bool`, defaults to `True`) : Whether to tokenize the output. If `False`, the output will be a string.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/v5.0.0rc1/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) : Select a strategy to pad the returned sequences (according to the model's padding side and padding index) among:  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided). - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).

truncation (`bool`, defaults to `False`) : Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.

max_length (`int`, *optional*) : Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If not specified, the tokenizer's `max_length` attribute will be used as a default.

return_tensors (`str` or [TensorType](/docs/transformers/v5.0.0rc1/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable values are: - `'pt'`: Return PyTorch `torch.Tensor` objects.

return_dict (`bool`, defaults to `False`) : Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`. If at least one conversation contains an image, its pixel values will be returned in the `pixel_values` key.

kwargs (additional keyword arguments, *optional*) : Not supported by `MistralCommonBackend.apply_chat_template`. Will raise an error if used.

**Returns:**

``Union[str, list[int], list[str], list[list[int]], BatchEncoding]``

A list of token ids representing the tokenized chat so far, including control
tokens. This output is ready to pass to the model, either directly or via methods like `generate()`.
#### batch_decode[[transformers.MistralCommonBackend.batch_decode]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/tokenization_mistral_common.py#L527)

Convert a list of lists of token ids into a list of strings by calling decode.

This method is provided for backwards compatibility. The `decode` method now handles batched input natively,
so you can use `decode` directly instead of `batch_decode`.

**Parameters:**

sequences (`Union[list[int], list[list[int]], np.ndarray, torch.Tensor]`) : List of tokenized input ids. Can be obtained using the `__call__` method.

skip_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not to remove special tokens in the decoding.

clean_up_tokenization_spaces (`bool`, *optional*) : Whether or not to clean up the tokenization spaces. If `None`, will default to `self.clean_up_tokenization_spaces`.

kwargs (additional keyword arguments, *optional*) : Not supported by `MistralCommonBackend.batch_decode`. Will raise an error if used.

**Returns:**

``list[str]``

The list of decoded sentences.
#### clean_up_tokenization[[transformers.MistralCommonBackend.clean_up_tokenization]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/tokenization_mistral_common.py#L281)

Clean up a list of simple English tokenization artifacts like spaces before punctuation.
#### convert_ids_to_tokens[[transformers.MistralCommonBackend.convert_ids_to_tokens]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/tokenization_mistral_common.py#L619)

Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
added tokens.

**Parameters:**

ids (`int` or `list[int]`) : The token id (or token ids) to convert to tokens.

skip_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not to remove special tokens in the decoding.

**Returns:**

``str` or `list[str]``

The decoded token(s).
#### convert_tokens_to_ids[[transformers.MistralCommonBackend.convert_tokens_to_ids]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/tokenization_mistral_common.py#L677)

Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
vocabulary.

**Parameters:**

tokens (`str` or `list[str]`) : One or several token(s) to convert to token id(s).

**Returns:**

``int` or `list[int]``

The token id or list of token ids.
#### decode[[transformers.MistralCommonBackend.decode]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/tokenization_mistral_common.py#L481)

Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.

**Parameters:**

token_ids (`Union[int, list[int], list[list[int]], np.ndarray, torch.Tensor]`) : A single sequence or a batch (list of sequences) of tokenized input ids. Can be obtained using the `__call__` method.

skip_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not to remove special tokens in the decoding.

clean_up_tokenization_spaces (`bool`, *optional*) : Whether or not to clean up the tokenization spaces. If `None`, will default to `self.clean_up_tokenization_spaces`.

kwargs (additional keyword arguments, *optional*) : Not supported by `MistralCommonBackend.decode`. Will raise an error if used.

**Returns:**

``Union[str, list[str]]``

The decoded string for a single sequence, or a list of decoded strings for a
batch of sequences.
#### encode[[transformers.MistralCommonBackend.encode]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/tokenization_mistral_common.py#L414)

Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

**Parameters:**

text (`str` or `list[int]`) : The first sequence to be encoded. This can be a string or a list of integers (tokenized string ids).

text_pair (`None`, *optional*) : Not supported by `MistralCommonBackend.encode`. Kept to match `PreTrainedTokenizerBase.encode` signature. 

add_special_tokens (`bool`, *optional*, defaults to `True`) : Whether or not to add special tokens when encoding the sequences. This will use the underlying `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens automatically. When Tokenizer is loading with `finetuning` mode it adds both `bos` and `eos`. Else, for "test" mode it only adds `bos`.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/v5.0.0rc1/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) : Activates and controls padding. Accepts the following values:  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence is provided). - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).

truncation (`bool`, `str` or [TruncationStrategy](/docs/transformers/v5.0.0rc1/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) : Activates and controls truncation. Accepts the following values:  - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).

max_length (`int`, *optional*) : Controls the maximum length to use by one of the truncation/padding parameters.  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.

stride (`int`, *optional*, defaults to 0) : If set to a number along with `max_length`, the overflowing tokens returned when `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence returned to provide some overlap between truncated and overflowing sequences. The value of this argument defines the number of overlapping tokens.

pad_to_multiple_of (`int`, *optional*) : If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).

padding_side (`str`, *optional*) : The side on which the model should have padding applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

return_tensors (`str` or [TensorType](/docs/transformers/v5.0.0rc1/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors instead of list of python integers. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. 

- ****kwargs** : Not supported by `MistralCommonBackend.encode`. Will raise an error if used.

**Returns:**

``list[int]`, `torch.Tensor``

The tokenized ids of the text.
#### from_pretrained[[transformers.MistralCommonBackend.from_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/tokenization_mistral_common.py#L1779)

Instantiate a `MistralCommonBackend` from a predefined
tokenizer.

**Parameters:**

pretrained_model_name_or_path (`str` or `os.PathLike`) : Can be either:  - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co. - A path to a *directory* containing the tokenizer config, for instance saved using the `MistralCommonBackend.tokenization_mistral_common.save_pretrained` method, e.g., `./my_model_directory/`.

mode (`Union[str, ValidationMode]`, *optional*, defaults to `ValidationMode.test`) : Validation mode for the `MistralTokenizer` tokenizer. Possible values are: - `"finetuning"` or `ValidationMode.finetuning`: The finetuning mode. - `"test"` or `ValidationMode.test`: The test mode. It changes how the tokenizer validates the input and prepare the request to the model.

cache_dir (`str` or `os.PathLike`, *optional*) : Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the standard cache should not be used.

force_download (`bool`, *optional*, defaults to `False`) : Whether or not to force the (re-)download the vocabulary files and override the cached versions if they exist.

token (`str` or *bool*, *optional*) : The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated when running `hf auth login` (stored in `~/.huggingface`).

local_files_only (`bool`, *optional*, defaults to `False`) : Whether or not to only rely on local files and not to attempt to download any files.

revision (`str`, *optional*, defaults to `"main"`) : The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.

max_length (`int`, *optional*) : Controls the maximum length to use by one of the truncation/padding parameters.  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.

padding_side (`str`, *optional*, defaults to `"left"`) : The side on which the model should have padding applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

truncation_side (`str`, *optional*, defaults to `"right"`) : The side on which the model should have truncation applied. Should be selected between ['right', 'left'].

model_input_names (`List[string]`, *optional*) : The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or `"attention_mask"`). Default value is picked from the class attribute of the same name.

clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`) : Whether or not the model should cleanup the spaces that were added when splitting the input text during the tokenization process.

kwargs (additional keyword arguments, *optional*) : Not supported by `MistralCommonBackend.from_pretrained`. Will raise an error if used.
#### get_special_tokens_mask[[transformers.MistralCommonBackend.get_special_tokens_mask]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/tokenization_mistral_common.py#L833)

Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

**Parameters:**

token_ids_0 (`list[int]`) : List of ids of the sequence.

token_ids_1 (`list[int]`, *optional*) : Not supported by `MistralCommonBackend`. Kept to match the interface of `PreTrainedTokenizerBase`.

already_has_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not the token list is already formatted with special tokens for the model.

**Returns:**

`A list of integers in the range [0, 1]`

1 for a special token, 0 for a sequence token.
#### get_vocab[[transformers.MistralCommonBackend.get_vocab]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/tokenization_mistral_common.py#L388)

Returns the vocabulary as a dictionary of token to index.

This is a lossy conversion. There may be multiple token ids that decode to the same
string due to partial UTF-8 byte sequences being converted to �.

**Returns:**

``Dict[str, int]``

The vocabulary.
#### pad[[transformers.MistralCommonBackend.pad]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/tokenization_mistral_common.py#L1213)

Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
in the batch.

Padding side (left/right) padding token ids are defined at the tokenizer level (with `self.padding_side`,
`self.pad_token_id`).

If the `encoded_inputs` passed are dictionary of numpy arrays, PyTorch tensors, the
result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
PyTorch tensors, you will lose the specific device of your tensors however.

**Parameters:**

encoded_inputs ([BatchEncoding](/docs/transformers/v5.0.0rc1/en/main_classes/tokenizer#transformers.BatchEncoding), list of [BatchEncoding](/docs/transformers/v5.0.0rc1/en/main_classes/tokenizer#transformers.BatchEncoding), `Dict[str, list[int]]`, `Dict[str, list[list[int]]` or `List[Dict[str, list[int]]]`) : Tokenized inputs. Can represent one input ([BatchEncoding](/docs/transformers/v5.0.0rc1/en/main_classes/tokenizer#transformers.BatchEncoding) or `Dict[str, list[int]]`) or a batch of tokenized inputs (list of [BatchEncoding](/docs/transformers/v5.0.0rc1/en/main_classes/tokenizer#transformers.BatchEncoding), *Dict[str, list[list[int]]]* or *List[Dict[str, list[int]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader collate function.  Instead of `list[int]` you can have tensors (numpy arrays, PyTorch tensors), see the note above for the return type.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/v5.0.0rc1/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `True`) : Select a strategy to pad the returned sequences (according to the model's padding side and padding index) among:  - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single sequence if provided). - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).

max_length (`int`, *optional*) : Maximum length of the returned list and optionally padding length (see above).

pad_to_multiple_of (`int`, *optional*) : If set will pad the sequence to a multiple of the provided value.  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).

padding_side (`str`, *optional*) : The side on which the model should have padding applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

return_attention_mask (`bool`, *optional*) : Whether to return the attention mask. If left to the default, will return the attention mask according to the specific tokenizer's default, defined by the `return_outputs` attribute.  [What are attention masks?](../glossary#attention-mask)

return_tensors (`str` or [TensorType](/docs/transformers/v5.0.0rc1/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors instead of list of python integers. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return Numpy `np.ndarray` objects.

verbose (`bool`, *optional*, defaults to `True`) : Whether or not to print more information and warnings.
#### prepare_for_model[[transformers.MistralCommonBackend.prepare_for_model]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/tokenization_mistral_common.py#L927)

Prepares a sequence of input id so that it can be used by the model. It
adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
manages a moving window (with user defined stride) for overflowing tokens.

**Parameters:**

ids (`list[int]`) : Tokenized input ids of the first sequence.

pair_ids (`None`, *optional*) : Not supported by `MistralCommonBackend`. Kept to match the interface of `PreTrainedTokenizerBase`. 

add_special_tokens (`bool`, *optional*, defaults to `True`) : Whether or not to add special tokens when encoding the sequences. This will use the underlying `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens automatically. When Tokenizer is loading with `finetuning` mode it adds both `bos` and `eos`. Else, for "test" mode it only adds `bos`.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/v5.0.0rc1/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) : Activates and controls padding. Accepts the following values:  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence is provided). - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).

truncation (`bool`, `str` or [TruncationStrategy](/docs/transformers/v5.0.0rc1/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) : Activates and controls truncation. Accepts the following values:  - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).

max_length (`int`, *optional*) : Controls the maximum length to use by one of the truncation/padding parameters.  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.

stride (`int`, *optional*, defaults to 0) : If set to a number along with `max_length`, the overflowing tokens returned when `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence returned to provide some overlap between truncated and overflowing sequences. The value of this argument defines the number of overlapping tokens.

pad_to_multiple_of (`int`, *optional*) : If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).

padding_side (`str`, *optional*) : The side on which the model should have padding applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

return_tensors (`str` or [TensorType](/docs/transformers/v5.0.0rc1/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors instead of list of python integers. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. 

return_attention_mask (`bool`, *optional*) : Whether to return the attention mask. If left to the default, will return the attention mask according to the specific tokenizer's default, defined by the `return_outputs` attribute.  [What are attention masks?](../glossary#attention-mask)

return_overflowing_tokens (`bool`, *optional*, defaults to `False`) : Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead of returning overflowing tokens.

return_special_tokens_mask (`bool`, *optional*, defaults to `False`) : Whether or not to return special tokens mask information.

return_length  (`bool`, *optional*, defaults to `False`) : Whether or not to return the lengths of the encoded inputs.

verbose (`bool`, *optional*, defaults to `True`) : Whether or not to print more information and warnings.

- ****kwargs** : passed to the `self.tokenize()` method

**Returns:**

`[BatchEncoding](/docs/transformers/v5.0.0rc1/en/main_classes/tokenizer#transformers.BatchEncoding)`

A [BatchEncoding](/docs/transformers/v5.0.0rc1/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

- **input_ids** -- List of token ids to be fed to a model.

  [What are input IDs?](../glossary#input-ids)

- **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)

- **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
- **length** -- The length of the inputs (when `return_length=True`)
#### save_pretrained[[transformers.MistralCommonBackend.save_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/tokenization_mistral_common.py#L1911)

Save the full tokenizer state.

This method make sure the full tokenizer can then be re-loaded using the
`~MistralCommonBackend.tokenization_mistral_common.from_pretrained` class method.

**Parameters:**

save_directory (`str` or `os.PathLike`) : The path to a directory where the tokenizer will be saved.

push_to_hub (`bool`, *optional*, defaults to `False`) : Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).

token (`str` or *bool*, *optional*, defaults to `None`) : The token to use to push to the model hub. If `True`, will use the token in the `HF_TOKEN` environment variable.

commit_message (`str`, *optional*) : The commit message to use when pushing to the hub.

repo_id (`str`, *optional*) : The name of the repository to which push to the Hub.

private (`bool`, *optional*) : Whether the model repository is private or not.

kwargs (`Dict[str, Any]`, *optional*) : Not supported by `MistralCommonBackend.save_pretrained`. Will raise an error if used.

**Returns:**

`A tuple of `str``

The files saved.
#### tokenize[[transformers.MistralCommonBackend.tokenize]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/tokenization_mistral_common.py#L711)

Converts a string into a sequence of tokens, using the tokenizer.

Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies.

**Parameters:**

text (`str`) : The sequence to be encoded.

- ****kwargs** (additional keyword arguments) : Not supported by `MistralCommonBackend.tokenize`. Will raise an error if used.

**Returns:**

``list[str]``

The list of tokens.
#### truncate_sequences[[transformers.MistralCommonBackend.truncate_sequences]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/tokenization_mistral_common.py#L1375)

Truncates a sequence pair in-place following the strategy.

**Parameters:**

ids (`list[int]`) : Tokenized input ids. Can be obtained from a string by chaining the `tokenize` and `convert_tokens_to_ids` methods.

pair_ids (`None`, *optional*) : Not supported by `MistralCommonBackend`. Kept to match the signature of `PreTrainedTokenizerBase.truncate_sequences`.

num_tokens_to_remove (`int`, *optional*, defaults to 0) : Number of tokens to remove using the truncation strategy.

truncation_strategy (`str` or [TruncationStrategy](/docs/transformers/v5.0.0rc1/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `'longest_first'`) : The strategy to follow for truncation. Can be:  - `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).

stride (`int`, *optional*, defaults to 0) : If set to a positive number, the overflowing tokens returned will contain some tokens from the main sequence returned. The value of this argument defines the number of additional tokens.

**Returns:**

``Tuple[list[int], None, list[int]]``

The truncated `ids` and the list of
overflowing tokens. `None` is returned to match Transformers signature.

## Mistral3Model[[transformers.Mistral3Model]]

#### transformers.Mistral3Model[[transformers.Mistral3Model]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/mistral3/modeling_mistral3.py#L197)

The Mistral3 model which consists of a vision backbone and a language model, without a language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v5.0.0rc1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.Mistral3Model.forwardhttps://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/mistral3/modeling_mistral3.py#L283[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "vision_feature_layer", "val": ": typing.Union[int, list[int], NoneType] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "cache_position", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "image_sizes", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.0.0rc1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.0.0rc1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/v5.0.0rc1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [PixtralImageProcessor](/docs/transformers/v5.0.0rc1/en/model_doc/pixtral#transformers.PixtralImageProcessor). See [PixtralImageProcessor.__call__()](/docs/transformers/v5.0.0rc1/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([PixtralProcessor](/docs/transformers/v5.0.0rc1/en/model_doc/pixtral#transformers.PixtralProcessor) uses
  [PixtralImageProcessor](/docs/transformers/v5.0.0rc1/en/model_doc/pixtral#transformers.PixtralImageProcessor) for processing images).
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
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/v5.0.0rc1/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **cache_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) --
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
- **image_sizes** (`torch.Tensor` of shape `(batch_size, 2)`, *optional*) --
  The sizes of the images in the batch, being (height, width) for each image.0`transformers.models.mistral3.modeling_mistral3.Mistral3ModelOutputWithPast` or `tuple(torch.FloatTensor)`A `transformers.models.mistral3.modeling_mistral3.Mistral3ModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Mistral3Config](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.Mistral3Config)) and inputs.

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
The [Mistral3Model](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.Mistral3Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([Mistral3Config](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.Mistral3Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.0.0rc1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.mistral3.modeling_mistral3.Mistral3ModelOutputWithPast` or `tuple(torch.FloatTensor)``

A `transformers.models.mistral3.modeling_mistral3.Mistral3ModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Mistral3Config](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.Mistral3Config)) and inputs.

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
#### get_image_features[[transformers.Mistral3Model.get_image_features]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/mistral3/modeling_mistral3.py#L216)

Obtains image last hidden states from the vision tower and apply multimodal projection.

**Parameters:**

pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`) : The tensors corresponding to the input images.

vision_feature_layer (`Union[int, list[int]]`, *optional*) : The index of the layer to select the vision feature. If multiple indices are provided, the vision feature of the corresponding indices will be concatenated to form the vision features.

image_sizes (`torch.Tensor`, *optional*) : Tensor containing the image sizes as returned by the processor.

**Returns:**

`image_features (`torch.Tensor`)`

Image feature tensor of shape `(num_images, image_length, embed_dim)`).
#### get_placeholder_mask[[transformers.Mistral3Model.get_placeholder_mask]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/mistral3/modeling_mistral3.py#L259)

Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
equal to the length of multimodal features. If the lengths are different, an error is raised.

## Mistral3ForConditionalGeneration[[transformers.Mistral3ForConditionalGeneration]]

#### transformers.Mistral3ForConditionalGeneration[[transformers.Mistral3ForConditionalGeneration]]

[Source](https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/mistral3/modeling_mistral3.py#L356)

The MISTRAL3 model which consists of a vision backbone and a language model.

This model inherits from [PreTrainedModel](/docs/transformers/v5.0.0rc1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.Mistral3ForConditionalGeneration.forwardhttps://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/mistral3/modeling_mistral3.py#L394[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "cache_position", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "logits_to_keep", "val": ": typing.Union[int, torch.Tensor] = 0"}, {"name": "image_sizes", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.0.0rc1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.0.0rc1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/v5.0.0rc1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [PixtralImageProcessor](/docs/transformers/v5.0.0rc1/en/model_doc/pixtral#transformers.PixtralImageProcessor). See [PixtralImageProcessor.__call__()](/docs/transformers/v5.0.0rc1/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([PixtralProcessor](/docs/transformers/v5.0.0rc1/en/model_doc/pixtral#transformers.PixtralProcessor) uses
  [PixtralImageProcessor](/docs/transformers/v5.0.0rc1/en/model_doc/pixtral#transformers.PixtralImageProcessor) for processing images).
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
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
  config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/v5.0.0rc1/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
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
  The sizes of the images in the batch, being (height, width) for each image.0`transformers.models.mistral3.modeling_mistral3.Mistral3CausalLMOutputWithPast` or `tuple(torch.FloatTensor)`A `transformers.models.mistral3.modeling_mistral3.Mistral3CausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Mistral3Config](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.Mistral3Config)) and inputs.

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
The [Mistral3ForConditionalGeneration](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.Mistral3ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Mistral3ForConditionalGeneration

>>> model = Mistral3ForConditionalGeneration.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503")
>>> processor = AutoProcessor.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503")

>>> prompt = "[INST][IMG]What is the image?[/INST]"
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, text=prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(**inputs, max_new_tokens=15)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"What is the image?The image depicts two cats lying on a pink blanket."
```

**Parameters:**

config ([Mistral3Config](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.Mistral3Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.0.0rc1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.mistral3.modeling_mistral3.Mistral3CausalLMOutputWithPast` or `tuple(torch.FloatTensor)``

A `transformers.models.mistral3.modeling_mistral3.Mistral3CausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Mistral3Config](/docs/transformers/v5.0.0rc1/en/model_doc/mistral3#transformers.Mistral3Config)) and inputs.

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

