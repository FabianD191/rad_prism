# Third-party models and artifacts

The MIT license in this repository covers the original RadPRISM code and the
distributed RadPRISM-trained tensors. It does not replace the licenses or terms
of third-party models, software, datasets, or images. The separately licensed
example chest X-rays are identified below.

## RAD-DINO-MAIRA-2

- Upstream: <https://huggingface.co/microsoft/rad-dino-maira-2>
- License: Microsoft Research License Terms (MSRLA)
- Role: frozen vision backbone
- Distributed here: no

The upstream terms restrict use to non-commercial, non-revenue-generating
research and prohibit redistribution of the model. Users must download the
backbone separately and comply with the upstream terms. Consequently, an
end-to-end RadPRISM system that loads this backbone is subject to those
restrictions even though the original RadPRISM code is MIT-licensed.

`data/radprism_checkpoint/radprism_heads.safetensors` contains RadPRISM concept
tokens, projections, cross-attention, and classification-head parameters. It
does not contain RAD-DINO-MAIRA-2 backbone parameters.

## Qwen3-Embedding-4B

- Upstream: <https://huggingface.co/Qwen/Qwen3-Embedding-4B>
- License: Apache License 2.0
- Role: report, prompt, and retrieval text embeddings
- Distributed here: no model weights

The repository includes embedding outputs in
`data/radprism_checkpoint/retrieval_bank.pt` and
`data/radprism_checkpoint/zero_shot_bank.pt`, but not the Qwen model itself.
Users who download or redistribute Qwen3-Embedding-4B must comply with its
Apache-2.0 terms.

## Python dependencies

The packages listed in the requirements files are installed separately and
remain subject to their respective licenses. They are not relicensed by this
repository.

## CheXpert and CheXlocalize

The optional external evaluation uses CheXpert images/labels and CheXlocalize
ground-truth segmentations. No CheXpert or CheXlocalize data is distributed in
this repository. Users must obtain the datasets separately, accept their
applicable access terms, and follow their citation requirements.

## Bundled example material

The example reports in `report_struct_label/data/`, the dummy identifiers and
embeddings, and the retrieval and zero-shot text databases are synthetic.

The seven example chest X-rays in `data/jpg/` are excluded from the repository's
MIT license and are separately licensed under the Creative Commons
Attribution-NonCommercial 4.0 International license (CC BY-NC 4.0). See
`data/jpg/README.md` for scope and attribution and `data/jpg/LICENSE` for the
complete legal text.
