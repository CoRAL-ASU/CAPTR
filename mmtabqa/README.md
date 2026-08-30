# Create Caption Datasets
First, bring the MMTabQA dataset in the format we need:
```
python3 create_dataset.py
```

Then, create the captions for this dataset:

```
python3 create_caption_dataset.py --caption_mode="naive"
python3 create_caption_dataset.py --caption_mode="short_with_context"
python3 create_caption_dataset.py --caption_mode="detailed"
```

Thats it. Now you have the captioned datasets in your [../datasets/](../datasets/) directory!
Specifically, MMTabQA is in `../datasets/MMTabQA_Dataset/converted_to_hf_dataset` and the captioned datasets in `../datasets/MMTabQA_Dataset/converted_to_hf_dataset/mmtabqa_captioned`

## Adding Your Own Dataset

CAPTR can work with any tabular reasoning dataset. Convert your data to HuggingFace Dataset format:

```python
{
    "id": "example-1",
    "question": "What is the capital of France?",
    "answer_text": ["Paris"],
    "table_id": "countries/france.csv",
    "table": {
        "section_title": "European Countries",
        "page_title": "Countries of the World",
        "header": ["Country", "Flag", "Capital", "Population"],
        "rows": [
            {
                "type": ["text", "image", "text", "text"],
                "content": ["France", "france_flag.png", "Paris", "67M"]
            },
            ...
        ]
    }
}
```

Then modify the dataset loading in `mmtabqa/load_mmtabqa_utils.py` to include your dataset.
