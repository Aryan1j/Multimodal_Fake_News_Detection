from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset

from .config import MAX_LENGTH


class FakeNewsTokenizedDataset(Dataset):
  
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = MAX_LENGTH
    ):
     
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize with truncation and no padding (padding done in collate_fn)
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    
    # Find max length in this batch
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    # Prepare lists for batching
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for item in batch:
        seq_len = item['input_ids'].size(0)
        padding_len = max_len - seq_len
        
        # Pad input_ids with 0 (PAD token)
        if padding_len > 0:
            input_ids = torch.cat([
                item['input_ids'],
                torch.zeros(padding_len, dtype=torch.long)
            ])
            attention_mask = torch.cat([
                item['attention_mask'],
                torch.zeros(padding_len, dtype=torch.long)
            ])
        else:
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(item['label'])
    
    return {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list),
        'labels': torch.stack(labels_list)
    }


def create_dataloader(
    texts: List[str],
    labels: List[int],
    tokenizer,
    batch_size: int,
    shuffle: bool = False,
    max_length: int = MAX_LENGTH,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for training or evaluation.
    
    """
    dataset = FakeNewsTokenizedDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
