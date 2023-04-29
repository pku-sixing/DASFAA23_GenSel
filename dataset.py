from torch.utils.data import Dataset
import torch


class FineDataset(Dataset):
    """

    """

    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __getitem__(self, index):
        input_ids = self.input_list[index]['input_ids']
        context_length = self.input_list[index]['context_length']
        full_length = self.input_list[index]['full_length']
        input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        context_length = torch.tensor(context_length, dtype=torch.long)
        full_length = torch.tensor(full_length, dtype=torch.long)
        return input_ids, context_length, full_length

    def __len__(self):
        return len(self.input_list)

class FineUniGPTDataset(Dataset):
    """

    """

    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __getitem__(self, index):
        """
        item = {
            "input_ids": merged_segment[0],
            "segment_ids": merged_segment[1],
            "source_ids": merged_segment[2],
            "source_indices": merged_segment[3],
            "type_ids": merged_segment[4],
            "fw_pos": merged_segment[5],
            "bw_pos": merged_segment[6],
            "word_align": merged_segment[7],
            "word_index_fw": merged_segment[8],
            "word_index_bw": merged_segment[9],
            "full_length": len(merged_segment[0]),
        }
        :param index:
        :return:
        """
        input_ids = self.input_list[index]['input_ids']
        context_length = self.input_list[index]['context_length']
        full_length = self.input_list[index]['full_length']
        segment_ids = self.input_list[index]['segment_ids']
        source_ids = self.input_list[index]['source_ids']
        type_ids = self.input_list[index]['type_ids']
        source_indices = self.input_list[index]['source_indices']
        fw_pos = self.input_list[index]['fw_pos']
        bw_pos = self.input_list[index]['bw_pos']
        word_align = self.input_list[index]['word_align']
        word_index_fw = self.input_list[index]['word_index_fw']
        word_index_bw = self.input_list[index]['word_index_bw']

        #
        # input_ids = input_ids[:self.max_len]
        assert len(input_ids) <= self.max_len
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        context_length = torch.tensor(context_length, dtype=torch.long)
        full_length = torch.tensor(full_length, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        source_ids = torch.tensor(source_ids, dtype=torch.long)
        source_indices = torch.tensor(source_indices, dtype=torch.long)
        type_ids = torch.tensor(type_ids, dtype=torch.long)
        fw_pos = torch.tensor(fw_pos, dtype=torch.long)
        bw_pos = torch.tensor(bw_pos, dtype=torch.long)
        word_align = torch.tensor(word_align, dtype=torch.long)
        word_index_fw = torch.tensor(word_index_fw, dtype=torch.long)
        word_index_bw = torch.tensor(word_index_bw, dtype=torch.long)

        return input_ids, context_length, full_length, segment_ids, source_ids, source_indices, type_ids, fw_pos, bw_pos,\
               word_align, word_index_fw, word_index_bw

    def __len__(self):
        return len(self.input_list)

class CPMDataset(Dataset):
    """

    """

    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __getitem__(self, index):
        input_ids = self.input_list[index]
        input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def __len__(self):
        return len(self.input_list)
