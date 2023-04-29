import torch
from torch import nn

from GPTModels import GPT2HybridLMHeadModel
from vocab.special_vocab import segment2id


def gen_causal_mask(input_size, dim_k, full_attention=False):
    """
    Generates a causal mask of size (input_size, dim_k) for linformer
    Else, it generates (input_size, input_size) for full attention
    """
    if full_attention:
        return (torch.triu(torch.ones(input_size, input_size)) == 1).transpose(0, 1)
    return (torch.triu(torch.ones(dim_k, input_size)) == 1).transpose(0, 1)


class FineUniGPT(nn.Module):

    def __init__(self, gpt_model, max_num=200, max_length=1024, n_dim=768, init_range=None,
                 copy_mode=False):
        super().__init__()
        self.gpt_model = gpt_model
        self.copy_mode = copy_mode
        assert init_range is not None
        # tokens, segments, sources, source_index, token_types, pos_fw, pos_bw, word_aligns, word_aligns_fw, word_aligns_bw
        self.segment_embedding = nn.Embedding(max_num, n_dim)
        self.source_embedding = nn.Embedding(max_num, n_dim)
        self.source_index_embedding = nn.Embedding(max_num, n_dim)
        self.token_type_embedding = nn.Embedding(max_num, n_dim)
        self.pos_fw_embedding = nn.Embedding(max_length, n_dim)
        self.pos_bw_embedding = nn.Embedding(max_length, n_dim)
        self.word_aligns_embedding = nn.Embedding(max_length, n_dim)
        self.word_aligns_fw_embedding = nn.Embedding(max_length, n_dim)
        self.word_aligns_bw_embedding = nn.Embedding(max_length, n_dim)

        self.segment_embedding.weight.data.normal_(mean=0.0, std=init_range)
        self.source_embedding.weight.data.normal_(mean=0.0, std=init_range)
        self.source_index_embedding.weight.data.normal_(mean=0.0, std=init_range)
        self.token_type_embedding.weight.data.normal_(mean=0.0, std=init_range)
        self.pos_fw_embedding.weight.data.normal_(mean=0.0, std=init_range)
        self.pos_bw_embedding.weight.data.normal_(mean=0.0, std=init_range)
        self.word_aligns_embedding.weight.data.normal_(mean=0.0, std=init_range)
        self.word_aligns_fw_embedding.weight.data.normal_(mean=0.0, std=init_range)
        self.word_aligns_bw_embedding.weight.data.normal_(mean=0.0, std=init_range)

    def forward(self, input_ids, labels, attention_mask, segments, sources, source_index, token_types, pos_fw,
                pos_bw, word_aligns, word_aligns_fw, word_aligns_bw, past_key_values=None, teach_force_mode=False,
                teach_force_rate=0.5):
        inputs_embeds = self.gpt_model.transformer.wte(input_ids)
        non_input_embeddings = self.segment_embedding(segments)
        non_input_embeddings += self.source_embedding(sources)
        non_input_embeddings += self.source_index_embedding(source_index)
        non_input_embeddings += self.token_type_embedding(token_types)
        non_input_embeddings += self.pos_fw_embedding(pos_fw)
        non_input_embeddings += self.pos_bw_embedding(pos_bw)
        non_input_embeddings += self.word_aligns_embedding(word_aligns)
        non_input_embeddings += self.word_aligns_fw_embedding(word_aligns_fw)
        non_input_embeddings += self.word_aligns_bw_embedding(word_aligns_bw)
        is_generative_knowledge_mask = segments == segment2id["[gs_csk_knowledge]"]
        # print(is_generative_knowledge_mask)

        inputs_embeds = inputs_embeds + non_input_embeddings

        if teach_force_mode:

            with torch.no_grad():
                if isinstance(self.gpt_model, GPT2HybridLMHeadModel):
                    eval_outputs = self.gpt_model(inputs_embeds=inputs_embeds, labels=labels,
                                                attention_mask=attention_mask,
                                             output_attentions=self.copy_mode,
                                             output_hidden_states=self.copy_mode,
                                             past_key_values=past_key_values,
                                             use_cache=True, lm2_mask=is_generative_knowledge_mask,
                                             )
                else:
                    eval_outputs = self.gpt_model(inputs_embeds=inputs_embeds, labels=labels,
                                             output_attentions=self.copy_mode,
                                                  attention_mask=attention_mask,
                                             output_hidden_states=self.copy_mode,
                                             past_key_values=past_key_values,
                                             use_cache=True,
                                             )

                teach_force_labels = eval_outputs.logits.topk(1).indices.squeeze(-1)
                teach_force_labels = torch.cat([input_ids[:, 0:1], teach_force_labels[:, :-1]], dim=-1)
                # teach_force_labels = torch.cat([input_ids[:, 0:1], teach_force_labels[:, -1]), -1)
                is_valid_mask = torch.rand_like(teach_force_labels.to(torch.float)) < teach_force_rate
                generative_mask = is_generative_knowledge_mask & is_valid_mask
                teach_force_labels = torch.where(generative_mask, teach_force_labels, input_ids)
                inputs_embeds = self.gpt_model.transformer.wte(teach_force_labels)
                inputs_embeds += non_input_embeddings

            if isinstance(self.gpt_model, GPT2HybridLMHeadModel):
                outputs = self.gpt_model(inputs_embeds=inputs_embeds, labels=labels,
                                         output_attentions=self.copy_mode,
                                         attention_mask=attention_mask,
                                         output_hidden_states=self.copy_mode,
                                         past_key_values=past_key_values,
                                         use_cache=True, lm2_mask=is_generative_knowledge_mask,
                                         )
            else:
                outputs = self.gpt_model(inputs_embeds=inputs_embeds, labels=labels,
                                         output_attentions=self.copy_mode,
                                         attention_mask=attention_mask,
                                         output_hidden_states=self.copy_mode,
                                         past_key_values=past_key_values,
                                         use_cache=True,
                                         )
            return outputs
        else:
            if not self.copy_mode:
                if past_key_values is not None:
                    inputs_embeds = inputs_embeds[:, -1:, :]
                    if labels is not None:
                        labels = labels[:, -1:]
                if isinstance(self.gpt_model, GPT2HybridLMHeadModel):
                    outputs = self.gpt_model(inputs_embeds=inputs_embeds, labels=labels,
                                             output_attentions=self.copy_mode,
                                             attention_mask=attention_mask,
                                             output_hidden_states=self.copy_mode,
                                             past_key_values=past_key_values,
                                             use_cache=True,
                                             lm2_mask=is_generative_knowledge_mask,
                                             )
                else:
                    outputs = self.gpt_model(inputs_embeds=inputs_embeds, labels=labels,
                                             output_attentions=self.copy_mode,
                                             attention_mask=attention_mask,
                                             output_hidden_states=self.copy_mode,
                                             past_key_values=past_key_values,
                                             use_cache=True,
                                             )

                return outputs
            else:
                pass
