from transformers import CpmTokenizer

# 这些是要加入到对话中的
from vocab.tokenizer_uni import UniTokenizer

dialog_orders = ['[U1Q0]', '[M1R0]', '[U1Q1]', '[M1R1]', '[U1Q2]', '[M1R2]', '[U1Q3]', '[M1R3]', '[U1Q4]', '[M1R4]',
                 '[U1Q5]', '[M1R5]']
special_token_list = ['[DHistory]', '[DResponse]', ] + dialog_orders
relations = [
    '[csk_knowledge]',
    '#N',
    '#NF',
    '#NT',
    '#NH',
    '#PR',
    '#NR',
    'RelatedTo',
    'Synonym',
    'Causes',
    'MotivatedByGoal',
    'HasContext',
    'HasSubevent',
    'CapableOf',
    'NotDesires',
    'Desires',
    'AtLocation',
    'CausesDesire',
    'IsA',
    'MadeOf',
    'UsedFor',
    'HasFirstSubevent',
    'HasA',
    'HasProperty',
    'PartOf',
    'SymbolOf',
    'DerivedFrom',
    'EtymologicallyDerivedFrom',
    'EtymologicallyRelatedTo',
    'FormOf',
    'Antonym',
    'SimilarTo',
    'DistinctFrom', ]
inv_relations = ['Inv'+x for x in relations]
special_token_list = special_token_list + relations + inv_relations
other_sp = ['[TE_SEP]','[CSK_SEP]','[gs_csk_knowledge]', '[GSI]', '[KEY]', '[VALUE]', '[Commentary]']
special_token_list = special_token_list + other_sp

csk_relation_set = set(relations + inv_relations)

# SegmentIds
segment_ids = ['[PAD]', '[DHistory]', '[DResponse]', '[GSKnowledge]', '[FineStart]', '[FineEnd]',
               '[csk_knowledge]', '[infobox_knowledge]', '[text_knowledge]', '[prototypes_knowledge]', '[ms_knowledge]',
               '[gs_csk_knowledge]', '[gs_infobox_knowledge]', '[gs_text_knowledge]', '[gs_prototypes_knowledge]',
               '[gs_ms_knowledge]', '[Prompting]'
               ]
segment2id = dict()
id2segment = dict()
for idx, segment in enumerate(segment_ids):
    segment2id[segment] = idx
    id2segment[idx] = segment

# SourceIDs
source_ids = ['[PAD]', '[Task]', '[ConceptNet]', '[WikipediaInfobox]', '[WikipediaText]',
              '[User]', '[Machine]', '[GeneratedKnowledge]', '[Action]'
              ]
source2id = dict()
id2source = dict()
for idx, source in enumerate(source_ids):
    source2id[source] = idx
    id2source[idx] = source

# TokenTypes
token_types = ['[PAD]', 'entity', 'relation', 'tips', 'action', 'sp', 'text', 'word', 'word_entity', 'copied',
               'head_entity','tail_entity','[TE_SEP]','[CSK_SEP]','gs_csk_knowledge']
type2id = dict()
id2type = dict()
for idx, item in enumerate(token_types):
    type2id[item] = idx
    id2type[idx] = item

while len(special_token_list) < 200:
    special_token_list.append('[SP_%d]' % len(special_token_list))

sp_stoi = {}
for idx, sp_token in enumerate(special_token_list):
    sp_stoi[sp_token] = idx


def get_tokenizer():
    tokenizer = UniTokenizer(vocab_file="vocab/chinese_vocab.model", additional_special_tokens=special_token_list)
    tokenizer.add_tokens(new_tokens=special_token_list)
    for idx, sp_token in enumerate(special_token_list):
        # 确保一致
        assert tokenizer.convert_tokens_to_ids(sp_token) == tokenizer.vocab_size + idx
    return tokenizer
