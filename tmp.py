def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    is_xlnet: bool = False,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """Loads a data file into a list of `InputFeatures`
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers
    cls_token_at_end = is_xlnet
    cls_token = tokenizer.cls_token
    cls_token_segment_id = 2 if is_xlnet else 0
    sep_token = tokenizer.sep_token
    sep_token_extra = False
    pad_on_left = is_xlnet
    pad_token = tokenizer.pad_token_id
    pad_token_segment_id = tokenizer.pad_token_type_id

    label_map = {label: i for i, label in enumerate(label_list)}

    features: List[InputFeatures] = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        if example.labels is not None:
            for word, label in zip(example.words, example.labels):
                word_tokens = tokenizer.tokenize(word)

                # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    label_ids.extend(
                        [label_map[label]]  # * len(word_tokens)
                        + [PAD_TOKEN_LABEL_ID] * (len(word_tokens) - 1)
                    )

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = tokenizer.num_special_tokens_to_add()
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]

            tokens += [sep_token]
            label_ids += [PAD_TOKEN_LABEL_ID]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [PAD_TOKEN_LABEL_ID]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [PAD_TOKEN_LABEL_ID]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [PAD_TOKEN_LABEL_ID] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids: List[int] = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = (
                    [0 if mask_padding_with_zero else 1] * padding_length
                ) + input_mask
                segment_ids = [pad_token_segment_id] * padding_length + segment_ids
                label_ids = ([PAD_TOKEN_LABEL_ID] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [PAD_TOKEN_LABEL_ID] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

            if "token_type_ids" not in tokenizer.model_input_names:
                segment_ids = None

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids,
                    label_ids=label_ids,
                )
            )

    return features
