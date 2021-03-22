import tensorflow as tf
from typing import Dict, Tuple, NamedTuple, Union, Optional, Iterable
from config import Config
from vocabularies import Code2VecVocabs
import abc
from functools import reduce
from enum import Enum


class EstimatorAction(Enum):
    Train = 'train'
    Evaluate = 'evaluate'
    Predict = 'predict'

    @property
    def is_train(self):
        return self is EstimatorAction.Train

    @property
    def is_evaluate(self):
        return self is EstimatorAction.Evaluate

    @property
    def is_predict(self):
        return self is EstimatorAction.Predict

    @property
    def is_evaluate_or_predict(self):
        return self.is_evaluate or self.is_predict


class ReaderInputTensors(NamedTuple):
    """
    Used mostly for convenient-and-clear access to input parts (by their names).
    """
    token_indices: tf.Tensor
    token_valid_mask: tf.Tensor
    token_strings: tf.Tensor
    target_index: Optional[tf.Tensor] = None
    target_string: Optional[tf.Tensor] = None


class ModelInputTensorsFormer(abc.ABC):
    """
    Should be inherited by the model implementation.
    An instance of the inherited class is passed by the model to the reader in order to help the reader
        to construct the input in the form that the model expects to receive it.
    This class also enables conveniently & clearly access input parts by their field names.
        eg: 'tensors.path_indices' instead if 'tensors[1]'.
    This allows the input tensors to be passed as pure tuples along the computation graph, while the
        python functions that construct the graph can easily (and clearly) access tensors.
    """

    @abc.abstractmethod
    def to_model_input_form(self, input_tensors: ReaderInputTensors):
        ...

    @abc.abstractmethod
    def from_model_input_form(self, input_row) -> ReaderInputTensors:
        ...


class PathContextReader:
    def __init__(self,
                 vocabs: Code2VecVocabs,
                 config: Config,
                 model_input_tensors_former: ModelInputTensorsFormer,
                 estimator_action: EstimatorAction,
                 repeat_endlessly: bool = False):
        self.vocabs = vocabs
        self.config = config
        self.model_input_tensors_former = model_input_tensors_former
        self.estimator_action = estimator_action
        self.repeat_endlessly = repeat_endlessly
        self.csv_record_defaults = [[self.vocabs.target_vocab.special_words.OOV]] + \
            ([[self.vocabs.token_vocab.special_words.PAD]] * self.config.MAX_TOKENS)

        # initialize the needed lookup tables (if not already initialized).
        self.create_needed_vocabs_lookup_tables(self.vocabs)

        self._dataset: Optional[tf.data.Dataset] = None

    @ classmethod
    def create_needed_vocabs_lookup_tables(cls, vocabs: Code2VecVocabs):
        vocabs.token_vocab.get_word_to_index_lookup_table()
        vocabs.target_vocab.get_word_to_index_lookup_table()

    @ tf.function
    def process_input_row(self, row_placeholder):
        parts = tf.io.decode_csv(
            row_placeholder, record_defaults=self.csv_record_defaults, field_delim=' ', use_quote_delim=False)
        # Note: we DON'T apply the filter `_filter_input_rows()` here.
        tensors = self._map_raw_dataset_row_to_input_tensors(*parts)

        # make it batched (first batch axis is going to have dimension 1)
        tensors_expanded = ReaderInputTensors(
            **{name: None if tensor is None else tf.expand_dims(tensor, axis=0)
               for name, tensor in tensors._asdict().items()})
        return self.model_input_tensors_former.to_model_input_form(tensors_expanded)

    def process_and_iterate_input_from_data_lines(self, input_data_lines: Iterable) -> Iterable:
        for data_row in input_data_lines:
            processed_row = self.process_input_row(data_row)
            yield processed_row

    def get_dataset(self, input_data_rows: Optional = None) -> tf.data.Dataset:
        if self._dataset is None:
            self._dataset = self._create_dataset_pipeline(input_data_rows)
        return self._dataset

    def _create_dataset_pipeline(self, input_data_rows: Optional = None) -> tf.data.Dataset:
        if input_data_rows is None:
            assert not self.estimator_action.is_predict
            dataset = tf.data.experimental.CsvDataset(
                self.config.data_path(
                    is_evaluating=self.estimator_action.is_evaluate),
                record_defaults=self.csv_record_defaults, field_delim=' ', use_quote_delim=False,
                buffer_size=self.config.CSV_BUFFER_SIZE)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(input_data_rows)
            dataset = dataset.map(
                lambda input_line: tf.io.decode_csv(
                    tf.reshape(tf.cast(input_line, tf.string), ()),
                    record_defaults=self.csv_record_defaults,
                    field_delim=' ', use_quote_delim=False),
                num_parallel_calls=tf.data.AUTOTUNE)

        if self.repeat_endlessly:
            dataset = dataset.repeat()
        if self.estimator_action.is_train:
            if not self.repeat_endlessly and self.config.NUM_TRAIN_EPOCHS > 1:
                dataset = dataset.repeat(self.config.NUM_TRAIN_EPOCHS)
            dataset = dataset.shuffle(
                self.config.SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)

        dataset = dataset.map(self._map_raw_dataset_row_to_expected_model_input_form,
                              num_parallel_calls=tf.data.AUTOTUNE)

        batch_size = self.config.batch_size(
            is_evaluating=self.estimator_action.is_evaluate)
        if self.estimator_action.is_predict:
            dataset = dataset.batch(1)
        else:
            dataset = dataset.filter(self._filter_input_rows)
            dataset = dataset.batch(batch_size)

        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def _filter_input_rows(self, *row_parts) -> tf.bool:
        row_parts = self.model_input_tensors_former.from_model_input_form(
            row_parts)

        # assert all(tensor.shape == (self.config.MAX_CONTEXTS,) for tensor in
        #           {row_parts.path_source_token_indices, row_parts.path_indices,
        #            row_parts.path_target_token_indices, row_parts.context_valid_mask})

        # FIXME: Does "valid" here mean just "no padding" or "neither padding nor OOV"? I assumed just "no padding".
        # any_word_valid_mask_per_context_part = [
        #     tf.not_equal(tf.reduce_max(row_parts.path_source_token_indices, axis=0),
        #                  self.vocabs.token_vocab.word_to_index[self.vocabs.token_vocab.special_words.PAD]),
        #     tf.not_equal(tf.reduce_max(row_parts.path_target_token_indices, axis=0),
        #                  self.vocabs.token_vocab.word_to_index[self.vocabs.token_vocab.special_words.PAD]),
        #     tf.not_equal(tf.reduce_max(row_parts.path_indices, axis=0),
        #                  self.vocabs.path_vocab.word_to_index[self.vocabs.path_vocab.special_words.PAD])]
        # any_contexts_is_valid = reduce(
        #     tf.logical_or, any_word_valid_mask_per_context_part)  # scalar
        tokens_are_valid = tf.not_equal(tf.reduce_max(row_parts.token_indices, axis=0),
                                        self.vocabs.token_vocab.word_to_index[self.vocabs.token_vocab.special_words.PAD])

        if self.estimator_action.is_evaluate:
            cond = tokens_are_valid  # scalar
        else:  # training
            target_is_valid = tf.greater(
                row_parts.target_index, self.vocabs.target_vocab.word_to_index[self.vocabs.target_vocab.special_words.OOV])  # scalar
            cond = tf.logical_and(
                target_is_valid, tokens_are_valid)  # scalar

        return cond  # scalar

    def _map_raw_dataset_row_to_expected_model_input_form(self, *row_parts) -> \
            Tuple[Union[tf.Tensor, Tuple[tf.Tensor, ...], Dict[str, tf.Tensor]], ...]:
        tensors = self._map_raw_dataset_row_to_input_tensors(*row_parts)
        return self.model_input_tensors_former.to_model_input_form(tensors)

    def _map_raw_dataset_row_to_input_tensors(self, *row_parts) -> ReaderInputTensors:
        row_parts = list(row_parts)
        target_str = row_parts[0]
        target_index = self.vocabs.target_vocab.lookup_index(target_str)

        token_strings = tf.stack(
            row_parts[1:(self.config.MAX_TOKENS + 1)], axis=0)

        split_tokens = tf.compat.v1.string_split(
            token_strings, sep=', ', skip_empty=False)

        # dense_split_tokens = tf.sparse_tensor_to_dense(
        #     split_tokens, default_value=self.vocabs.token_vocab.special_words.PAD)
        sparse_split_tokens = tf.sparse.SparseTensor(
            indices=split_tokens.indices, values=split_tokens.values, dense_shape=[self.config.MAX_TOKENS, self.config.MAX_TOKEN_VOCAB_SIZE])

        dense_split_tokens = tf.reshape(
            tf.sparse.to_dense(sp_input=sparse_split_tokens,
                               default_value=self.vocabs.token_vocab.special_words.PAD),
            shape=[self.config.MAX_TOKENS, self.config.MAX_TOKEN_VOCAB_SIZE])

        token_indices = self.vocabs.token_vocab.lookup_index(
            token_strings)  # (max_contexts, )

        token_valid_mask = tf.cast(tf.not_equal(token_indices,
                                                self.vocabs.token_vocab.word_to_index[self.vocabs.token_vocab.special_words.PAD]), dtype=tf.float32)

        return ReaderInputTensors(
            token_indices=token_indices,
            token_strings=token_strings,
            token_valid_mask=token_valid_mask,
            target_index=target_index,
            target_string=target_str,
        )
