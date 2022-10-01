import tensorflow as tf


class DataLoader():
    """create tensorflow data pipeline.
    This class provides the same action with `Data_Loader` in `pytorch` using 
    `tf.data.Datasets.from_tensor_slices`. It create a `Dataset` whose elements 
    are slices of the given tensors, and also provide shuffle, batch and repeat
    , prefetch operations.
    shuffle:
        Randomly shuffles the elements of this dataset.
        This dataset fills a buffer with `shuffle_buffer_size` elements, then
        randomly samples elements from this buffer, replacing the selected 
        elements with new elements. 
        Note: For perfect shuffling, a buffer size greater than or equal to the 
        full size of the dataset is required. Having a low `shuffle_buffer_size`
        will not just give you inferior shuffling in some cases, it can also
        mess up your whole training. e.g., if your dataset contains 10,000 
        elements (cat & non-cat images) but `shuffle_buffer_size` is set to 
        1,000, then `shuffle` will initially select a random element from only 
        the first 1,000 elements in the buffer. Once an element is selected, 
        its space in the buffer is replaced by the next (i.e., 1,001-st) 
        element, maintaining the 1,000 element buffer. However, an extreme 
        scenario is that the first 1,000 is all cat images, which means the 
        low `shuffle_buffer_size` could result in unexpected result. `shuffle`
        doesn't signal the end of an epoch until the shuffle buffer is empty.
        So a shuffle placed before a repeat will show every element of one
        epoch before moving to the next. But a repeat before shuffle mixes the
        epoch boundaries together. thus we all applied sequence of ```python 
        tf.data.Datasets(dataset).shuffle(2).repeat(2)```
    batch:
        Combines consecutive elements of this dataset into batches.
    repeat:
        Repeats the dataset so each original value is seen `count` times. 
        if we have 100 samples, and get 5 batches (20 of each), then we need to
        feed it to the model, but if the model were trained for 5 epochs, we 
        need to REPEAT 100 samples for 5 times. 
        Note: This operation need large internal memory space. `count` usually 
        is the same with `epochs`. The `repeat` transformation concatenates its 
        arguments without signaling the end of one epoch and the begining of 
        the next epoch. Because of this, a `batch` applied after `repeat` will
        yield batches that straddle epoch boundaries. If you need clear epoch
        seperation, put `batch` before the `repeat`. i.e., ```python tf.data.
        Datasets(dataset).batch(2).repeat(2)```
    prefetch:
        Creates a Dataset that prefetches elements from this dataset.
        Most dataset input pipelines should end with a call to `prefetch`. This
        allows later elements to be prepared while the current element is being
        processed. This often improves latency and throughput, at the cost of
        using additional memory to store prefetched elements.
        Note: It has no concept of examples vs batches. example.prefetch(2) will
        prefetch two elements (2 samples), while example.batch(10).prefetch(2)
        will prefetch 2 elements (2 batches, of 10 samples each). Furthermore,
        typically it is most useful to add a small prefetch buffer i.e., perhaps
        just a single element at the end of the pipeline, but more complex 
        pipelines can benefits from additional prefetching, especially when
        the time to produce a single element can vary.
    Args:
        features ():
        label ():
        epochs (int): 
        batch_size (int): 
        drop_remainder (bool, optional): 
            [description]. Defaults to False.
        shuffle (bool, optional): 
            [description]. Defaults to True.
        shuffle_buffer_size (int, optional): 
            represent the number of elements from this dataset from which the 
            new dataset will sample. Defaults to 100000.
        reshuffle_each_iteration (bool, optional): 
            A boolean, which if `true` indicates that the dataset should be 
            pseudorandomly reshuffled each time it is iterated over.
        repeat_size (int, optional): 
            [description]. Defaults to None.
        prefetch_buffer_size ([type], optional): 
            [description]. Defaults to None.
    """

    def __init__(self,
                 epochs: int,
                 batch_size: int,
                 drop_remainder=False,
                 shuffle=True,
                 shuffle_buffer_size=100000,
                 reshuffle_each_iteration=False,
                 repeat_size=None,
                 prefetch_buffer_size=None,
                 **kwargs):
        self.epochs = epochs
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.reshuffle_each_iteration = reshuffle_each_iteration

        if repeat_size is not None:
            self.repeat = True
            self.repeat_size = repeat_size
        else:
            self.repeat = False

        if prefetch_buffer_size is not None:
            self.prefetch_buffer_size = prefetch_buffer_size
        else:
            self.prefetch_buffer_size = 2

    def build(self): pass

    def _get_inputs_spec(self): pass

    def _validate_inputs_spec(self): pass

    def _validate_outputs_spec(self): pass

    def _check_tf_cardinality(self, dataset):

        cardinality = dataset.cardinality()

        if (cardinality == tf.data.UNKNOWN_CARDINALITY).numpy():
            pass
        elif (cardinality == tf.data.INFINITE_CARDINALITY).numpy():
            pass

    def _process_inputs(self): pass

    def fit(self, features, label):
        self.features = features
        self.label = label

        """construct tensorflow dataset from numpy array."""
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.features, self.label))
        train_dataset = train_dataset.shuffle(
            buffer_size=self.shuffle_buffer_size,
            reshuffle_each_iteration=self.shuffle_buffer_size)
        train_dataset = train_dataset.batch(
            batch_size=self.batch_size,
            drop_remainder=self.drop_remainder)
        """
        train_dataset = train_dataset.repeat(
            count=self.repeat_size)
        """
        train_dataset = train_dataset.prefetch(
            buffer_size=self.prefetch_buffer_size)

        return train_dataset