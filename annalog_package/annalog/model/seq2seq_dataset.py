from torchtext.legacy import data
import os
import io
from torchtext.legacy.datasets.translation import TranslationDataset


class SMILESDataset(TranslationDataset):
    
    @classmethod
    def splits(cls, exts, fields, root='.data',
               train='train', validation='valid', test='test', **kwargs):
        """Create dataset objects for splits of the custom SMILES dataset.

        Args:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        # TODO: This is a _HORRIBLE_ patch related to #208
        # 'path' can be passed as a kwarg to the translation dataset constructor
        # or has to be set (so the download wouldn't be duplicated). A good idea
        # seems to rename the existence check variable from path to something else
        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(SMILESDataset, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)

