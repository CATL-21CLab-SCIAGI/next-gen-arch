import tempfile
from pathlib import Path
import unittest
from archlab.storage.model_migration import prepare, commit


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.source, self.destination = root / 'source', root / 'destination'
        self.source.mkdir()
        self.destination.mkdir()
        self.journal = root / 'journal.json'
        (self.source / 'model').mkdir()
        (self.source / 'model/weight').write_bytes(b'1234')
        (self.destination / 'existing').write_text('keep')

    def test_verified_copy_and_link_preserve_destination(self):
        prepare(self.source, self.destination, self.journal)
        self.assertFalse(self.source.is_symlink())
        commit(self.journal)
        self.assertTrue(self.source.is_symlink())
        self.assertEqual((self.source / 'model/weight').read_bytes(), b'1234')
        self.assertEqual((self.destination / 'existing').read_text(), 'keep')

    def test_existing_different_object_is_not_overwritten(self):
        (self.destination / 'model').mkdir()
        (self.destination / 'model/weight').write_bytes(b'bad!')
        with self.assertRaises(ValueError):
            prepare(self.source, self.destination, self.journal)
        self.assertEqual((self.destination / 'model/weight').read_bytes(), b'bad!')
        self.assertFalse(self.source.is_symlink())

    def test_changed_source_prevents_commit(self):
        prepare(self.source, self.destination, self.journal)
        (self.source / 'model/new').write_text('new')
        with self.assertRaises(ValueError):
            commit(self.journal)
        self.assertFalse(self.source.is_symlink())

    def test_changed_destination_prevents_commit(self):
        prepare(self.source, self.destination, self.journal)
        (self.destination / 'model/weight').write_bytes(b'bad!')
        with self.assertRaises(ValueError):
            commit(self.journal)
        self.assertEqual((self.source / 'model/weight').read_bytes(), b'1234')

    def test_destination_symlink_prevents_write(self):
        (self.destination / 'model').symlink_to(self.source / 'model', target_is_directory=True)
        with self.assertRaises(ValueError):
            prepare(self.source, self.destination, self.journal)


if __name__ == '__main__':
    unittest.main()
