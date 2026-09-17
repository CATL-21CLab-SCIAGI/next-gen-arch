import hashlib
from pathlib import Path
import tempfile
import unittest
from archlab.storage.bulk_offload import transfer
from archlab.storage.results_plan import build


class OffloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.source=self.root/'nas/weight';self.source.parent.mkdir();self.source.write_bytes(b'payload');self.target=self.root/'oss/weight'

    def test_copy_then_link_and_resume(self):
        result=transfer(self.source,self.target);self.assertFalse(self.source.is_symlink());self.assertEqual(result['sha256'],hashlib.sha256(b'payload').hexdigest())
        transfer(self.source,self.target,link=True);self.assertTrue(self.source.is_symlink());self.assertEqual(self.source.read_bytes(),b'payload')
        self.assertTrue(transfer(self.source,self.target,link=True)['linked'])

    def test_same_path_cannot_destroy_source(self):
        with self.assertRaises(ValueError):transfer(self.source,self.source,link=True)
        self.assertFalse(self.source.is_symlink());self.assertEqual(self.source.read_bytes(),b'payload')

    def test_destination_collision_is_preserved(self):
        self.target.parent.mkdir();self.target.write_bytes(b'changed')
        with self.assertRaises(ValueError):transfer(self.source,self.target,link=True)
        self.assertEqual(self.target.read_bytes(),b'changed');self.assertFalse(self.source.is_symlink())

    def test_sealed_digest_mismatch_never_publishes(self):
        with self.assertRaises(ValueError):transfer(self.source,self.target,link=True,expected_sha='0'*64)
        self.assertFalse(self.source.is_symlink());self.assertFalse(self.target.exists())

    def test_selector_preserves_active_saves_and_source_repos(self):
        src=self.root/'results';dst=self.root/'outputs';src.mkdir();dst.mkdir()
        for name in ['ready/checkpoints/step-000001','active/checkpoints/step-000002','source/.git']:(src/name).mkdir(parents=True)
        (src/'ready/checkpoints/step-000001/COMPLETE.json').write_text('{}')
        for name in ['ready/checkpoints/step-000001/weight.pt','active/checkpoints/step-000002/weight.pt','source/weight.pt']:(src/name).write_bytes(b'1234')
        tasks,summary=build(src,dst,minimum_bytes=1);self.assertEqual(len(tasks),1);self.assertIn('/ready/',tasks[0]['source']);self.assertEqual(len(summary['incomplete_checkpoints_skipped']),1);self.assertEqual(summary['source_repositories_skipped'],1)


if __name__=='__main__':unittest.main()
