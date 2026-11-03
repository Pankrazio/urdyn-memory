import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from urdyn import Urdyn
from urdyn._errors import UrdynStorageError

class StorageVerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'brain'
        self.engine = Urdyn.init(self.root)
        self.engine.remember('Synthetic verification fixture')
        self.db = self.root / '.urdyn' / 'memory.db'

    def test_valid_is_read_only(self):
        before = hashlib.sha256(self.db.read_bytes()).digest()
        self.engine.verify_storage()
        self.assertEqual(before, hashlib.sha256(self.db.read_bytes()).digest())

    def test_missing_is_not_recreated(self):
        self.db.unlink()
        with self.assertRaises(UrdynStorageError): self.engine.verify_storage()
        self.assertFalse(self.db.exists())

    def test_corrupt_is_not_repaired(self):
        self.db.write_bytes(b'corrupt')
        with self.assertRaises(UrdynStorageError): self.engine.verify_storage()
        self.assertEqual(self.db.read_bytes(), b'corrupt')

    def test_directory_is_typed_error(self):
        self.db.unlink(); self.db.mkdir()
        with self.assertRaises(UrdynStorageError): self.engine.verify_storage()

    def test_missing_canonical_table_is_rejected(self):
        with sqlite3.connect(self.db) as conn: conn.execute('DROP TABLE source_observations')
        with self.assertRaises(UrdynStorageError): self.engine.verify_storage()

    def test_legacy_empty_is_explicitly_ambiguous(self):
        empty = Urdyn.init(Path(self.tmp.name) / 'empty')
        (empty.path / ".urdyn" / "memory.db").unlink()
        with self.assertRaises(UrdynStorageError): empty.verify_storage()

    def test_unknown_schema_is_not_migrated(self):
        with sqlite3.connect(self.db) as conn: conn.execute('PRAGMA user_version=999')
        with self.assertRaises(UrdynStorageError): self.engine.verify_storage()
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute('PRAGMA user_version').fetchone()[0],999)
