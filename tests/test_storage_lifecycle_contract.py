"""Storage contract campaign: real processes and synthetic archives only."""
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import pytest
from urdyn import Urdyn
from urdyn._errors import UrdynStorageError, UrdynError
from urdyn.backup import create_backup, restore_backup, recover_interrupted_restore


def digest(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def child(code, *args):
    return subprocess.Popen([sys.executable, '-c', code, *map(str,args)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def finish(process, expected=0):
    stdout,stderr=process.communicate(timeout=20)
    assert process.returncode==expected, stderr.decode()
    return stdout


@pytest.fixture
def brain(tmp_path):
    root=tmp_path/'brain-default'
    engine=Urdyn.create(root)
    engine.remember('Original café memory')
    return root,engine


def test_missing_rejected_on_existing_handle_for_read_and_write(brain):
    root,e=brain; (root/'.urdyn/memory.db').unlink(); before=digest(root)
    for call in [lambda:e.state(),lambda:e.recall('café'),lambda:e.remember('new'),lambda:e.context('café')]:
        with pytest.raises(UrdynStorageError):call()
        assert digest(root)==before


def test_existing_valid_open_is_unchanged(brain):
    root,e=brain; before=digest(root)
    assert Urdyn.open(root).recall('café')[0].content=='Original café memory'
    assert digest(root)==before


def test_manifest_only_not_initialized(brain):
    root,e=brain; (root/'.urdyn/memory.db').unlink();before=digest(root)
    with pytest.raises(UrdynStorageError):Urdyn.init(root)
    assert digest(root)==before


def test_permission_failure_preserves_bytes(brain):
    root,e=brain;db=root/'.urdyn/memory.db';before=digest(root);db.chmod(0)
    try:
        with pytest.raises(UrdynStorageError):Urdyn.open(root)
    finally:db.chmod(0o600)
    assert digest(root)==before


def test_concurrent_create(tmp_path):
    root=tmp_path/'race'
    code="from urdyn import Urdyn; import sys; Urdyn.create(sys.argv[1])"
    jobs=[child(code,root) for _ in range(2)]
    results=[j.communicate(timeout=20) for j in jobs]
    assert sorted(j.returncode for j in jobs)==[0,1]
    assert Urdyn.open(root).state()==[]


def test_threads_reads_and_writes(brain):
    root,e=brain
    def work(i):
        e.remember(f'Concurrent memory {i}');return len(e.state())
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(work,range(12)))
    assert len(e.state())==13


def test_process_writes(brain):
    root,e=brain
    code="from urdyn import Urdyn; import sys; e=Urdyn.open(sys.argv[1]); [e.remember(sys.argv[2]+str(i)) for i in range(8)]"
    jobs=[child(code,root,prefix) for prefix in ('alpha','beta')]
    for j in jobs:finish(j)
    assert len(e.state())==17


@pytest.mark.parametrize('after',[False,True])
def test_create_crash_before_after_publish(tmp_path,after):
    root=tmp_path/'crash'
    code='''import sys,os
from urdyn import Urdyn
import urdyn._workspace as w
original=w.os.rename
def crash(a,b):
 if sys.argv[2]=='True': original(a,b)
 os._exit(71)
w.os.rename=crash
Urdyn.create(sys.argv[1])
'''
    finish(child(code,root,after),71)
    if after:assert Urdyn.open(root).state()==[]
    else:
        assert (root/'.urdyn.creating/memory.db').exists()
        with pytest.raises(UrdynError):Urdyn.init(root)
        assert Urdyn.recover_creation(root).state()==[]


@pytest.mark.parametrize('step',[1,2])
def test_restore_crash_activation_recovery(brain,tmp_path,step):
    root,e=brain;archive=tmp_path/'backup';create_backup(root,archive);e.remember('Later')
    code='''import os,sys
import urdyn.backup as b
original=b._replace
count=0
def crash(a,c):
 global count
 original(a,c); count+=1
 if count==int(sys.argv[3]):os._exit(72)
b._replace=crash
b.restore_backup(sys.argv[1],sys.argv[2])
'''
    finish(child(code,root,archive,step),72)
    recover_interrupted_restore(root)
    expected=2 if step==1 else 1
    assert len(Urdyn.open(root).state())==expected
    assert Urdyn.open(root).recall('café')


def test_restore_rollback_on_activation_failure(brain,tmp_path):
    import urdyn.backup as b
    root,e=brain;archive=tmp_path/'backup';create_backup(root,archive);e.remember('Later');before=digest(root)
    original=b._validate_urdyn
    def fail(path):
        if Path(path)==root:raise RuntimeError('injected activation failure')
        return original(path)
    with patch.object(b,'_validate_urdyn',fail):
        with pytest.raises(RuntimeError):restore_backup(root,archive)
    assert digest(root)==before
    assert len(e.state())==2


def test_writer_waits_for_restore_lock(brain,tmp_path):
    import select
    import urdyn.backup as b
    root,e=brain;archive=tmp_path/'backup';create_backup(root,archive)
    code="""import sys
import urdyn._lifecycle as l
from urdyn import Urdyn
original=l.fcntl.flock
def waiting(fd, flags):
 print('at-lock', flush=True)
 return original(fd, flags)
l.fcntl.flock=waiting
Urdyn.open(sys.argv[1]).remember('After restore')
"""
    original = b._replace
    writers = []
    def pause(source, target):
        if source == root:
            writer=child(code,root); writers.append(writer)
            assert select.select([writer.stdout], [], [], 10)[0]
            assert writer.stdout.readline() == b'at-lock\n'
            assert writer.poll() is None
        return original(source,target)
    with patch.object(b, '_replace', pause):
        restore_backup(root,archive)
    for writer in writers: finish(writer)
    assert len(Urdyn.open(root).state())==2


def test_crash_during_restore_staging_preserves_original(brain,tmp_path):
    root,e=brain;archive=tmp_path/'backup';create_backup(root,archive);e.remember('Later');before=digest(root)
    code="""import os,sys
import urdyn.backup as b
original=b._validate_urdyn
def crash(path):
 if str(path).endswith('.stage'):os._exit(73)
 return original(path)
b._validate_urdyn=crash
b.restore_backup(sys.argv[1],sys.argv[2])
"""
    finish(child(code,root,archive),73)
    recover_interrupted_restore(root)
    assert digest(root)==before
    assert (tmp_path/'.brain-default.stage').exists()


def test_incomplete_create_not_recovered(tmp_path):
    root=tmp_path/'unfinished';stage=root/'.urdyn.creating';stage.mkdir(parents=True)
    (stage/'manifest.json').write_text('{}')
    before=digest(root)
    with pytest.raises(UrdynError):Urdyn.recover_creation(root)
    assert digest(root)==before


def test_init_does_not_create_over_interrupted_restore(brain,tmp_path):
    root,e=brain;before=digest(root);root.rename(tmp_path/'.brain-default.rollback')
    with pytest.raises(UrdynError):Urdyn.init(root)
    assert not root.exists()
    recover_interrupted_restore(root)
    assert digest(root)==before


def test_m07_manifest_only_archive_explicit_restore(brain,tmp_path):
    import json,zipfile
    root,e=brain;archive=tmp_path/'full';legacy=tmp_path/'legacy'
    create_backup(root,archive)
    with zipfile.ZipFile(archive) as source,zipfile.ZipFile(legacy,'w') as target:
        metadata=json.loads(source.read('backup.json'))
        del metadata['files']['.urdyn/memory.db']
        target.writestr('backup.json',json.dumps(metadata))
        target.writestr('.urdyn/manifest.json',source.read('.urdyn/manifest.json'))
    restore_backup(root,legacy)
    assert Urdyn.open(root).state()==[]
    assert Urdyn.open(tmp_path/'.brain-default.rollback').recall('café')


def test_fork_does_not_inherit_reentrant_permission(brain):
    import select
    from urdyn._lifecycle import lifecycle
    root,e=brain
    reader,writer=os.pipe()
    with lifecycle(root):
        pid=os.fork()
        if pid==0:
            os.close(reader)
            try:
                Urdyn.open(root).remember('forked writer')
                os.write(writer,b'ok')
                os._exit(0)
            except BaseException:os._exit(1)
        os.close(writer)
        assert not select.select([reader],[],[],0.15)[0]
    assert select.select([reader],[],[],10)[0]
    assert os.read(reader,2)==b'ok'
    os.close(reader)
    assert os.waitpid(pid,0)[1]==0
    assert len(e.state())==2


def test_export_loser_preserves_winner(brain, tmp_path, monkeypatch):
    import urdyn.backup as b
    root, _ = brain
    output = tmp_path / 'winner.zip'
    original = b.zipfile.ZipFile
    def competing(path, mode='r', *args, **kwargs):
        if Path(path) == output and mode == 'x':
            output.write_bytes(b'winner')
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(b.zipfile, 'ZipFile', competing)
    with pytest.raises(FileExistsError):
        create_backup(root, output)
    assert output.read_bytes() == b'winner'


def test_stage_mkdir_loser_preserves_owner(brain, tmp_path, monkeypatch):
    root, _ = brain
    output = tmp_path / 'backup.zip'
    create_backup(root, output)
    stage = tmp_path / '.brain-default.stage'
    original = Path.mkdir
    def competing(path, *args, **kwargs):
        if path == stage:
            original(path)
            (path / 'owner').write_bytes(b'winner')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'mkdir', competing)
    with pytest.raises(FileExistsError):
        restore_backup(root, output)
    assert (stage / 'owner').read_bytes() == b'winner'


def test_multilock_global_order_and_inversion(tmp_path):
    from urdyn._lifecycle import lifecycle, lifecycle_many, _local
    a, z = tmp_path/'a', tmp_path/'z'
    with lifecycle_many([z,a]):
        assert _local.held == {str(a), str(z)}
        with lifecycle_many([a,z]): pass
    with lifecycle(z):
        with pytest.raises(UrdynStorageError, match='order'):
            with lifecycle_many([a,z]): pass


@pytest.mark.parametrize('action', ['open', 'close'])
def test_fork_descriptor_critical_window(tmp_path, monkeypatch, action):
    import threading
    import urdyn._lifecycle as l
    entered, release, fork_started = threading.Event(), threading.Event(), threading.Event()
    original = getattr(l.os, action)
    def paused(*args, **kwargs):
        value = original(*args, **kwargs)
        if threading.current_thread().name == 'descriptor-worker':
            entered.set()
            assert release.wait(10)
        return value
    monkeypatch.setattr(l.os, action, paused)
    def worker():
        with l.lifecycle(tmp_path/'fork'): pass
    thread = threading.Thread(target=worker, name='descriptor-worker')
    thread.start(); assert entered.wait(10)
    children = []
    def fork():
        fork_started.set()
        pid = os.fork()
        if pid == 0:
            os._exit(0 if not l._fds else 3)
        children.append(pid)
    forker = threading.Thread(target=fork)
    forker.start(); assert fork_started.wait(10)
    release.set()
    thread.join(10); forker.join(10)
    assert not thread.is_alive() and not forker.is_alive()
    assert len(children) == 1
    assert os.waitpid(children[0], 0)[1] == 0
    assert not l._fds


def test_stale_profile_and_symlinks(brain, tmp_path):
    import json
    root, engine = brain
    manifest = root/'.urdyn/manifest.json'
    data = json.loads(manifest.read_text()); data['profile'] = 'dev'
    manifest.write_text(json.dumps(data))
    with pytest.raises(UrdynStorageError): engine.state()
    db = root/'.urdyn/memory.db'
    target = tmp_path/'external.db'; db.rename(target); db.symlink_to(target)
    before = target.read_bytes()
    with pytest.raises(UrdynStorageError): Urdyn.open(root)
    assert target.read_bytes() == before


def test_missing_column_rejected_without_repair(brain):
    import sqlite3
    root, engine = brain
    with sqlite3.connect(root/'.urdyn/memory.db') as conn:
        conn.execute('ALTER TABLE memories RENAME COLUMN content TO missing_content')
    before=digest(root)
    with pytest.raises(UrdynStorageError): Urdyn.open(root)
    with pytest.raises(UrdynStorageError): engine.remember('refuse')
    assert digest(root)==before


@pytest.mark.parametrize('step', range(1, 6))
def test_restore_directory_sync_faults_preserve_original(brain, tmp_path, monkeypatch, step):
    import urdyn.backup as b
    root, engine=brain; archive=tmp_path/'snapshot'
    create_backup(root, archive); engine.remember('Later retained original')
    before=digest(root); original=b.sync_directory; calls=0
    def fail(path):
        nonlocal calls
        calls += 1
        if calls == step: raise OSError('injected directory fsync failure')
        return original(path)
    monkeypatch.setattr(b,'sync_directory',fail)
    try: restore_backup(root, archive)
    except OSError: pass
    rollback=tmp_path/'.brain-default.rollback'
    assert any(candidate.exists() and digest(candidate)==before for candidate in (root,rollback))


@pytest.mark.parametrize('step', [1,2,3])
def test_create_directory_sync_faults_recover_same_identity(tmp_path, monkeypatch, step):
    import urdyn._workspace as w
    root=tmp_path/'create-fault'; original=w.sync_directory; calls=0
    def fail(path):
        nonlocal calls
        calls += 1
        if calls == step: raise OSError('injected directory fsync failure')
        return original(path)
    monkeypatch.setattr(w,'sync_directory',fail)
    with pytest.raises(OSError): Urdyn.create(root)
    monkeypatch.setattr(w,'sync_directory',original)
    stage=root/'.urdyn.creating'
    if stage.exists():
        before=(stage/'manifest.json').read_bytes()
        Urdyn.recover_creation(root)
        assert (root/'.urdyn/manifest.json').read_bytes()==before
    elif (root/'.urdyn').exists(): assert Urdyn.open(root).state()==[]
    else: assert Urdyn.init(root).state()==[]


def test_concurrent_multilock_overlapping_locations(tmp_path):
    import threading
    from urdyn._lifecycle import lifecycle_many
    start=threading.Barrier(2)
    paths=[tmp_path/'brain', tmp_path/'.brain.rollback', tmp_path/'.brain.stage']
    def work(keys):
        start.wait(timeout=10)
        with lifecycle_many(keys):
            assert all(not p.exists() for p in paths)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(work, keys) for keys in (paths,list(reversed(paths)))]
        for result in futures: result.result(timeout=10)


@pytest.mark.parametrize('after',[False, True])
def test_create_process_dies_at_schema(brain, tmp_path, after):
    root=tmp_path/'schema-crash'
    code='''import os,sys
from urdyn import Urdyn
from urdyn._store import MemoryStore
original=MemoryStore.create_new
@classmethod
def crash(cls,path):
 if sys.argv[2]=='True':
  with original(path):pass
 os._exit(74)
MemoryStore.create_new=crash
Urdyn.create(sys.argv[1])
'''
    finish(child(code,root,after),74)
    before=digest(root)
    if after:
        identity=(root/'.urdyn.creating/manifest.json').read_bytes()
        Urdyn.recover_creation(root)
        assert (root/'.urdyn/manifest.json').read_bytes()==identity
    else:
        with pytest.raises(UrdynError): Urdyn.recover_creation(root)
        assert digest(root)==before


@pytest.mark.parametrize('location',['root','parent'])
def test_real_permission_denied_lifecycle(brain, tmp_path, location):
    root,engine=brain; before=digest(root)
    target=root if location=='root' else tmp_path
    old=target.stat().st_mode & 0o777
    target.chmod(0)
    try:
        with pytest.raises(UrdynError): Urdyn.open(root)
    finally: target.chmod(old)
    assert digest(root)==before


def test_process_exports_one_valid_winner(tmp_path):
    from urdyn.backup import inspect_backup
    roots=[tmp_path/'one',tmp_path/'two']; output=tmp_path/'shared.zip'
    for i,root in enumerate(roots): Urdyn.create(root).remember(f'Owned record {i}')
    code='''import sys
from urdyn.backup import create_backup
create_backup(sys.argv[1], sys.argv[2])
'''
    jobs=[child(code,root,output) for root in roots]
    for job in jobs: job.communicate(timeout=20)
    assert sorted(job.returncode for job in jobs)==[0,1]
    before=output.read_bytes()
    inspect_backup(output)
    assert output.read_bytes()==before


def test_admitted_parent_alias_shares_location(brain, tmp_path):
    from urdyn._lifecycle import normalize_location
    root,engine=brain; alias=tmp_path/'alias';alias.symlink_to(tmp_path,target_is_directory=True)
    other=alias/root.name
    assert normalize_location(root)==normalize_location(other)
    Urdyn.open(other).remember('Alias record')
    assert len(engine.state())==2


def test_overlapping_restores_preserve_rollback(brain, tmp_path):
    import threading
    from urdyn.backup import BackupError
    root,engine=brain; archive=tmp_path/'archive';create_backup(root,archive)
    engine.remember('Original before overlapping restore'); before=digest(root)
    barrier=threading.Barrier(2)
    def run():
        barrier.wait(timeout=10)
        try: restore_backup(root,archive); return 'ok'
        except BackupError: return 'refused'
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(run) for _ in range(2)]
        assert sorted(job.result(timeout=20) for job in jobs)==['ok','refused']
    assert digest(tmp_path/'.brain-default.rollback')==before
    assert len(Urdyn.open(root).state())==1


@pytest.mark.parametrize('step', range(1, 7))
def test_restore_file_and_directory_fsync_failures(brain, tmp_path, monkeypatch, step):
    import urdyn.backup as b
    root,engine=brain;archive=tmp_path/'archive';create_backup(root,archive)
    engine.remember('Original durable memory');before=digest(root)
    original=b.os.fsync;calls=0
    def fail(fd):
        nonlocal calls
        calls+=1
        if calls==step: raise OSError('injected fsync failure')
        return original(fd)
    monkeypatch.setattr(b.os,'fsync',fail)
    with pytest.raises(OSError):restore_backup(root,archive)
    assert calls>=step
    rollback=tmp_path/'.brain-default.rollback'
    assert any(p.exists() and digest(p)==before for p in (root,rollback))
